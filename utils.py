# 文件名: utils.py (8.3)
import torch
import numpy as np
import random
import argparse
import sys
import inspect
import traceback
import logging
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F
import os

# --- 日志记录器 ---
logger_utils = logging.getLogger(__name__)

def get_logger(log_dir, name='default_log', level=logging.INFO, print_to_console=True, save_to_file=True):
    logger = logging.getLogger(name)
    # 检查是否已经有 handlers，避免重复添加，特别是在Jupyter等环境中
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        if print_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if save_to_file:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, f"{name}.log")
            file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.propagate = False
    return logger

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = True

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = False

# --- 动态类导入 ---
def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    if not mod_str:
        if _sep == '':
            mod_str, class_str = import_str, None
        else: # e.g. ".MyClass" - this case is less common and __import__ might behave differently
              # For simplicity, we assume absolute or top-level module imports.
             raise ImportError(f"不明确的导入字符串 '{import_str}'. 期望格式 'package.module.ClassName' 或 'module.ClassName'.")

    try:
        if not mod_str and class_str: # Trying to import a class from current/top level without module path
            # This is tricky and depends on where things are.
            # A more robust way for project structure is to always have module paths.
            # However, if class_str is a module name (because original import_str was single word)
            if class_str in sys.modules:
                return sys.modules[class_str] # It was already imported as a module
            else: # Try to import it as a module
                 __import__(class_str)
                 return sys.modules[class_str]


        imported_module = __import__(mod_str, fromlist=[class_str] if class_str else [])

        if class_str is None:
            return imported_module # Import the module itself

        imported_obj = getattr(imported_module, class_str)

        if not (inspect.isclass(imported_obj) or inspect.isfunction(imported_obj)):
             logger_utils.warning(f"'{import_str}' 导入成功，但目标 '{class_str}' 不是类或函数。类型: {type(imported_obj)}")
        return imported_obj
    except ModuleNotFoundError:
         logger_utils.error(f"无法找到模块 '{mod_str}' (从 '{import_str}' 解析得到)。请检查路径和 PYTHONPATH。")
         logger_utils.debug(f"当前 sys.path (部分): {sys.path[:5]} ... {sys.path[-5:]}")
         raise
    except AttributeError:
        logger_utils.error(f"在模块 '{mod_str}' 中找不到属性 '{class_str}' (从 '{import_str}' 解析得到)。")
        raise
    except Exception as e:
        logger_utils.error(f"导入 '{import_str}' 时发生其他错误: {e}\n{traceback.format_exc()}")
        raise

# --- 字符串转布尔值 ---
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'布尔值期望得到 (yes,true,t,y,1 or no,false,f,n,0)，但传入 {v}')

# --- DictAction 类 (argparse) ---
class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed for DictAction")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parsed_dict = {}
        if not values:
            setattr(namespace, self.dest, parsed_dict)
            return
        try:
            import ast
            parsed_dict = ast.literal_eval(values)
            if not isinstance(parsed_dict, dict):
                raise ValueError("输入不是有效的字典字面量格式")
        except (ValueError, SyntaxError):
            parsed_dict = {}
            try:
                pairs = values.split(',')
                for pair in pairs:
                    if not pair.strip(): continue
                    kv = pair.split('=', 1)
                    if len(kv) == 2:
                        key = kv[0].strip()
                        value_str = kv[1].strip()
                        try: value = ast.literal_eval(value_str)
                        except (ValueError, SyntaxError): value = value_str
                        parsed_dict[key] = value
            except Exception as e_kv:
                raise argparse.ArgumentTypeError(
                    f"无法将 '{values}' 解析为字典: {e_kv}"
                ) from e_kv
        setattr(namespace, self.dest, parsed_dict)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# --- 自定义 Collate 函数 ---
def collate_fn_filter_none(batch):
    original_len = len(batch)
    valid_batch = []
    for item_idx, item in enumerate(batch):
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            if any(sub_item is None for sub_item in item):
                continue
        valid_batch.append(item)
    batch = valid_batch
    filtered_len = len(batch)

    if original_len > filtered_len:
        logger_utils.warning(f"Collate: 从 {original_len} 个样本中过滤掉 {original_len - filtered_len} 个无效样本 (包含 None)。")
    if not batch:
        logger_utils.warning("Collate: 整个批次无效 (所有样本都被过滤)，返回 None。")
        return None
    try:
        return default_collate(batch)
    except RuntimeError as e:
        logger_utils.error(f"错误: default_collate 失败: {e}")
        for i, item_tuple in enumerate(batch):
            if isinstance(item_tuple, (list, tuple)):
                shapes = [f"[{j}]:{t.shape if hasattr(t, 'shape') else type(t)}" for j, t in enumerate(item_tuple)]
                logger_utils.error(f" Collate 错误样本 {i} (元组) 各元素形状/类型: {shapes}")
            else:
                logger_utils.error(f" Collate 错误样本 {i} 类型: {type(item_tuple)}")
        return None
    except Exception as e_other:
        logger_utils.error(f"错误: collate_fn 发生未知错误: {e_other}", exc_info=True)
        return None

# --- 特征缩放器 (StandardScaler) ---
class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean_ = mean
        self.scale_ = std
        self.epsilon_ = epsilon
        if self.mean_ is not None and not isinstance(self.mean_, torch.Tensor):
            self.mean_ = torch.tensor(self.mean_, dtype=torch.float32)
        if self.scale_ is not None and not isinstance(self.scale_, torch.Tensor):
            self.scale_ = torch.tensor(self.scale_, dtype=torch.float32)

    def fit(self, data: torch.Tensor):
        if data.ndim < 2: raise ValueError("数据至少需要2个维度 (样本数, 特征数)")
        axes_to_reduce = tuple(range(data.ndim - 1))
        self.mean_ = torch.mean(data, dim=axes_to_reduce, keepdim=False)
        self.scale_ = torch.std(data, dim=axes_to_reduce, keepdim=False)
        self.scale_[self.scale_ < self.epsilon_] = self.epsilon_
        # logger_utils.info(f"StandardScaler fitted. Mean shape: {self.mean_.shape}, Std shape: {self.scale_.shape}") # 可以按需开启
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.scale_ is None: raise RuntimeError("Scaler 未被拟合。")
        device = data.device
        dtype = data.dtype
        mean = self.mean_.to(device=device, dtype=dtype)
        scale = self.scale_.to(device=device, dtype=dtype)
        return (data - mean) / scale

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.scale_ is None: raise RuntimeError("Scaler 未被拟合。")
        device = data.device
        dtype = data.dtype
        mean = self.mean_.to(device=device, dtype=dtype)
        scale = self.scale_.to(device=device, dtype=dtype)
        return (data * scale) + mean

    def to(self, device):
        if self.mean_ is not None: self.mean_ = self.mean_.to(device)
        if self.scale_ is not None: self.scale_ = self.scale_.to(device)
        return self

# --- 数据移动到设备 ---
def move2device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, (list, tuple)):
        return [move2device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: move2device(v, device) for k, v in data.items()}
    else:
        return data