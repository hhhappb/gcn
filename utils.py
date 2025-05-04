# 文件名: utils.py
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

logger = logging.getLogger(__name__) # Logger for utils

# --- 随机种子初始化 ---
def init_seed(seed):
    """
    初始化各种库的随机种子，以确保实验的可复现性。
    Args:
        seed (int): 要设置的随机种子。
    """
    if seed is None: return
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 动态类导入 ---
def import_class(import_str):
    """
    根据字符串动态导入 Python 类或模块。
    """
    mod_str, _sep, class_str = import_str.rpartition('.')
    if not mod_str:
        mod_str, class_str = class_str, None
    try:
        __import__(mod_str)
        module = sys.modules[mod_str]
        imported_obj = module if class_str is None else getattr(module, class_str)
        if class_str and not (inspect.isclass(imported_obj) or inspect.isfunction(imported_obj)):
             raise ImportError(f"'{import_str}' 导入成功，但不是类或函数。")
        return imported_obj
    except ModuleNotFoundError:
         raise ImportError(f"无法找到模块 '{mod_str}'。请检查路径和 PYTHONPATH。")
    except AttributeError:
        raise ImportError(f"在模块 '{mod_str}' 中找不到属性 '{class_str}'。")
    except Exception as e:
        raise ImportError(f"导入 '{import_str}' 时发生错误: {e}\n{traceback.format_exc()}")

# --- 字符串转布尔值 ---
def str2bool(v):
    """
    将常见的表示布尔值的字符串转换为 Python 布尔值。
    """
    if isinstance(v, bool): return v
    low_v = str(v).lower()
    if low_v in ('yes', 'true', 't', 'y', '1'): return True
    elif low_v in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError(f'不支持的布尔值: {v}')

# --- DictAction 类 ---
class DictAction(argparse.Action):
    """
    argparse Action 用于解析字典格式的命令行参数。
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None: raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            import ast; parsed_dict = ast.literal_eval(values)
            if not isinstance(parsed_dict, dict): raise ValueError("输入必须是字典格式")
            setattr(namespace, self.dest, parsed_dict)
        except (ValueError, SyntaxError):
            try:
                parsed_dict = {}; parts = values.split(',')
                for item in parts:
                    key_value = item.split('=', 1)
                    if len(key_value) == 2:
                        key, value_str = key_value; value_str = value_str.strip()
                        try: value = ast.literal_eval(value_str)
                        except (ValueError, SyntaxError): value = value_str
                        parsed_dict[key.strip()] = value
                    else:
                        print(f"警告: 忽略无法解析的 key=value 项: {item}")
                setattr(namespace, self.dest, parsed_dict)
                print(f"警告: 解析参数 '{values}' 为 key=value 对。如果这不是预期行为，请使用标准的 Python 字典格式 '{{key:value,...}}'")
            except Exception as e:
                raise argparse.ArgumentTypeError(f"无法将 '{values}' 解析为字典: {e}")

# --- 自定义标签平滑交叉熵损失 ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失函数。
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing 值必须在 [0, 1) 范围内"
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# --- 自定义 Collate 函数 ---
def collate_fn_filter_none(batch):
    """
    自定义的 DataLoader collate_fn，过滤掉返回 None 的样本。
    """
    original_len = len(batch)
    batch = [item for item in batch if item is not None and all(sub_item is not None for sub_item in item)]
    filtered_len = len(batch)
    if original_len > filtered_len:
        logger.warning(f"Collate: 过滤掉 {original_len - filtered_len} 个无效样本。")
    if not batch:
        logger.warning("Collate: 整个批次无效，返回 None。")
        return None
    try:
        return default_collate(batch)
    except RuntimeError as e:
        logger.error(f"错误: default_collate 失败: {e}")
        for i, item in enumerate(batch):
            if isinstance(item, (list, tuple)):
                shapes = [t.shape if hasattr(t, 'shape') else type(t) for t in item]
                logger.error(f" Collate 错误样本 {i} 形状: {shapes}")
        return None
    except Exception as e:
        logger.error(f"错误: collate_fn 发生未知错误: {e}")
        return None