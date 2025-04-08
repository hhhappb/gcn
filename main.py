# 文件名: main.py (最终完整版 v9 - 修正数据处理逻辑)
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import inspect
import logging
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate # <--- 导入 default_collate
from tensorboardX import SummaryWriter
from tqdm import tqdm

# --- DictAction 类 ---
class DictAction(argparse.Action):
    # ... (DictAction 定义不变) ...
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
                    else: print(f"警告: 忽略无法解析的 key=value 项: {item}")
                setattr(namespace, self.dest, parsed_dict)
                print(f"警告: 解析参数 '{values}' 为 key=value 对。")
            except Exception as e: raise argparse.ArgumentTypeError(f"无法将 '{values}' 解析为字典: {e}")

# --- 平台特定的文件限制调整 ---
if sys.platform != "win32":
    # ... (resource 模块处理不变) ...
    try:
        import resource; rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_limit = min(4096, rlimit[1]); resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, rlimit[1]))
        print(f"Info: 文件描述符限制已尝试设置为 {target_limit}。(平台: {sys.platform})")
    except Exception as e: print(f"警告: 无法增加 RLIMIT_NOFILE 限制: {e}。(平台: {sys.platform})")
else: print(f"Info: 跳过 RLIMIT_NOFILE 设置。(平台: {sys.platform})")

# --- 随机种子初始化 ---
def init_seed(seed):
    # ... (init_seed 定义不变) ...
    if seed is None: print("Info: 未设置随机种子。"); return
    print(f"设置随机种子为: {seed}")
    torch.cuda.manual_seed_all(seed); torch.manual_seed(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# --- 动态类导入 ---
def import_class(import_str):
    # ... (import_class 定义不变) ...
    mod_str, _sep, class_str = import_str.rpartition('.')
    if not mod_str: mod_str, class_str = class_str, None
    try:
        __import__(mod_str); module = sys.modules[mod_str]
        imported_obj = module if class_str is None else getattr(module, class_str)
        if class_str and not (inspect.isclass(imported_obj) or inspect.isfunction(imported_obj)):
             raise ImportError(f"'{import_str}' 导入成功，但不是类或函数。")
        return imported_obj
    except Exception as e: raise ImportError(f"无法导入 '{import_str}': {e}\n{traceback.format_exc()}")

# --- 字符串转布尔值 ---
def str2bool(v):
    # ... (str2bool 定义不变) ...
    if isinstance(v, bool): return v
    low_v = str(v).lower()
    if low_v in ('yes', 'true', 't', 'y', '1'): return True
    elif low_v in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError(f'不支持的布尔值: {v}')

# --- 参数解析器 ---
def get_parser():
    # ... (get_parser 定义不变) ...
    parser = argparse.ArgumentParser(description='骨骼动作识别训练器 (SDT-GRU & feeder_ucla)')
    parser.add_argument('--work-dir', default='./work_dir/sdtgru_ucla_run', help='工作目录')
    parser.add_argument('--config', default='config/nw_ucla_sdtgru.yaml', help='配置文件路径')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help='运行阶段')
    parser.add_argument('--device', type=int, default=None, nargs='+', help='GPU 索引, 默认自动选择')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--model', default='model.SDT_GRUs_Gesture.SDT_GRU_Classifier', help='模型类导入路径')
    parser.add_argument('--model-args', action=DictAction, default={}, help='模型初始化参数 (覆盖YAML)')
    parser.add_argument('--weights', default=None, help='预训练权重路径')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='加载权重时忽略的层名')
    parser.add_argument('--feeder', default='feeders.feeder_ucla.Feeder', help='数据加载器类导入路径')
    parser.add_argument('--train-feeder-args', action=DictAction, default={}, help='训练数据加载器参数 (覆盖YAML)')
    parser.add_argument('--test-feeder-args', action=DictAction, default={}, help='测试数据加载器参数 (覆盖YAML)')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小 (覆盖YAML)')
    parser.add_argument('--test-batch-size', type=int, default=None, help='测试批次大小 (覆盖YAML)')
    parser.add_argument('--num-epoch', type=int, default=None, help='总训练 epoch 数 (覆盖YAML)')
    parser.add_argument('--start-epoch', type=int, default=0, help='起始 epoch')
    parser.add_argument('--optimizer', default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='优化器类型')
    parser.add_argument('--base-lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--step', type=int, default=[50, 70], nargs='+', help='学习率衰减 epoch 节点')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='学习率预热 epoch 数 (0 禁用)')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='是否使用 Nesterov (SGD)')
    parser.add_argument('--log-interval', type=int, default=50, help='打印/记录 batch 日志间隔')
    parser.add_argument('--eval-interval', type=int, default=1, help='评估模型 epoch 间隔')
    parser.add_argument('--save-interval', type=int, default=10, help='保存 checkpoint epoch 间隔 (0 禁用)')
    parser.add_argument('--save-epoch', type=int, default=0, help='开始保存 checkpoint 的 epoch')
    parser.add_argument('--print-log', type=str2bool, default=True, help='是否打印日志')
    parser.add_argument('--save-score', type=str2bool, default=False, help='是否保存测试分数 (logits)')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='显示哪些 Top K 准确率')
    return parser

# --- 自定义 Collate 函数 ---
def collate_fn_filter_none(batch):
    # ... (collate_fn_filter_none 定义不变) ...
    original_len = len(batch)
    batch = [item for item in batch if item is not None and item[0] is not None]
    filtered_len = len(batch)
    if original_len > filtered_len: pass # print(f"Collate: 过滤掉 {original_len - filtered_len} 个无效样本。")
    if not batch: return None
    try: return default_collate(batch)
    except RuntimeError as e: print(f"错误: default_collate 失败: {e}"); return None

# --- 主处理器类 ---
class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.setup_device()
        self.setup_logging_and_writers()
        self.print_log("Processor 初始化开始...")
        self.save_arg()
        self.global_step = 0
        self.scaler = None
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.lr = self.arg.base_lr
        if self.lr is None: raise ValueError("必须设置 base_lr")
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.model = self.model.to(self.output_device)
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)
            self.print_log(f'模型已在 GPUs {self.arg.device} 上启用 DataParallel。')
        self.print_log("Processor 初始化完成。")

    # ... (setup_device, setup_logging_and_writers, print_log, save_arg 不变) ...
    def setup_device(self):
        """设置运行设备"""
        if self.arg.device is None: self.arg.device = [0] if torch.cuda.is_available() else [-1]; print(f"自动选择设备: {self.arg.device}")
        if self.arg.device[0] == -1 or not torch.cuda.is_available(): self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]; print("将在 CPU 上运行。")
        else:
            if not torch.cuda.is_available(): print("错误: CUDA 不可用。将使用 CPU。"); self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]
            else:
                valid_devices = [d for d in self.arg.device if 0 <= d < torch.cuda.device_count()]
                if not valid_devices: print(f"错误: 无效的 GPU 设备索引 {self.arg.device}。将使用 CPU。"); self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]
                else:
                    self.arg.device = valid_devices; self.output_device = torch.device(f"cuda:{self.arg.device[0]}");
                    try: torch.cuda.set_device(self.output_device); print(f"使用 GPU: {self.arg.device}。主设备: {self.output_device}")
                    except Exception as e: print(f"错误: 设置 CUDA 设备失败: {e}。将使用 CPU。"); self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]

    def setup_logging_and_writers(self):
        """初始化日志和 TensorBoard"""
        if not os.path.exists(self.arg.work_dir): os.makedirs(self.arg.work_dir)
        log_file = os.path.join(self.arg.work_dir, 'log.txt')
        should_clear_log = self.arg.phase == 'train' and self.arg.start_epoch == 0
        filemode = 'w' if should_clear_log else 'a'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file, mode=filemode, encoding='utf-8'),
                                      logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger(__name__)
        self.print_log(f'工作目录: {self.arg.work_dir}')
        if self.arg.phase == 'train':
            try:
                runs_dir = os.path.join(self.arg.work_dir, 'runs');
                if not os.path.exists(runs_dir): os.makedirs(runs_dir)
                self.train_writer = SummaryWriter(os.path.join(runs_dir, 'train'), comment='_train')
                self.val_writer = SummaryWriter(os.path.join(runs_dir, 'val'), comment='_val')
                self.print_log(f'TensorBoard 日志已设置在: {runs_dir}')
            except Exception as e: self.print_log(f"警告: 初始化 TensorBoard失败: {e}", logging.WARNING); self.train_writer = self.val_writer = None

    def print_log(self, msg, level=logging.INFO):
        """打印日志到 logger"""
        if hasattr(self, 'logger') and self.logger and self.arg.print_log: self.logger.log(level, msg)
        elif self.arg.print_log: print(msg)

    def save_arg(self):
        """保存最终使用的参数到 YAML 文件"""
        arg_dict = vars(self.arg); work_dir = arg_dict.get('work_dir', './work_dir/default')
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        try:
            filepath = os.path.join(work_dir, 'config_used.yaml')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Command line: {' '.join(sys.argv)}\n"); f.write(f"# Python: {sys.version.splitlines()[0]}\n")
                f.write(f"# PyTorch: {torch.__version__}\n"); f.write(f"# CUDA Available: {torch.cuda.is_available()}\n")
                if torch.cuda.is_available():
                    f.write(f"# CUDA Version: {torch.version.cuda}\n"); f.write(f"# CuDNN Version: {torch.backends.cudnn.version()}\n")
                    try: f.write(f"# GPU: {torch.cuda.get_device_name(self.output_device)}\n")
                    except: pass
                f.write("\n"); yaml.dump(arg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True, Dumper=Dumper)
            self.print_log(f"最终配置已保存到: {filepath}")
        except Exception as e: print(f"警告: 保存 config_used.yaml 失败: {e}")

    def load_data(self):
        """加载数据加载器 (适配 feeder_ucla, 移除 Scaler, 使用顶层 batch size, 添加 collate_fn)"""
        self.print_log("开始加载数据...")
        self.scaler = None
        try:
            if not self.arg.feeder: raise ValueError("'feeder' 参数未设置。")
            Feeder = import_class(self.arg.feeder)
        except (ImportError, ValueError) as e: self.print_log(f"错误: 无法导入或 Feeder 未设置 '{self.arg.feeder}'.", logging.ERROR); raise e
        self.data_loader = dict()
        try:
            train_batch_size = self.arg.batch_size
            test_batch_size = self.arg.test_batch_size
            if train_batch_size is None: raise ValueError("必须设置 batch_size")
            if test_batch_size is None: raise ValueError("必须设置 test_batch_size")
            num_worker = self.arg.num_worker if self.arg.num_worker is not None else 4

            if self.arg.phase == 'train':
                feeder_args = self.arg.train_feeder_args.copy()
                feeder_args['split'] = 'train' # 明确设置 split
                self.print_log(f"训练 Feeder 参数: {feeder_args}")
                train_dataset = Feeder(**feeder_args)
                self.data_loader['train'] = DataLoader(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=num_worker, drop_last=getattr(self.arg, 'drop_last', True), # drop_last 从顶层 arg 读取
                    worker_init_fn=init_seed, pin_memory=True,
                    collate_fn=collate_fn_filter_none # <--- 使用自定义 collate_fn
                )
                self.print_log(f"训练数据加载器 '{self.arg.feeder}' 加载成功。样本数: {len(train_dataset)}")

            feeder_args = self.arg.test_feeder_args.copy()
            feeder_args['split'] = 'val' # 明确设置 split 为 'val'
            self.print_log(f"测试/验证 Feeder 参数: {feeder_args}")
            test_dataset = Feeder(**feeder_args)
            self.data_loader['val'] = DataLoader( # 使用 'val' 作为键名
                test_dataset, batch_size=test_batch_size, shuffle=False,
                num_workers=num_worker, drop_last=False, worker_init_fn=init_seed, pin_memory=True,
                collate_fn=collate_fn_filter_none # <--- 使用自定义 collate_fn
            )
            self.data_loader['test'] = self.data_loader['val'] # 测试和验证使用相同加载器
            self.print_log(f"测试/验证数据加载器 '{self.arg.feeder}' 加载成功。样本数: {len(test_dataset)}")
        except Exception as e: self.print_log(f"错误: 加载数据失败: {e}", logging.ERROR); traceback.print_exc(); raise e
        self.print_log("数据加载完成。")

    def load_model(self):
        """加载模型结构和权重"""
        self.print_log(f"模型将运行在设备: {self.output_device}")
        try:
            if not self.arg.model: raise ValueError("'model' 参数未设置。")
            Model = import_class(self.arg.model)
            # 尝试复制模型文件
            try: model_file_path = inspect.getfile(Model); shutil.copy2(model_file_path, self.arg.work_dir)
            except Exception as e: self.print_log(f"警告: 复制模型文件失败: {e}", logging.WARNING)
            # 实例化模型
            if not self.arg.model_args: raise ValueError("'model_args' 参数未设置或为空。")
            # --- 使用 model_cfg=... 的方式实例化 ---
            self.model = Model(model_cfg=self.arg.model_args)
            self.print_log(f"模型 '{self.arg.model}' 实例化成功。")
        except (ImportError, ValueError, Exception) as e: self.print_log(f"错误: 模型加载/实例化失败: {e}", logging.ERROR); traceback.print_exc(); raise e

        self.loss = nn.CrossEntropyLoss().to(self.output_device)
        self.print_log(f"损失函数: CrossEntropyLoss")

        # 加载预训练权重
        if self.arg.weights:
            self.print_log(f'加载权重自: {self.arg.weights}')
            if not os.path.exists(self.arg.weights): self.print_log(f"错误: 权重文件不存在: {self.arg.weights}", logging.ERROR); raise FileNotFoundError()
            try:
                weights = torch.load(self.arg.weights, map_location=self.output_device)
                weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
                # ... (忽略权重的逻辑) ...
                if self.arg.ignore_weights:
                     keys = list(weights.keys())
                     for w_name in self.arg.ignore_weights:
                         removed_keys = [k for k in keys if w_name in k]
                         if not removed_keys: self.print_log(f'警告: 未找到要忽略的权重: {w_name}', logging.WARNING)
                         for key in removed_keys:
                             if weights.pop(key, None) is not None: self.print_log(f'移除权重: {key}')
                # 加载 state_dict
                missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
                if missing_keys: self.print_log(f"警告: 加载权重时模型中缺失的键: {missing_keys}", logging.WARNING)
                if unexpected_keys: self.print_log(f"警告: 加载权重时权重文件中多余的键: {unexpected_keys}", logging.WARNING)
                self.print_log("权重加载完成 (strict=False)。")
            except Exception as e: self.print_log(f"错误: 加载权重失败: {e}", logging.ERROR); traceback.print_exc(); raise

    def load_optimizer(self):
        """加载优化器和学习率调度器"""
        optimizer_type = (self.arg.optimizer or 'AdamW').lower() # 默认 AdamW
        lr = self.arg.base_lr or 0.001
        wd = self.arg.weight_decay if self.arg.weight_decay is not None else 0.01

        params = self.model.parameters()

        if optimizer_type == 'sgd':
            momentum = getattr(self.arg, 'momentum', 0.9)
            nesterov = self.arg.nesterov or False
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        elif optimizer_type == 'adamw':
             self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
        else: raise ValueError(f"不支持的优化器: {self.arg.optimizer}")
        self.print_log(f"优化器: {self.arg.optimizer.upper()} (lr={lr}, wd={wd})")

        steps = self.arg.step or [50, 70]
        decay_rate = self.arg.lr_decay_rate or 0.1
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps, gamma=decay_rate)
        self.print_log(f"调度器: MultiStepLR (milestones={steps}, gamma={decay_rate})")
        warmup = self.arg.warm_up_epoch or 0
        if warmup > 0: self.print_log(f'使用学习率预热, epochs: {warmup}')

    def adjust_learning_rate(self, epoch):
        """调整学习率，包含 warmup"""
        warmup_epochs = self.arg.warm_up_epoch or 0
        base_lr = self.arg.base_lr or 0.001
        lr = base_lr
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
            return lr
        else:
            # Warmup 结束后，返回当前优化器的学习率 (由 scheduler 控制)
            return self.optimizer.param_groups[0]['lr']

    def record_time(self): self.cur_time = time.time()
    def split_time(self): split = time.time() - self.cur_time; self.record_time(); return split

    # --- train 方法 (适配 feeder_ucla, 使用 collate_fn) ---
    def train(self, epoch):
        self.model.train()
        self.print_log(f'======> 训练 Epoch: {epoch + 1}')
        loader = self.data_loader['train']
        current_lr = self.adjust_learning_rate(epoch)
        if epoch < (self.arg.warm_up_epoch or 0): self.print_log(f'Warmup - 当前学习率: {current_lr:.6f}')
        elif epoch == (self.arg.warm_up_epoch or 0): self.print_log(f'Warmup 结束 - 当前学习率: {current_lr:.6f}')

        loss_value, acc_value = [], []
        if hasattr(self, 'train_writer') and self.train_writer: self.train_writer.add_scalar('meta/epoch', epoch+1, epoch + 1)
        self.record_time()
        timer = dict(dataloader=0.0, model=0.0, statistics=0.0)
        process = tqdm(loader, desc=f"Epoch {epoch+1} Train", ncols=100, leave=False)

        for batch_idx, batch_data in enumerate(process):
            # --- 添加对 None batch 的检查 ---
            if batch_data is None:
                self.print_log(f"警告: DataLoader 在训练 batch {batch_idx} 返回 None，跳过。", logging.WARNING)
                continue
            self.global_step += 1
            # --- 数据处理 ---
            try:
                # collate_fn 返回 (data_batch, label_batch, mask_batch, index_batch)
                data, label, mask, index = batch_data
                # data 形状已经是 (B, T, N, C)
                data = data.float().to(self.output_device, non_blocking=True)
                label = label.long().to(self.output_device, non_blocking=True)
                mask = mask.bool().to(self.output_device, non_blocking=True) # <-- 传递 mask
            except Exception as e: self.print_log(f"错误: 处理训练 batch {batch_idx} 数据失败: {e}", logging.ERROR); continue
            timer['dataloader'] += self.split_time()

            # --- 模型计算与优化 ---
            try:
                output, _ = self.model(data, mask=mask) # <--- 传入 mask
                if not isinstance(output, torch.Tensor): self.print_log(f"错误: 模型输出类型错误 {type(output)}", logging.ERROR); continue
                loss = self.loss(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                max_norm = getattr(self.arg, 'max_grad_norm', 0) or 0
                if max_norm > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                self.optimizer.step()
            except Exception as e: self.print_log(f"错误: 训练 batch {batch_idx} 模型计算/优化失败: {e}", logging.ERROR); traceback.print_exc(); continue
            timer['model'] += self.split_time()

            # --- 统计 ---
            loss_item = loss.item(); loss_value.append(loss_item)
            with torch.no_grad(): preds = torch.argmax(output, 1); acc_item = (preds == label).float().mean().item(); acc_value.append(acc_item)
            process.set_postfix_str(f"Loss: {loss_item:.4f}, Acc: {acc_item:.3f}")
            if hasattr(self, 'train_writer') and self.train_writer and self.global_step % self.arg.log_interval == 0:
                 self.train_writer.add_scalar('train/batch_loss', loss_item, self.global_step)
                 self.train_writer.add_scalar('train/batch_acc', acc_item, self.global_step)
                 self.train_writer.add_scalar('meta/learning_rate', current_lr, self.global_step)
            timer['statistics'] += self.split_time()

        # --- Epoch 结束统计 ---
        avg_loss = np.mean(loss_value) if loss_value else float('nan')
        avg_acc = np.mean(acc_value) * 100 if acc_value else 0.0
        total_time = sum(timer.values()); proportion = { k: f"{int(round(v * 100 / total_time))}%" if total_time > 0 else '0%' for k, v in timer.items() }
        self.print_log(f'\t平均训练损失: {avg_loss:.4f}.  平均训练准确率: {avg_acc:.2f}%.')
        self.print_log(f'\t时间消耗: [数据加载]{proportion["dataloader"]}, [模型计算]{proportion["model"]}, [统计]{proportion["statistics"]}')
        if hasattr(self, 'train_writer') and self.train_writer:
             self.train_writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
             self.train_writer.add_scalar('train/epoch_acc', avg_acc / 100.0, epoch + 1)

        # --- 更新学习率调度器 ---
        if hasattr(self, 'scheduler') and self.scheduler and epoch >= (self.arg.warm_up_epoch or 0):
            self.scheduler.step() # MultiStepLR 在每个 epoch 后调用

        # --- 保存模型 ---
        save_interval = self.arg.save_interval or 0
        save_start_epoch = self.arg.save_epoch or 0
        num_epochs_total = self.arg.num_epoch or 80 # 使用默认值以防万一
        is_last_epoch = (epoch + 1 == num_epochs_total)
        should_save = (save_interval > 0 and (epoch + 1) >= save_start_epoch and (epoch + 1) % save_interval == 0) or is_last_epoch
        if should_save:
             model_path = os.path.join(self.arg.work_dir, f'epoch-{epoch+1}_step-{self.global_step}.pt')
             state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
             weights = OrderedDict([[k.replace('module.', ''), v.cpu()] for k, v in state_dict.items()])
             try: torch.save(weights, model_path); self.print_log(f'模型已保存到: {model_path}')
             except Exception as e: self.print_log(f"警告: 保存模型失败 {model_path}: {e}", logging.WARNING)

    # --- eval 方法 (适配 feeder_ucla, 使用 sklearn, 添加 collate_fn 处理) ---
    def eval(self, epoch, save_score=False, loader_name=['val'], wrong_file=None, result_file=None):
        f_w = open(wrong_file, 'w', encoding='utf-8') if wrong_file else None
        f_r = open(result_file, 'w', encoding='utf-8', newline='') if result_file else None
        csv_writer = csv.writer(f_r) if f_r else None
        if csv_writer: csv_writer.writerow(["Sample_Index", "Prediction", "True_Label"])

        self.model.eval()
        self.print_log(f'======> 评估 Epoch: {epoch + 1} on {", ".join(loader_name)}')
        eval_acc_final = 0.0; eval_loss_final = 0.0; total_loaders = 0

        for ln in loader_name:
            if ln not in self.data_loader: self.print_log(f"警告: 找不到加载器 '{ln}'", logging.WARNING); continue
            total_loaders += 1
            all_loss, all_logits, all_labels, all_indices = [], [], [], []
            process = tqdm(self.data_loader[ln], desc=f"Epoch {epoch+1} Eval {ln}", ncols=100, leave=False)
            for batch_idx, batch_data in enumerate(process):
                # --- 添加对 None batch 的检查 ---
                if batch_data is None:
                    self.print_log(f"警告: DataLoader 在评估 {ln} 的 batch {batch_idx} 返回 None，跳过。", logging.WARNING)
                    continue
                # --- 数据处理 ---
                try:
                    data, label, mask, index = batch_data # <-- 接收 mask 和 index
                    # data 形状已经是 (B, T, N, C)
                    data = data.float().to(self.output_device, non_blocking=True)
                    label = label.long().to(self.output_device, non_blocking=True)
                    mask = mask.bool().to(self.output_device, non_blocking=True) # <-- 传递 mask
                    index = index.long().to(self.output_device, non_blocking=True)
                except Exception as e: self.print_log(f"错误: 处理评估 batch {batch_idx} 数据失败: {e}", logging.ERROR); continue
                # --- 模型计算 ---
                with torch.no_grad():
                    try:
                        output, _ = self.model(data, mask=mask) # <--- 传入 mask
                        loss = self.loss(output, label)
                        all_loss.append(loss.item()); all_logits.append(output.cpu())
                        all_labels.append(label.cpu()); all_indices.append(index.cpu())
                    except Exception as e: self.print_log(f"错误: 评估 batch {batch_idx} 模型计算失败: {e}", logging.ERROR); continue

            # --- 计算总指标 ---
            if not all_logits: self.print_log(f"警告: 在 {ln} 上没有处理任何数据。", logging.WARNING); continue
            try:
                logits_all = torch.cat(all_logits, dim=0).numpy()
                labels_all = torch.cat(all_labels, dim=0).numpy()
                preds_all = np.argmax(logits_all, axis=1)
                indices_all = torch.cat(all_indices, dim=0).numpy()
            except RuntimeError as e: self.print_log(f"错误: 组合评估结果失败: {e}", logging.ERROR); continue

            eval_loss = np.mean(all_loss) if all_loss else float('nan')
            eval_acc = accuracy_score(labels_all, preds_all)
            if ln == loader_name[-1]: # 保存最后一个 loader 的结果
                eval_loss_final = eval_loss; eval_acc_final = eval_acc

            # --- 日志和 TensorBoard ---
            writer_prefix = 'val'
            if self.arg.phase == 'train' and hasattr(self, f'{writer_prefix}_writer') and getattr(self, f'{writer_prefix}_writer') is not None:
                writer = getattr(self, f'{writer_prefix}_writer')
                writer.add_scalar(f'eval/{ln}_epoch_loss', eval_loss, epoch + 1)
                writer.add_scalar(f'eval/{ln}_epoch_acc', eval_acc, epoch + 1)
            self.print_log(f'\t{ln} 集上的平均损失: {eval_loss:.4f}')
            self.print_log(f'\t{ln} 集上的准确率 (Top-1): {eval_acc * 100:.2f}%')

            # --- Top-K ---
            num_classes = self.arg.model_args.get('num_classes', 0)
            if num_classes > 0:
                class_labels = np.arange(num_classes)
                for k in self.arg.show_topk:
                     if k >= num_classes: continue
                     try:
                         topk_acc = top_k_accuracy_score(labels_all, logits_all, k=k, labels=class_labels, normalize=True)
                         self.print_log(f'\t{ln} 集上的准确率 (Top-{k}): {topk_acc * 100:.2f}%')
                         if self.arg.phase == 'train' and hasattr(self, f'{writer_prefix}_writer') and getattr(self, f'{writer_prefix}_writer') is not None:
                              writer.add_scalar(f'eval/{ln}_epoch_acc_top{k}', topk_acc, epoch + 1)
                     except Exception as e: self.print_log(f"警告: 计算 Top-{k} 准确率失败: {e}", logging.WARNING)

            # --- 保存分数 ---
            if save_score and ln == 'test':
                 score_dict = {idx.item(): score_vec for idx, score_vec in zip(indices_all, logits_all)}
                 score_path = os.path.join(self.arg.work_dir, f'epoch{epoch+1}_{ln}_score.pkl')
                 try:
                     with open(score_path, 'wb') as f: pickle.dump(score_dict, f)
                     self.print_log(f"预测分数已保存到: {score_path}")
                 except Exception as e: self.print_log(f"警告: 保存分数失败 {score_path}: {e}", logging.WARNING)

            # --- 记录样本结果 ---
            if (f_w or f_r) and ln == 'test':
                 for i in range(len(labels_all)):
                      pred_i, true_i, index_i = preds_all[i].item(), labels_all[i].item(), indices_all[i].item()
                      if f_r: csv_writer.writerow([index_i, pred_i, true_i])
                      if f_w and pred_i != true_i: f_w.write(f"{index_i},{pred_i},{true_i}\n")

            # --- 混淆矩阵 ---
            if num_classes > 0 and ln == 'test':
                try:
                    confusion = confusion_matrix(labels_all, preds_all, labels=class_labels)
                    acc_csv_path = os.path.join(self.arg.work_dir, f'epoch{epoch+1}_{ln}_confusion_matrix.csv')
                    with open(acc_csv_path, 'w', newline='', encoding='utf-8') as f:
                         writer = csv.writer(f); list_diag = np.diag(confusion); list_raw_sum = np.sum(confusion, axis=1)
                         each_acc = np.divide(list_diag, list_raw_sum, out=np.zeros_like(list_diag, dtype=float), where=list_raw_sum!=0)
                         writer.writerow(["Class_Index", "Accuracy"]); [writer.writerow([i, acc_i]) for i, acc_i in enumerate(each_acc)]
                         writer.writerow([]); writer.writerow(["Confusion Matrix (True \\ Pred)"])
                         writer.writerow(["True\\Pred"] + [f"Pred_{i}" for i in class_labels]); [writer.writerow([f"True_{i}"] + row.tolist()) for i, row in enumerate(confusion)]
                    self.print_log(f"各类别准确率和混淆矩阵已保存到: {acc_csv_path}")
                except Exception as e: self.print_log(f"警告: 计算或保存混淆矩阵失败: {e}", logging.WARNING)

        # --- 关闭文件 ---
        if f_w: f_w.close()
        if f_r: f_r.close()
        # 返回最后一个 loader 的准确率
        return eval_acc_final

    # --- start 方法 (读取顶层 num_epoch) ---
    def start(self):
        if self.arg.phase == 'train':
             self.print_log('开始训练阶段...')
             self.print_log('参数:\n{}\n'.format(yaml.dump(vars(self.arg), default_flow_style=None, sort_keys=False)))
             try: self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
             except KeyError: self.global_step = 0; self.print_log("警告: 无法获取训练加载器长度。", logging.WARNING)
             def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
             self.print_log(f'模型可训练参数量: {count_parameters(self.model):,}')

             num_epochs_train = self.arg.num_epoch # <-- 从顶层 arg 读取
             if num_epochs_train is None: raise ValueError("必须在 YAML 或命令行设置 num_epoch")
             self.print_log(f"总训练 Epochs: {num_epochs_train}")

             for epoch in range(self.arg.start_epoch, num_epochs_train):
                 self.train(epoch) # 训练

                 if (epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == num_epochs_train:
                     val_acc = self.eval(epoch, save_score=False, loader_name=['val']) # 在验证集 'val' 上评估
                     if val_acc > self.best_acc:
                         self.best_acc = val_acc
                         self.best_acc_epoch = epoch + 1
                         best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                         try:
                             state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                             weights = OrderedDict([[k.replace('module.', ''), v.cpu()] for k, v in state_dict.items()])
                             torch.save(weights, best_model_path)
                             self.print_log(f'*** 新的最佳准确率: {self.best_acc*100:.2f}% (Epoch: {self.best_acc_epoch}). 模型已保存到 {best_model_path} ***')
                         except Exception as e: self.print_log(f"警告: 保存最佳模型失败 {best_model_path}: {e}", logging.WARNING)

             # --- 训练结束总结和最终测试 ---
             self.print_log('训练完成。')
             if self.best_acc_epoch > 0 :
                 self.print_log(f'最佳验证准确率 (Top-1): {self.best_acc*100:.2f}% 在 Epoch {self.best_acc_epoch}.')
                 self.print_log('加载最佳模型进行最终测试...')
                 best_weights_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                 if os.path.exists(best_weights_path):
                      try:
                          weights = torch.load(best_weights_path, map_location=self.output_device)
                          self.model.load_state_dict(weights)
                          self.print_log(f"最佳模型权重 {best_weights_path} 加载成功。")
                          wf = os.path.join(self.arg.work_dir, 'final_test_wrong.txt')
                          rf = os.path.join(self.arg.work_dir, 'final_test_results.csv')
                          self.print_log('对测试集进行最终评估...')
                          self.eval(epoch=self.best_acc_epoch - 1, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf) # 使用 'test' loader
                      except Exception as e: self.print_log(f"错误: 加载或测试最佳模型失败: {e}", logging.ERROR)
                 else: self.print_log(f"警告: 找不到最佳模型文件 {best_weights_path}，无法进行最终测试。", logging.WARNING)
             else: self.print_log("训练过程中没有找到效果更好的模型。", logging.WARNING)
             # ... (打印最终总结信息) ...
             self.print_log('====== 训练总结 ======'); self.print_log(f'最佳验证准确率 (Top-1): {self.best_acc*100:.2f}% (Epoch: {self.best_acc_epoch})')
             def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
             self.print_log(f'模型参数量: {count_parameters(self.model):,}'); self.print_log(f'工作目录: {self.arg.work_dir}'); self.print_log(f'配置文件: {self.arg.config}'); self.print_log('=======================')


        elif self.arg.phase == 'test':
             self.print_log('开始测试阶段...')
             if not self.arg.weights: raise ValueError('--weights 必须指定')
             if not os.path.exists(self.arg.weights): raise FileNotFoundError(f"找不到权重: {self.arg.weights}")
             wf = self.arg.weights.replace('.pt', '_wrong.txt'); rf = self.arg.weights.replace('.pt', '_results.csv')
             self.print_log('模型:   {}'.format(self.arg.model)); self.print_log('权重: {}'.format(self.arg.weights))
             self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf) # 在 test loader 上评估
             self.print_log('测试完成。')

# --- 主程序入口 ---
if __name__ == '__main__':
    parser = get_parser()
    # --- 参数加载与合并 ---
    p, unknown_args = parser.parse_known_args()
    if unknown_args: print(f"警告: 发现未知命令行参数，将被忽略: {unknown_args}")
    config_path = p.config
    default_arg = {}
    yaml_loaded = False
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f: default_arg = yaml.load(f, Loader=Loader)
            if default_arg is None: default_arg = {}
            yaml_loaded = True
            print(f"--- 已加载配置文件: {config_path} ---")
            parser.set_defaults(**default_arg)
        except Exception as e: print(f"错误: 解析配置文件 {config_path} 失败: {e}"); sys.exit(1)
    else: print(f"警告: 未指定或找不到配置文件: {config_path}。")

    arg = parser.parse_args() # 应用 YAML 默认值
    # 用命令行值覆盖 YAML 默认值
    cmd_args_dict = vars(p)
    for k, v_cmd in cmd_args_dict.items():
        if v_cmd is not None and hasattr(arg, k) and v_cmd != parser.get_default(k):
             if isinstance(getattr(arg, k, None), dict) and isinstance(v_cmd, dict):
                  print(f"Info: 合并命令行字典参数 '{k}': {v_cmd}")
                  getattr(arg, k).update(v_cmd)
             else: setattr(arg, k, v_cmd)
    arg.config = config_path if config_path else arg.config

    # --- 打印最终参数 ---
    print("\n--- 最终使用的参数 ---")
    final_args_dict = vars(arg)
    filtered_args = {k: v for k, v in final_args_dict.items() if not (isinstance(v, dict) and not v) or k in ['model_args', 'train_feeder_args', 'test_feeder_args']}
    print(yaml.dump(filtered_args, default_flow_style=False, sort_keys=False))
    print("---------------------\n")

    # --- 检查关键参数 ---
    required_params = ['feeder', 'model', 'model_args', 'train_feeder_args', 'test_feeder_args',
                       'optimizer', 'base_lr', 'step', 'test_batch_size', 'num_epoch', 'batch_size']
    missing = [k for k in required_params if getattr(arg, k, None) is None or (isinstance(getattr(arg, k, None), dict) and not getattr(arg, k, None))]
    if missing: print(f"错误：缺少必要的配置参数或参数值为空: {missing}。请检查 YAML 或命令行。"); sys.exit(1)

    # --- 初始化并启动 ---
    init_seed(arg.seed)
    processor = None
    try:
        processor = Processor(arg)
        processor.start()
    except KeyboardInterrupt: print("\n训练被手动中断。")
    except Exception as e:
        if processor and hasattr(processor, 'logger') and processor.logger:
             processor.logger.error(f"程序顶层意外终止: {e}", exc_info=True)
        else: print(f"程序顶层意外终止: {e}"); traceback.print_exc()
    finally:
        if processor and hasattr(processor, 'train_writer') and processor.train_writer:
            try: processor.train_writer.close()
            except Exception as e: print(f"关闭 train_writer 时出错: {e}")
        if processor and hasattr(processor, 'val_writer') and processor.val_writer:
            try: processor.val_writer.close()
            except Exception as e: print(f"关闭 val_writer 时出错: {e}")