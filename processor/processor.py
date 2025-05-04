# 文件名: processor/processor.py (v17.5 - 最终修正版)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tensorboardX import SummaryWriter
import numpy as np
import yaml
try:
    # 尝试使用更快的 C 加载器/卸载器
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import logging
import os
import sys
import time
import shutil
import inspect
import pickle
import csv
import glob
import traceback
import random
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score

# --- 从 utils 导入 ---
try:
    # 假设你的 utils.py 在上一级目录或者 PYTHONPATH 中
    # from ..utils import init_seed, import_class, str2bool, DictAction, LabelSmoothingCrossEntropy, collate_fn_filter_none
    # 或者如果 processor.py 和 utils.py 在同一级
    from utils import init_seed, import_class, str2bool, DictAction, LabelSmoothingCrossEntropy, collate_fn_filter_none
except ImportError:
     # Fallback: 尝试添加项目根目录（假设 processor 在 processor/ 子目录）
     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
     from utils import init_seed, import_class, str2bool, DictAction, LabelSmoothingCrossEntropy, collate_fn_filter_none

# --- 添加 CosineLRScheduler 的导入 ---
try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    # 如果没有安装 timm，记录错误并设置 CosineLRScheduler 为 None
    logging.getLogger(__name__).error("无法从 timm.scheduler 导入 CosineLRScheduler。请安装 timm: pip install timm")
    CosineLRScheduler = None

logger = logging.getLogger(__name__)

class Processor():
    """
    封装了训练和测试的主要逻辑。
    (v17.5: 最终修正版，修复初始化顺序和 Cosine 调度器处理)
    """
    def __init__(self, arg):
        """
        初始化 Processor。
        Args:
            arg (argparse.Namespace): 解析后的命令行参数和配置。
        """
        self.arg = arg
        self.save_arg() # 先保存配置
        self.setup_device()
        self.setup_logging_and_writers()
        self.print_log("Processor 初始化开始...")

        # 1. 加载数据
        self.load_data()

        # 2. 加载模型
        self.load_model()

        # 3. 初始化优化器和调度器 (如果需要)
        self.optimizer = None
        self.scheduler = None # for epoch-based scheduler
        self.lr_scheduler_each_step = None # for step-based scheduler

        n_iter_per_epoch = 0 # 初始化
        if self.arg.phase == 'train':
            # 计算每个 epoch 的迭代次数
            if 'train' in self.data_loader and self.data_loader['train'] is not None:
                try:
                    n_iter_per_epoch = len(self.data_loader['train'])
                    if n_iter_per_epoch == 0: logger.warning("训练数据加载器长度为 0。")
                except Exception as e:
                    self.print_log(f"警告: 获取训练迭代次数失败: {e}", logging.WARNING)
            else:
                 self.print_log("警告: 训练数据加载器未初始化，无法计算 n_iter_per_epoch", logging.WARNING)

            # 计算初始 global_step (基于 0-based epoch)
            self.global_step = self.arg.start_epoch * n_iter_per_epoch

            # 加载优化器
            self.load_optimizer()
            # 加载调度器
            self.load_scheduler(n_iter_per_epoch)

        else: # 测试阶段
             self.global_step = 0
             # 测试阶段也创建一个优化器对象，以便代码其他地方（如日志）引用 optimizer.param_groups[0]['lr'] 不会出错
             # 但这个优化器不会被用来更新权重
             if self.optimizer is None: self.load_optimizer()

        # 获取当前学习率
        self.lr = self.optimizer.param_groups[0]['lr'] if self.optimizer and self.optimizer.param_groups else self.arg.base_lr
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.best_state_dict = None # 用于存储最佳模型状态

        # 将模型移至设备并设置 DataParallel
        self.model = self.model.to(self.output_device)
        if type(self.arg.device) is list and len(self.arg.device) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device
            )
            self.print_log(f'模型已在 GPUs {self.arg.device} 上启用 DataParallel。')

        self.print_log("Processor 初始化完成。")

    def setup_device(self):
        """根据参数设置运行设备 (GPU 或 CPU)"""
        if not hasattr(self.arg, 'device') or self.arg.device is None: # 检查属性是否存在
            self.arg.device = [0] if torch.cuda.is_available() else [-1]
        if not isinstance(self.arg.device, list): self.arg.device = [self.arg.device]
        if self.arg.device[0] == -1 or not torch.cuda.is_available():
            self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]
            self.print_log("将在 CPU 上运行。")
        else:
            valid_devices = [d for d in self.arg.device if 0 <= d < torch.cuda.device_count()]
            if not valid_devices:
                self.print_log(f"错误: 无效 GPU 索引 {self.arg.device}。将使用 CPU。", logging.ERROR)
                self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]
            else:
                self.arg.device = valid_devices
                self.output_device = torch.device(f"cuda:{self.arg.device[0]}")
                try:
                    torch.cuda.set_device(self.output_device)
                    self.print_log(f"使用 GPU: {self.arg.device}。主输出设备: {self.output_device}")
                except Exception as e:
                    self.print_log(f"错误: 设置 CUDA 设备失败: {e}。将使用 CPU。", logging.ERROR)
                    self.output_device = torch.device("cpu"); self.arg.device = ["cpu"]

    def setup_logging_and_writers(self):
        """初始化日志记录器和 TensorBoard"""
        work_dir = self.arg.work_dir
        os.makedirs(work_dir, exist_ok=True) # 使用 exist_ok=True 简化
        log_file = os.path.join(work_dir, 'log.txt')
        should_clear_log = self.arg.phase == 'train' and self.arg.start_epoch == 0
        filemode = 'w' if should_clear_log else 'a'

        # 配置 logging
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', # 添加 logger name
                            handlers=[logging.FileHandler(log_file, mode=filemode, encoding='utf-8'),
                                      logging.StreamHandler(sys.stdout)])
        # 获取 logger
        self.logger = logging.getLogger(f"Processor_{os.path.basename(work_dir)}")
        self.print_log(f'工作目录: {work_dir}') # 使用 self.print_log

        # 配置 TensorBoard
        if self.arg.phase == 'train' and not getattr(self.arg, 'debug', False):
            runs_dir = os.path.join(work_dir, 'runs')
            if os.path.isdir(runs_dir) and should_clear_log:
                 self.print_log(f"清空已存在的 TensorBoard 日志目录: {runs_dir}")
                 try: shutil.rmtree(runs_dir)
                 except OSError as e: self.print_log(f"警告: 清空 TensorBoard 目录失败: {e}", logging.WARNING)
            os.makedirs(runs_dir, exist_ok=True)
            try:
                self.train_writer = SummaryWriter(os.path.join(runs_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(runs_dir, 'val'), 'val')
                self.print_log(f'TensorBoard 日志已设置在: {runs_dir}')
            except Exception as e:
                self.print_log(f"警告: 初始化 TensorBoard 失败: {e}", logging.WARNING)
                self.train_writer = self.val_writer = None
        else:
             self.train_writer = self.val_writer = None

    def print_log(self, msg, level=logging.INFO, print_time=True):
        """打印日志信息"""
        # (代码与 v17.4 相同)
        if getattr(self.arg, 'print_log', True): # 检查 print_log 标志
            log_msg = msg
            if print_time:
                localtime = time.asctime(time.localtime(time.time()))
                log_msg = f"[{localtime}] {msg}"
            # 使用 self.logger (如果存在)
            if hasattr(self, 'logger') and self.logger:
                self.logger.log(level, log_msg)
            else: # Fallback to print
                 print(log_msg)

    def save_arg(self):
        """保存配置到工作目录"""
        # (代码与 v17.4 相同)
        arg_dict = vars(self.arg)
        work_dir = self.arg.work_dir
        os.makedirs(work_dir, exist_ok=True)
        try:
            filepath = os.path.join(work_dir, 'config_used.yaml')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Work Dir: {work_dir}\n") # 添加工作目录信息
                f.write(f"# Phase: {self.arg.phase}\n") # 添加阶段信息
                f.write(f"# Command line: {' '.join(sys.argv)}\n\n")
                yaml.dump(arg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True, Dumper=Dumper)
        except Exception as e:
            self.print_log(f"警告: 保存 config_used.yaml ({work_dir}) 失败: {e}", logging.WARNING)

    def load_data(self):
        """加载数据加载器"""
        # (代码与 v17.4 基本相同, 增加 feeder 存在性检查)
        self.print_log("开始加载数据...")
        feeder_path = getattr(self.arg, 'feeder', None)
        if not feeder_path: raise ValueError("'feeder' 参数未在配置中设置。")
        try:
            Feeder = import_class(feeder_path)
        except (ImportError, ValueError) as e:
            self.print_log(f"错误: 无法导入或 Feeder 未设置 '{feeder_path}'. {e}", logging.ERROR); raise e

        self.data_loader = dict()
        try:
            num_worker = getattr(self.arg, 'num_worker', 4)

            # 加载训练数据
            if self.arg.phase == 'train':
                train_batch_size = getattr(self.arg, 'batch_size', None)
                if train_batch_size is None: raise ValueError("训练阶段必须设置 batch_size")
                train_feeder_args = getattr(self.arg, 'train_feeder_args', {})
                if not isinstance(train_feeder_args, dict): raise TypeError("train_feeder_args 应为字典")
                feeder_args_train = train_feeder_args.copy()
                if 'data_path' not in feeder_args_train: raise ValueError("train_feeder_args 缺少 data_path")
                feeder_args_train.setdefault('split', 'train') # 设置默认 split

                train_dataset = Feeder(**feeder_args_train)
                self.data_loader['train'] = DataLoader(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=num_worker, drop_last=getattr(self.arg, 'drop_last', True),
                    worker_init_fn=init_seed, pin_memory=True, collate_fn=collate_fn_filter_none
                )
                self.print_log(f"训练数据加载器 '{self.arg.feeder}' (模态: {feeder_args_train.get('data_path', '?')}) 加载成功。样本数: {len(train_dataset)}")

            # 加载测试/验证数据
            test_batch_size = getattr(self.arg, 'test_batch_size', None)
            if test_batch_size is None: raise ValueError("必须设置 test_batch_size")
            test_feeder_args = getattr(self.arg, 'test_feeder_args', {})
            if not isinstance(test_feeder_args, dict): raise TypeError("test_feeder_args 应为字典")
            feeder_args_test = test_feeder_args.copy()
            if 'data_path' not in feeder_args_test: raise ValueError("test_feeder_args 缺少 data_path")
            feeder_args_test.setdefault('split', 'val') # 设置默认 split

            test_dataset = Feeder(**feeder_args_test)
            self.data_loader['val'] = DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=False,
                num_workers=num_worker, drop_last=False, worker_init_fn=init_seed,
                pin_memory=True, collate_fn=collate_fn_filter_none
            )
            self.data_loader['test'] = self.data_loader['val'] # test 和 val 使用相同加载器
            self.print_log(f"测试/验证数据加载器 '{self.arg.feeder}' (模态: {feeder_args_test.get('data_path', '?')}) 加载成功。样本数: {len(test_dataset)}")

        except Exception as e:
            self.print_log(f"错误: 加载数据失败: {e}", logging.ERROR); traceback.print_exc(); raise e
        self.print_log("数据加载完成。")

    def load_model(self):
        """加载模型结构和权重"""
        # (代码与 v17.4 基本相同)
        self.print_log(f"模型将运行在设备: {self.output_device}")
        model_path_str = getattr(self.arg, 'model', None)
        if not model_path_str: raise ValueError("'model' 参数未设置。")
        try:
            Model = import_class(model_path_str)
            try: model_file_path = inspect.getfile(Model); shutil.copy2(model_file_path, self.arg.work_dir)
            except Exception as e: self.print_log(f"警告: 复制模型文件失败: {e}", logging.WARNING)

            model_args_dict = getattr(self.arg, 'model_args', {})
            if not isinstance(model_args_dict, dict) or not model_args_dict: raise ValueError("'model_args' 参数未设置、类型错误或为空。")
            self.model = Model(model_cfg=model_args_dict)
            self.print_log(f"模型 '{self.arg.model}' 实例化成功。")
        except (ImportError, ValueError, TypeError, Exception) as e:
            self.print_log(f"错误: 模型加载/实例化失败: {e}", logging.ERROR); traceback.print_exc(); raise e

        loss_type = getattr(self.arg, 'loss_type', 'CE').upper()
        if loss_type == 'SMOOTHCE': self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).to(self.output_device); self.print_log(f"损失函数: LabelSmoothingCrossEntropy (smoothing=0.1)")
        else: self.loss = nn.CrossEntropyLoss().to(self.output_device); self.print_log(f"损失函数: CrossEntropyLoss")

        weights_path = getattr(self.arg, 'weights', None)
        if weights_path:
            self.print_log(f'加载权重自: {weights_path}'); assert os.path.exists(weights_path), f"权重文件不存在: {weights_path}"
            try:
                weights = torch.load(weights_path, map_location=self.output_device); weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
                ignore_weights_list = getattr(self.arg, 'ignore_weights', [])
                if ignore_weights_list:
                     # ... (忽略权重的逻辑) ...
                     keys = list(weights.keys())
                     for w_name in ignore_weights_list:
                         removed_keys = [k for k in keys if w_name in k]
                         if not removed_keys: self.print_log(f'警告: 未找到要忽略的权重关键字: {w_name}', logging.WARNING)
                         for key in removed_keys:
                             if weights.pop(key, None) is not None: self.print_log(f'已忽略权重: {key}')

                missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
                if missing_keys: self.print_log(f"警告: 模型中缺失的键: {missing_keys}", logging.WARNING)
                if unexpected_keys: self.print_log(f"警告: 权重文件中多余的键: {unexpected_keys}", logging.WARNING)
                self.print_log("权重加载完成 (strict=False)。")
            except Exception as e: self.print_log(f"错误: 加载权重失败: {e}", logging.ERROR); traceback.print_exc(); raise e

    def load_optimizer(self):
        """加载优化器"""
        # (代码与 v17.4 相同)
        optimizer_type = (getattr(self.arg, 'optimizer', 'AdamW')).lower()
        lr = getattr(self.arg, 'base_lr', 0.001)
        wd = getattr(self.arg, 'weight_decay', 0.01)
        # 过滤掉不需要梯度的参数
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        if not params_to_optimize:
             self.print_log("警告: 模型中没有需要优化的参数！", level=logging.WARNING)
             # 创建一个空的优化器或者不创建？最好还是创建一个，避免后续引用报错
             # self.optimizer = None # 这样后续代码可能报错
             # 创建一个作用于空列表的优化器
             self.optimizer = optim.AdamW([], lr=lr, weight_decay=wd)
             return

        if optimizer_type == 'sgd':
            momentum = getattr(self.arg, 'momentum', 0.9)
            nesterov = getattr(self.arg, 'nesterov', False)
            self.optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd)
            self.print_log(f"优化器: SGD (lr={lr}, momentum={momentum}, nesterov={nesterov}, wd={wd})")
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
            self.print_log(f"优化器: ADAM (lr={lr}, wd={wd})")
        elif optimizer_type == 'adamw':
             self.optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd)
             self.print_log(f"优化器: ADAMW (lr={lr}, wd={wd})")
        else:
            raise ValueError(f"不支持的优化器: {getattr(self.arg, 'optimizer', '未指定')}")


    # <<<--- 修改后的 load_scheduler (与 v17.4 相同) --- >>>
    def load_scheduler(self, n_iter_per_epoch=0):
        """加载学习率调度器 (支持 MultiStepLR 和 CosineLRScheduler from timm)"""
        scheduler_type = (getattr(self.arg, 'lr_scheduler', 'multistep')).lower()
        self.scheduler = None # epoch-based
        self.lr_scheduler_each_step = None # step-based

        self.print_log(f"尝试加载调度器: {scheduler_type}")

        # 确保优化器已加载
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.print_log("错误：优化器未初始化，无法加载调度器。", logging.ERROR)
            return

        if scheduler_type == 'multistep':
            steps = getattr(self.arg, 'step', None)
            if not steps or not isinstance(steps, list):
                 self.print_log(f"MultiStepLR 需要设置 'step' 参数 (epoch 列表)，当前为 {steps}。将不使用 epoch 调度器。", logging.WARNING)
            else:
                decay_rate = getattr(self.arg, 'lr_decay_rate', 0.1)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps, gamma=decay_rate)
                self.print_log(f"调度器: MultiStepLR (milestones_epoch={steps}, gamma={decay_rate})")

        elif scheduler_type == 'cosine':
            if CosineLRScheduler is None:
                self.print_log("错误: 已配置使用 cosine 调度器，但无法导入 CosineLRScheduler from timm。请安装 timm。", logging.ERROR); return
            if n_iter_per_epoch <= 0:
                 self.print_log("错误: CosineLRScheduler 需要知道每 epoch 的迭代次数 (n_iter_per_epoch > 0)。调度器初始化失败。", logging.ERROR); return

            num_steps = int(getattr(self.arg, 'num_epoch', 0) * n_iter_per_epoch)
            warmup_steps = int(getattr(self.arg, 'warm_up_epoch', 0) * n_iter_per_epoch)
            warmup_prefix = getattr(self.arg, 'warmup_prefix', True) # 默认 True
            if isinstance(getattr(self.arg, 'warmup_prefix', None), bool) and not self.arg.warmup_prefix: warmup_prefix = False

            self.print_log(f"CosineLRScheduler 配置: num_total_steps={num_steps}, warmup_steps={warmup_steps}, warmup_prefix={warmup_prefix}")

            try:
                self.lr_scheduler_each_step = CosineLRScheduler(
                    self.optimizer,
                    t_initial=(num_steps - warmup_steps) if warmup_prefix else num_steps,
                    lr_min=getattr(self.arg, 'min_lr', 1e-6),
                    warmup_lr_init=getattr(self.arg, 'warmup_lr', 1e-6),
                    warmup_t=warmup_steps,
                    cycle_limit=1, t_in_epochs=False, warmup_prefix=warmup_prefix,
                )
                self.print_log(f"调度器: CosineLRScheduler (timm) (t_initial={'adjusted' if warmup_prefix else num_steps}, warmup={warmup_steps}, min_lr={getattr(self.arg, 'min_lr', 1e-6)}, warmup_lr={getattr(self.arg, 'warmup_lr', 1e-6)}) 加载成功。")
            except Exception as e:
                self.print_log(f"错误: 初始化 CosineLRScheduler 失败: {e}", logging.ERROR); self.lr_scheduler_each_step = None

        else:
            self.print_log(f"警告: 不支持的学习率调度器类型 '{scheduler_type}'。将不使用调度器。", logging.WARNING)

        if getattr(self.arg, 'warm_up_epoch', 0) > 0 and scheduler_type != 'cosine':
            self.print_log(f'使用手动学习率预热 (非 Cosine), epochs: {self.arg.warm_up_epoch}')


    def record_time(self): self.cur_time = time.time()
    def split_time(self): split = time.time() - self.cur_time; self.record_time(); return split

    # <<<--- train 方法 (与 v17.4 相同) --- >>>
    def train(self, epoch):
        self.model.train()
        self.print_log(f'======> 训练 Epoch: {epoch + 1}')

        loader = self.data_loader.get('train') # 使用 get 获取，避免 KeyError
        if not loader: self.print_log("错误：训练数据加载器 'train' 不存在！", level=logging.ERROR); return
        n_iter_per_epoch = len(loader)
        if n_iter_per_epoch == 0: self.print_log("警告：训练数据加载器为空！", level=logging.WARNING); return

        loss_value, acc_value, grad_norm_value = [], [], []
        if self.train_writer: self.train_writer.add_scalar('meta/epoch', epoch + 1, epoch + 1)
        self.record_time(); timer = dict(dataloader=0.0, model=0.0, statistics=0.0)
        process = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=80, leave=False)

        for batch_idx, batch_data in enumerate(process):
            if hasattr(self, 'lr_scheduler_each_step') and self.lr_scheduler_each_step is not None:
                 self.lr_scheduler_each_step.step(self.global_step) # 使用 global_step 更新 timm scheduler
            self.global_step += 1
            if batch_data is None: continue

            try:
                if len(batch_data) == 4: data, label, mask, index = batch_data
                elif len(batch_data) == 3: data, label, index = batch_data; mask = None
                else: raise ValueError(f"未知的 batch_data 格式，长度为 {len(batch_data)}")
                data = data.float().to(self.output_device, non_blocking=True)
                label = label.long().to(self.output_device, non_blocking=True)
                if mask is not None: mask = mask.bool().to(self.output_device, non_blocking=True)
            except ValueError as e:
                if batch_idx % self.arg.log_interval == 0: self.print_log(f"警告: 处理 batch {batch_idx} 数据失败: {e}", logging.WARNING); continue
            timer['dataloader'] += self.split_time()

            try:
                output, _ = self.model(data, mask=mask)
                loss = self.loss(output, label)
                if torch.isnan(loss) or torch.isinf(loss):
                    if batch_idx % self.arg.log_interval == 0: self.print_log(f"警告: Batch {batch_idx} 损失为 NaN/Inf！", logging.WARNING); continue
            except Exception as e:
                 if batch_idx % self.arg.log_interval == 0: self.print_log(f"错误: Batch {batch_idx} 前向计算失败: {e}", logging.ERROR); traceback.print_exc(); continue

            try:
                self.optimizer.zero_grad(); loss.backward(); total_norm_before_clip = 0; valid_grad = True
                for p in self.model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): valid_grad = False; break
                        param_norm = p.grad.data.norm(2); total_norm_before_clip += param_norm.item() ** 2
                if valid_grad:
                    total_norm_before_clip = total_norm_before_clip ** 0.5; grad_norm_value.append(total_norm_before_clip)
                    if getattr(self.arg, 'grad_clip', False): max_norm = getattr(self.arg, 'grad_max', 1.0); torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                    self.optimizer.step()
                else: self.optimizer.zero_grad()
            except Exception as e:
                if batch_idx % self.arg.log_interval == 0: self.print_log(f"错误: Batch {batch_idx} 优化失败: {e}", logging.ERROR); continue
            timer['model'] += self.split_time()

            acc_item = float('nan')
            if valid_grad:
                loss_item = loss.item(); loss_value.append(loss_item)
                try:
                    with torch.no_grad(): _, predict_label = torch.max(output.data, 1); acc = torch.mean((predict_label == label.data).float()); acc_item = acc.item(); acc_value.append(acc_item)
                except Exception as e_acc: logger.warning(f"Batch {batch_idx} 计算准确率失败: {e_acc}")
            else: loss_item = float('nan')

            process.set_postfix_str(f"Loss: {loss_item:.4f}, Acc: {acc_item:.3f}")

            if self.global_step % self.arg.log_interval == 0:
                 current_lr_for_log = self.optimizer.param_groups[0]['lr']; grad_str = f"{total_norm_before_clip:.4f}" if valid_grad and not np.isnan(total_norm_before_clip) else "NaN"
                 loss_str = f"{loss_item:.4f}" if not np.isnan(loss_item) else "NaN"; acc_str = f"{acc_item:.3f}" if not np.isnan(acc_item) else "NaN"
                 log_str = (f"Epoch: [{epoch+1}][{batch_idx+1}/{len(loader)}]\tLoss: {loss_str}\tAcc: {acc_str}\tLR: {current_lr_for_log:.6f}\tGradNorm: {grad_str}")
                 self.print_log(log_str, print_time=False)
                 if self.train_writer:
                    try: # Tensorboard logging
                        if not np.isnan(loss_item): self.train_writer.add_scalar('train/batch_loss', loss_item, self.global_step)
                        if not np.isnan(acc_item): self.train_writer.add_scalar('train/batch_acc', acc_item, self.global_step)
                        if not np.isnan(total_norm_before_clip): self.train_writer.add_scalar('train/grad_norm', total_norm_before_clip, self.global_step)
                        self.train_writer.add_scalar('meta/learning_rate', current_lr_for_log, self.global_step)
                    except Exception as e: self.print_log(f"警告: 写入 TensorBoard 失败: {e}", level=logging.WARNING)
            timer['statistics'] += self.split_time()

        process.close()
        avg_loss = np.nanmean(loss_value) if loss_value else float('nan'); avg_acc = np.nanmean(acc_value) * 100 if acc_value else 0.0; avg_grad_norm = np.nanmean(grad_norm_value) if grad_norm_value else float('nan')
        total_time = sum(timer.values()); proportion = { k: f"{int(round(v * 100 / total_time))}%" if total_time > 0 else '0%' for k, v in timer.items() }
        self.print_log(f'\t平均训练损失: {avg_loss:.4f}. 平均训练准确率: {avg_acc:.2f}%.')
        self.print_log(f'\t时间消耗: [数据加载]{proportion["dataloader"]}, [模型计算]{proportion["model"]}, [统计]{proportion["statistics"]}')
        if self.train_writer: # Epoch summary to Tensorboard
            try:
                if not np.isnan(avg_loss): self.train_writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
                if not np.isnan(avg_acc): self.train_writer.add_scalar('train/epoch_acc', avg_acc / 100.0, epoch + 1)
                if not np.isnan(avg_grad_norm): self.train_writer.add_scalar('train/epoch_grad_norm', avg_grad_norm, epoch + 1)
            except Exception as e: self.print_log(f"警告: 写入 TensorBoard epoch 统计失败: {e}", level=logging.WARNING)

        # MultiStepLR 更新
        if hasattr(self, 'scheduler') and self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.MultiStepLR):
             if epoch >= (getattr(self.arg, 'warm_up_epoch', 0) or 0):
                 self.scheduler.step(); current_lr = self.optimizer.param_groups[0]['lr']; self.print_log(f"\tMultiStepLR step, new LR: {current_lr:.6f}", print_time=False)

        # 周期性保存模型
        save_interval = getattr(self.arg, 'save_interval', 0); save_start_epoch = getattr(self.arg, 'save_epoch', 0)
        try: num_epochs_total = int(self.arg.num_epoch); assert num_epochs_total > 0
        except: num_epochs_total = float('inf')
        should_save_interval = (save_interval > 0 and (epoch + 1) >= save_start_epoch and (epoch + 1) % save_interval == 0)
        if should_save_interval:
             model_path = os.path.join(self.arg.work_dir, f'epoch-{epoch+1}_step-{self.global_step}.pt')
             try: state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(); weights = OrderedDict([[k.replace('module.', ''), v.cpu()] for k, v in state_dict.items()]); torch.save(weights, model_path); self.print_log(f'模型已按间隔保存到: {model_path}')
             except Exception as e: self.print_log(f"警告: 按间隔保存模型失败 {model_path}: {e}", logging.WARNING)


    # <<<--- eval 方法 (与 v17.3 相同) --- >>>
    def eval(self, epoch, save_score=False, loader_name=['val'], wrong_file=None, result_file=None):
        """评估模型性能 (修复 UnboundLocalError, 使用 accuracy_score)"""
        f_w = None; f_r = None; csv_writer = None; score_path = None
        try:
            if wrong_file is not None: f_w = open(wrong_file, 'w', encoding='utf-8')
            if result_file is not None: f_r = open(result_file, 'w', encoding='utf-8', newline=''); csv_writer = csv.writer(f_r);
            if csv_writer: csv_writer.writerow(["Sample_Index", "Prediction", "True_Label"])

            self.model.eval()
            self.print_log(f'======> 评估 Epoch: {epoch + 1} on {", ".join(loader_name)}')
            eval_acc_final = 0.0; eval_loss_final = 0.0

            for ln in loader_name:
                loader = self.data_loader.get(ln) # 使用 get 获取
                if not loader: self.print_log(f"警告: 找不到加载器 '{ln}'。", logging.WARNING); continue

                all_loss, all_logits, all_labels, all_indices = [], [], [], []
                process = tqdm(loader, desc=f"Eval {ln} (Epoch {epoch+1})", ncols=100, leave=False)

                for batch_idx, batch_data in enumerate(process):
                    if batch_data is None: continue
                    try:
                        if len(batch_data) == 4: data, label, mask, index = batch_data
                        elif len(batch_data) == 3: data, label, index = batch_data; mask = None
                        else: raise ValueError(f"未知的 batch_data 格式，长度为 {len(batch_data)}")
                        data = data.float().to(self.output_device, non_blocking=True)
                        label_cpu = label.long(); label = label_cpu.to(self.output_device, non_blocking=True)
                        if mask is not None: mask = mask.bool().to(self.output_device, non_blocking=True)
                    except ValueError as e:
                        if batch_idx == 0: self.print_log(f"错误: 处理评估 batch {batch_idx} 数据失败: {e}", logging.ERROR); continue

                    with torch.no_grad():
                        try:
                            output, _ = self.model(data, mask=mask)
                            loss = self.loss(output, label)
                            if not (torch.isnan(loss) or torch.isinf(loss)): all_loss.append(loss.item())
                            all_logits.append(output.cpu()); all_labels.append(label_cpu); all_indices.append(index)
                        except Exception as e:
                             if batch_idx == 0: self.print_log(f"错误: 评估 batch {batch_idx} 模型计算失败: {e}", logging.ERROR); traceback.print_exc(); continue
                process.close()

                if not all_logits: self.print_log(f"警告: 在 {ln} 上没有处理任何数据。", logging.WARNING); continue

                try:
                    logits_all = torch.cat(all_logits, dim=0).numpy(); labels_all = torch.cat(all_labels, dim=0).numpy(); preds_all = np.argmax(logits_all, axis=1)
                    if all_indices and isinstance(all_indices[0], torch.Tensor): indices_all = torch.cat(all_indices, dim=0).numpy()
                    elif all_indices and isinstance(all_indices[0], (np.ndarray, list, int, float)): indices_all = np.concatenate([np.array(i).reshape(-1) for i in all_indices], axis=0)
                    else: indices_all = np.array([]); logger.warning("无法拼接索引列表，类型未知或列表为空。")
                except Exception as e_concat: self.print_log(f"错误: 拼接评估结果/索引时出错: {e_concat}", logging.ERROR); continue

                eval_loss = np.nanmean(all_loss) if all_loss else float('nan')
                try: eval_acc = accuracy_score(labels_all, preds_all)
                except Exception as e_acc_score: logger.error(f"计算 accuracy_score 失败: {e_acc_score}"); eval_acc = 0.0

                if ln == loader_name[-1]: eval_loss_final, eval_acc_final = eval_loss, eval_acc

                writer = getattr(self, 'val_writer', None)
                if self.arg.phase == 'train' and writer:
                    try:
                        if not np.isnan(eval_loss): writer.add_scalar(f'eval/{ln}_epoch_loss', eval_loss, epoch + 1)
                        writer.add_scalar(f'eval/{ln}_epoch_acc', eval_acc, epoch + 1)
                    except Exception as e: self.print_log(f"警告: 写入 TensorBoard eval loss/acc 失败: {e}", level=logging.WARNING)

                self.print_log(f'\t{ln} 集上的平均损失: {eval_loss:.4f}')
                self.print_log(f'\t{ln} 集上的准确率 (Top-1): {eval_acc * 100:.2f}%')

                num_classes_cfg = self.arg.model_args.get('num_classes', 0)
                if num_classes_cfg > 0 and len(labels_all)>0:
                    class_labels = np.arange(num_classes_cfg)
                    for k in self.arg.show_topk:
                         if k > 1 and k < num_classes_cfg:
                             try:
                                 topk_acc = top_k_accuracy_score(labels_all, logits_all, k=k, labels=class_labels)
                                 self.print_log(f'\t{ln} 集上的准确率 (Top-{k}): {topk_acc * 100:.2f}%')
                                 if self.arg.phase == 'train' and writer: writer.add_scalar(f'eval/{ln}_epoch_acc_top{k}', topk_acc, epoch + 1)
                             except Exception as e_topk: self.print_log(f"警告: 计算/记录 Top-{k} 准确率失败: {e_topk}", logging.WARNING)

                if save_score and ln == 'test':
                     if len(indices_all) == len(logits_all):
                         score_dict = {idx.item() if hasattr(idx, 'item') else idx : score_vec for idx, score_vec in zip(indices_all, logits_all)}
                         score_filename = f'final_score_{ln}.pkl'; score_path = os.path.join(self.arg.work_dir, score_filename)
                         try:
                             with open(score_path, 'wb') as f_score: pickle.dump(score_dict, f_score); self.print_log(f"预测分数已保存到: {score_path}")
                         except Exception as e_save_score: self.print_log(f"警告: 保存分数失败 {score_path}: {e_save_score}", logging.WARNING); score_path = None
                     else: self.print_log(f"警告: 索引 ({len(indices_all)}) 和 Logits ({len(logits_all)}) 数量不匹配，无法保存分数文件。", logging.WARNING)

                if (f_w is not None or csv_writer is not None) and ln == 'test':
                     if len(indices_all) == len(labels_all):
                        for i in range(len(labels_all)):
                            pred_i, true_i = preds_all[i].item(), labels_all[i].item(); index_i = indices_all[i].item() if hasattr(indices_all[i], 'item') else indices_all[i]
                            if csv_writer is not None:
                                try: csv_writer.writerow([index_i, pred_i, true_i])
                                except Exception as e_csv: logger.warning(f"写入 CSV 行失败: {e_csv}")
                            if f_w is not None and pred_i != true_i:
                                try: f_w.write(f"{index_i},{pred_i},{true_i}\n")
                                except Exception as e_fw: logger.warning(f"写入 wrong file 失败: {e_fw}")
                     else: self.print_log(f"警告: 索引 ({len(indices_all)}) 和标签 ({len(labels_all)}) 数量不匹配，无法写入结果文件。", logging.WARNING)

                should_save_final_cm = (self.arg.phase == 'test' or (self.arg.phase == 'train' and epoch + 1 == self.arg.num_epoch))
                if num_classes_cfg > 0 and ln == 'test' and should_save_final_cm and len(labels_all) > 0:
                    try:
                        confusion = confusion_matrix(labels_all, preds_all, labels=class_labels); acc_csv_path = os.path.join(self.arg.work_dir, f'final_{ln}_confusion_matrix.csv')
                        with open(acc_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
                             writer_csv = csv.writer(f_csv); list_diag = np.diag(confusion); list_raw_sum = np.sum(confusion, axis=1); each_acc = np.divide(list_diag, list_raw_sum, out=np.zeros_like(list_diag, dtype=float), where=list_raw_sum!=0)
                             writer_csv.writerow(["Class_Index", "Recall"]); [writer_csv.writerow([i, acc_i]) for i, acc_i in enumerate(each_acc)]; writer_csv.writerow([]); writer_csv.writerow(["Confusion Matrix (True \\ Pred)"]); writer_csv.writerow(["True\\Pred"] + [f"Pred_{i}" for i in class_labels]); [writer_csv.writerow([f"True_{i}"] + row.tolist()) for i, row in enumerate(confusion)]
                        self.print_log(f"最终混淆矩阵已保存到: {acc_csv_path}")
                    except Exception as e: self.print_log(f"警告: 计算或保存混淆矩阵失败: {e}", logging.WARNING)

        finally:  # 确保文件总是被尝试关闭
            if f_w is not None:
                try:
                    f_w.close()
                except Exception as e_close_w: # 使用 except 关键字
                    # 记录关闭 wrong_file 时的潜在错误
                    logger.error(f"关闭 wrong_file 时发生错误: {e_close_w}")
            if f_r is not None:
                try:
                    f_r.close()
                except Exception as e_close_r: # 使用 except 关键字
                    # 记录关闭 result_file 时的潜在错误
                    logger.error(f"关闭 result_file 时发生错误: {e_close_r}")

        return eval_acc_final, score_path


    # <<<--- 修改后的 start 方法 (与 v17.3 相同) --- >>>
    def start(self):
        """根据 --phase 参数启动训练或测试流程 (适配 CosineScheduler)"""
        best_score_path = None

        if self.arg.phase == 'train':
             self.print_log('开始训练阶段...')
             self.print_log('参数:\n{}\n'.format(yaml.dump(vars(self.arg), default_flow_style=None, sort_keys=False, allow_unicode=True, Dumper=Dumper)))
             n_iter_per_epoch = 0
             if 'train' in self.data_loader and self.data_loader['train'] is not None:
                 try: n_iter_per_epoch = len(self.data_loader['train'])
                 except: pass
             self.global_step = self.arg.start_epoch * n_iter_per_epoch

             def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
             self.print_log(f'模型可训练参数量: {count_parameters(self.model):,}')
             try: num_epochs_train = int(self.arg.num_epoch); assert num_epochs_train > 0
             except: raise ValueError("必须在 YAML 或命令行设置有效的 num_epoch")
             self.print_log(f"总训练 Epochs: {num_epochs_train}")

             patience = getattr(self.arg, 'early_stop_patience', 0)
             patience_counter = 0
             if patience > 0: self.print_log(f"启用 Early Stopping, patience={patience}")
             else: self.print_log("禁用 Early Stopping。")

             for epoch in range(self.arg.start_epoch, num_epochs_train):
                 self.train(epoch)

                 if (epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == num_epochs_train:
                     self.print_log(f"--- 开始评估 Epoch {epoch + 1} ---")
                     val_acc, _ = self.eval(epoch, save_score=False, loader_name=['val'])
                     self.print_log(f"--- 结束评估 Epoch {epoch + 1} (Val Acc: {val_acc*100:.2f}%) ---")

                     if val_acc > self.best_acc:
                         self.best_acc = val_acc; self.best_acc_epoch = epoch + 1; patience_counter = 0
                         try:
                              self.best_state_dict = OrderedDict([[k.replace('module.', ''), v.cpu()] for k, v in self.model.state_dict().items()])
                              self.print_log(f'*** 新的最佳准确率: {self.best_acc*100:.2f}% (Epoch: {self.best_acc_epoch}). 状态已记录 ***')
                         except Exception as e: self.print_log(f"警告: 记录最佳模型状态失败: {e}", logging.WARNING)
                     else:
                          if patience > 0:
                             patience_counter += 1; self.print_log(f'验证集准确率未提升. EarlyStopping Counter: {patience_counter}/{patience}')
                             if patience_counter >= patience: self.print_log(f'触发 Early Stopping (在 Epoch {epoch + 1})'); break
                          else: self.print_log(f'验证集准确率未提升. EarlyStopping Counter: {patience_counter}/{patience if patience > 0 else "inf"}')

             self.print_log('训练完成。')
             if self.best_state_dict is not None:
                 self.print_log(f'最佳验证准确率 (Top-1): {self.best_acc*100:.2f}% 在 Epoch {self.best_acc_epoch}.')
                 best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                 self.print_log(f'保存最终最佳模型到: {best_model_path}')
                 try: torch.save(self.best_state_dict, best_model_path)
                 except Exception as e: self.print_log(f"错误: 保存最终最佳模型失败: {e}", logging.ERROR)

                 self.print_log('加载最佳模型进行最终测试...')
                 if os.path.exists(best_model_path):
                      try:
                          weights = torch.load(best_model_path, map_location=self.output_device)
                          is_parallel = isinstance(self.model, nn.DataParallel); has_module_prefix = list(weights.keys())[0].startswith('module.')
                          if not is_parallel and has_module_prefix: weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
                          elif is_parallel and not has_module_prefix: weights = OrderedDict([['module.'+k, v] for k, v in weights.items()])
                          self.model.load_state_dict(weights, strict=True); self.print_log(f"最佳模型权重 {best_model_path} 加载成功。")
                          wf = os.path.join(self.arg.work_dir, 'final_test_wrong.txt'); rf = os.path.join(self.arg.work_dir, 'final_test_results.csv')
                          self.print_log('对测试集进行最终评估...')
                          _, test_score_path = self.eval(epoch=self.best_acc_epoch - 1, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
                          best_score_path = test_score_path
                      except Exception as e: self.print_log(f"错误: 加载或测试最佳模型失败: {e}", logging.ERROR); traceback.print_exc()
                 else: self.print_log(f"警告: 找不到刚才保存的最佳模型文件 {best_model_path}。", logging.WARNING)
             else: self.print_log("训练过程中没有记录到有效的最佳模型状态。", logging.WARNING)

        elif self.arg.phase == 'test':
             self.print_log('开始测试阶段...'); assert getattr(self.arg, 'weights', None), '--weights 必须指定'; assert os.path.exists(self.arg.weights), f"找不到权重: {self.arg.weights}"
             wf = self.arg.weights.replace('.pt', '_wrong.txt'); rf = self.arg.weights.replace('.pt', '_results.csv')
             self.print_log('模型:   {}'.format(self.arg.model)); self.print_log('权重: {}'.format(self.arg.weights))
             _, test_score_path = self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
             best_score_path = test_score_path; self.print_log('测试完成。')

        return best_score_path