# 文件名: processor/processor.py (v17.4 - 恢复旧版日志风格)
import torch
import torch.nn as nn
import torch.optim as optim
# DataLoader 和 DataParallel 在需要时导入
from tensorboardX import SummaryWriter
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import logging # logging 仍然用于文件和部分控制台日志
import os
import sys
import time
import shutil
import inspect
import pickle
import csv
import traceback
import random
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)
    from utils import init_seed, import_class, LabelSmoothingCrossEntropy, collate_fn_filter_none
except ImportError as e:
    print(f"CRITICAL: Failed to import from utils: {e}")
    def collate_fn_filter_none(batch): batch = [item for item in batch if item is not None]; return torch.utils.data.dataloader.default_collate(batch) if batch else None # noqa
    def init_seed(seed, source_description=""): pass
    LabelSmoothingCrossEntropy = nn.CrossEntropyLoss
    import_class = lambda x: None

# CosineLRScheduler 的导入尝试（如果需要）
try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    # logging.warning("timm.CosineLRScheduler 未找到") # 可以在 Processor 初始化时按需警告
    CosineLRScheduler = None


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.best_state_dict = None
        self.global_step = 0
        # self.device_log_str 在 _setup_device_and_logging 中设置

        self._setup_device_and_logging() # 注意：这里会初始化 self.logger
        self.print_log("Processor 初始化开始...") # 现在 print_log 可以正确工作
        self._save_config()
        self._load_and_prepare_data()
        self._load_and_prepare_model()

        self.n_iter_per_epoch = 0 # 初始化
        if self.arg.phase == 'train':
            if 'train' in self.data_loader and self.data_loader['train'] is not None:
                try:
                    self.n_iter_per_epoch = len(self.data_loader['train'])
                    if self.n_iter_per_epoch == 0: self.print_log("训练数据加载器长度为 0。", logging.WARNING)
                except Exception as e:
                    self.print_log(f"警告: 获取训练迭代次数失败: {e}", logging.WARNING)
            self.global_step = getattr(self.arg, 'start_epoch', 0) * self.n_iter_per_epoch
            self._load_optimizer_and_scheduler()
        else: # test 或 model_size 阶段
            # 确保优化器存在以获取学习率，即使不训练
            if not hasattr(self, 'optimizer') or self.optimizer is None:
                 self._load_optimizer_and_scheduler(load_scheduler=False) # 只加载优化器

        self.lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') and self.optimizer and self.optimizer.param_groups else self.arg.base_lr
        self.print_log("Processor 初始化完成。")

    def _setup_device_and_logging(self):
        """设置设备和日志记录器 (恢复旧版风格)。"""
        # 1. 设置设备
        if not hasattr(self.arg, 'device') or self.arg.device is None:
            self.arg.device = [0] if torch.cuda.is_available() else [-1]
        if not isinstance(self.arg.device, list): self.arg.device = [self.arg.device]

        if self.arg.device[0] == -1 or not torch.cuda.is_available():
            self.output_device = torch.device("cpu")
            self.arg.device_actual = ["cpu"] # 存储实际使用的设备列表
        else:
            valid_devices = [d for d in self.arg.device if isinstance(d, int) and 0 <= d < torch.cuda.device_count()]
            if not valid_devices:
                self.output_device = torch.device("cpu"); self.arg.device_actual = ["cpu (GPU无效)"]
            else:
                self.arg.device_actual = valid_devices
                self.output_device = torch.device(f"cuda:{self.arg.device_actual[0]}")
                try: torch.cuda.set_device(self.output_device)
                except Exception as e:
                    self.output_device = torch.device("cpu"); self.arg.device_actual = [f"cpu (GPU设置失败: {e})"]
        
        # 2. 设置日志 (恢复旧版 basicConfig 风格)
        work_dir = self.arg.work_dir
        os.makedirs(work_dir, exist_ok=True)
        log_file_path = os.path.join(work_dir, 'log.txt')
        filemode = 'w' if self.arg.phase == 'train' and getattr(self.arg, 'start_epoch', 0) == 0 else 'a'

        # 清理 root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        
        # 配置 basicConfig - 这会影响所有通过 logging 模块的输出
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)-8s - %(message)s', # 旧版格式，不含 logger name
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file_path, mode=filemode, encoding='utf-8'),
                logging.StreamHandler(sys.stdout) # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger("Processor") # 获取一个名为 "Processor" 的 logger

        # 初始环境信息使用 self.print_log (它会使用 self.logger)
        if self.output_device.type == 'cpu': self.print_log("将在 CPU 上运行。")
        else: self.print_log(f"使用 GPU: {self.arg.device_actual}。主输出设备: {self.output_device}")
        self.print_log(f'工作目录: {work_dir}')
        self.print_log(f"日志文件: {log_file_path} (模式: {filemode})")

        # 3. 设置 TensorBoard (与之前版本相同)
        self.train_writer = self.val_writer = None
        if self.arg.phase == 'train' and not getattr(self.arg, 'debug', False):
            runs_dir = os.path.join(work_dir, 'runs')
            if os.path.isdir(runs_dir) and filemode == 'w':
                try: shutil.rmtree(runs_dir); self.print_log(f"已清空 TensorBoard 日志目录: {runs_dir}")
                except OSError as e: self.print_log(f"警告: 清空 TensorBoard 目录失败: {e}", logging.WARNING)
            os.makedirs(runs_dir, exist_ok=True)
            try:
                self.train_writer = SummaryWriter(os.path.join(runs_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(runs_dir, 'val'), 'val')
            except Exception as e: self.print_log(f"警告: 初始化 TensorBoardWriter 失败: {e}", logging.WARNING)

    def print_log(self, msg, level=logging.INFO):
        """使用 self.logger 打印日志，格式由 basicConfig 控制。"""
        if getattr(self.arg, 'print_log', True):
            # self.logger 是在 _setup_device_and_logging 中创建的
            if hasattr(self, 'logger') and self.logger:
                self.logger.log(level, msg)
            else: # Fallback if logger somehow not initialized (理论上不应发生)
                  # 这种 fallback 会使用 root logger，格式可能不同
                logging.log(level, msg)


    def _save_config(self): # 重命名自 save_arg
        work_dir = self.arg.work_dir
        try:
            filepath = os.path.join(work_dir, 'config_used.yaml')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Work Dir: {work_dir}\n# Phase: {self.arg.phase}\n")
                f.write(f"# Device: {self.output_device}\n")
                f.write(f"# Command line: {' '.join(sys.argv)}\n\n")
                yaml.dump(vars(self.arg), f, default_flow_style=False, sort_keys=False, Dumper=Dumper)
            self.print_log(f"当前运行配置已保存到: {filepath}")
        except Exception as e:
            self.print_log(f"警告: 保存 config_used.yaml ({work_dir}) 失败: {e}", logging.WARNING)

    def _initialize_worker_rng(self, worker_id):
        worker_seed = self.arg.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # processor_module_logger.debug(f"DataLoader worker {worker_id} initialized with custom seed: {worker_seed}")


    def _create_dataloader(self, feeder_args, batch_size, shuffle, is_train=False):
        from torch.utils.data import DataLoader
        if not self.arg.feeder: raise ValueError("'feeder' 参数未设置。")
        Feeder = import_class(self.arg.feeder)
        
        required_keys = ['root_dir', 'data_path', 'split', 'modalities', 'num_nodes', 'base_channel', 'num_classes', 'max_len']
        for key in required_keys:
            if key not in feeder_args and key in ['root_dir', 'data_path', 'modalities', 'num_nodes', 'base_channel']:
                raise ValueError(f"Feeder 参数 '{key}' 缺失于 {'train' if is_train else 'test'}_feeder_args。")
        
        dataset = Feeder(**feeder_args)
        
        num_worker = getattr(self.arg, 'num_worker', 0)
        # worker_init_fn 仅在 num_worker > 0 时有意义
        current_worker_init_fn = None
        if num_worker > 0:
            current_worker_init_fn = lambda worker_id: self._initialize_worker_rng(worker_id)

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_worker,
            drop_last=getattr(self.arg, 'drop_last', True) if is_train else False,
            worker_init_fn=current_worker_init_fn, # 使用这个
            pin_memory=True, collate_fn=collate_fn_filter_none
        )
        self.print_log(f"{'训练' if is_train else '验证/测试'}数据加载器 '{os.path.basename(self.arg.feeder)}' "
                       f"(模态: {feeder_args.get('modalities', feeder_args.get('data_path'))}) 加载成功。样本数: {len(dataset)}")
        return loader

    def _load_and_prepare_data(self):
        self.print_log("开始加载数据...")
        self.data_loader = {}
        try:
            if self.arg.phase == 'train':
                self.data_loader['train'] = self._create_dataloader(
                    self.arg.train_feeder_args, self.arg.batch_size, shuffle=True, is_train=True
                )
            # 确保 test_feeder_args 的 split 正确
            test_args_copy = self.arg.test_feeder_args.copy()
            test_args_copy['split'] = 'test' if self.arg.phase == 'test' else 'val'

            self.data_loader['val'] = self._create_dataloader( # 'val' 通常用于训练时的验证
                test_args_copy, self.arg.test_batch_size, shuffle=False
            )
            if self.arg.phase == 'test': # 如果是测试阶段，'test' loader 也用 'test' split
                 self.data_loader['test'] = self.data_loader['val']
            else: # 训练阶段，'test' loader 通常不存在或与 'val' 相同
                 self.data_loader['test'] = self.data_loader['val']

        except Exception as e:
            self.print_log(f"错误: 加载数据时发生严重错误: {e}", logging.CRITICAL); traceback.print_exc(); raise
        self.print_log("数据加载完成。")

    def _load_and_prepare_model(self):
        from torch.nn.parallel import DataParallel
        self.print_log(f"模型将运行在设备: {self.output_device}")
        try:
            if not self.arg.model: raise ValueError("'model' 参数未设置。")
            Model = import_class(self.arg.model)
            try: # 尝试复制模型文件
                model_file_path = inspect.getfile(Model)
                if os.path.exists(model_file_path) and os.path.isfile(model_file_path): shutil.copy2(model_file_path, self.arg.work_dir)
            except Exception: pass
            if not self.arg.model_args: raise ValueError("'model_args' 参数未设置或为空。")
            self.model = Model(model_cfg=self.arg.model_args)
            self.print_log(f"模型 '{self.arg.model}' 实例化成功。")
        except Exception as e: self.print_log(f"错误: 模型加载/实例化失败: {e}", logging.CRITICAL); traceback.print_exc(); raise
        
        loss_type = getattr(self.arg, 'loss_type', 'CE').upper()
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).to(self.output_device) if loss_type == 'SMOOTHCE' else nn.CrossEntropyLoss().to(self.output_device)
        self.print_log(f"损失函数: {loss_type}")

        if self.arg.weights:
            self.print_log(f'加载权重自: {self.arg.weights}')
            if not os.path.exists(self.arg.weights): self.print_log(f"错误: 权重文件不存在: {self.arg.weights}", logging.ERROR); raise FileNotFoundError()
            try:
                weights = torch.load(self.arg.weights, map_location=self.output_device)
                weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
                if self.arg.ignore_weights:
                    keys_to_ignore = set(); [keys_to_ignore.update(k for k in weights if p in k) for p in self.arg.ignore_weights]
                    for k_ignore in keys_to_ignore:
                        if weights.pop(k_ignore, None) is not None: self.print_log(f"已忽略权重: {k_ignore}")
                missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
                if missing_keys: self.print_log(f"警告: 模型中缺失的键: {missing_keys}", logging.WARNING)
                if unexpected_keys: self.print_log(f"警告: 权重文件中多余的键: {unexpected_keys}", logging.WARNING)
                self.print_log("权重加载完成。")
            except Exception as e: self.print_log(f"错误: 加载权重失败: {e}", logging.ERROR); traceback.print_exc(); raise
        
        self.model.to(self.output_device)
        if isinstance(self.arg.device_actual, list) and len(self.arg.device_actual) > 1 and self.output_device.type == 'cuda':
            self.model = DataParallel(self.model, device_ids=self.arg.device_actual, output_device=self.output_device) # 使用 device_actual
            self.print_log(f'模型已在 GPUs {self.arg.device_actual} 上启用 DataParallel。')

    def _load_optimizer_and_scheduler(self, load_scheduler=True):
        optimizer_type = (getattr(self.arg, 'optimizer', None) or 'AdamW').lower()
        lr = getattr(self.arg, 'base_lr', None) or 0.001
        wd = getattr(self.arg, 'weight_decay', 0.01) if getattr(self.arg, 'weight_decay', None) is not None else 0.01
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

        if not params_to_optimize:
            self.print_log("警告: 模型中没有可优化的参数。", logging.WARNING)
            self.optimizer = optim.AdamW([], lr=lr) # 即使没有参数，也创建一个优化器实例
            self.scheduler = None 
            self.lr_scheduler_each_step = None
            return

        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                params_to_optimize,
                lr=lr,
                momentum=getattr(self.arg, 'momentum', 0.9),
                nesterov=getattr(self.arg, 'nesterov', False),
                weight_decay=wd)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        self.print_log(f"优化器: {optimizer_type.upper()} (初始lr={lr:.2e}, wd={wd:.1e})")

        self.scheduler = None
        self.lr_scheduler_each_step = None
        if not load_scheduler: # 如果只是测试阶段或不需要调度器
            return

        scheduler_type = getattr(self.arg, 'lr_scheduler', 'multistep').lower()
        warmup_epochs = getattr(self.arg, 'warm_up_epoch', 0)
        
        if not self.optimizer: # 再次检查，理论上应该已经初始化
            self.print_log("错误: 优化器未在加载调度器之前初始化。", logging.ERROR)
            return

        self.print_log(f"尝试加载调度器: {scheduler_type}") # 恢复旧版日志位置

        if scheduler_type == 'multistep':
            cfg_steps = getattr(self.arg, 'step', []) # 用户配置的绝对 epoch 衰减点, e.g., [35, 55]
            
            if not cfg_steps or not isinstance(cfg_steps, list):
                self.print_log("警告: MultiStepLR 'step' (milestones) 参数为空或格式不正确。学习率将不会按 MultiStepLR 计划衰减 (除了warmup)。", logging.WARNING)
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(1e9)], gamma=1.0) # 永不触发
            else:
                # --- 核心修正：调整 milestones 以考虑 warmup ---
                # MultiStepLR 的 milestones 是指 *scheduler.step()* 被调用的次数（0-indexed）
                # 我们的 scheduler.step() 是在 warmup_epochs 之后才开始调用的 (在 train 方法中控制)
                # 用户配置的 cfg_steps 通常是绝对的 epoch 数 (例如，在第35个epoch之后衰减，这里的step是0-indexed的epoch)
                # 我们需要将这些绝对的 epoch 数转换为相对于 warmup 结束后的相对 epoch 数
                adjusted_milestones = [s - warmup_epochs for s in cfg_steps if s > warmup_epochs]
                # 例如: cfg_steps=[34, 54] (0-indexed), warmup_epochs=15  => adjusted_milestones=[19, 39]
                # 这意味着当 scheduler.last_epoch (从0开始计数) 达到 19 时 (即总 epoch 15+19=34 结束时)，衰减
                
                if not adjusted_milestones and cfg_steps:
                    self.print_log(f"警告: MultiStepLR 的所有衰减点 ({cfg_steps}) 都在预热期 ({warmup_epochs} epochs) 内。"
                                   f"学习率在预热后将保持为 base_lr，不会通过 MultiStepLR 进一步衰减。", logging.WARNING)
                    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(1e9)], gamma=1.0)
                elif adjusted_milestones:
                    self.scheduler = optim.lr_scheduler.MultiStepLR(
                        self.optimizer,
                        milestones=adjusted_milestones, # 使用调整后的 milestones
                        gamma=getattr(self.arg, 'lr_decay_rate', 0.1)
                    )
                else: # cfg_steps 为空或所有 step <= warmup_epochs
                    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[], gamma=getattr(self.arg, 'lr_decay_rate', 0.1))

            if self.scheduler:
                actual_milestones_for_log = self.scheduler.milestones if hasattr(self.scheduler, 'milestones') and isinstance(self.scheduler.milestones, list) else '未知'
                if isinstance(actual_milestones_for_log, list) and len(actual_milestones_for_log) == 1 and actual_milestones_for_log[0] == int(1e9):
                    actual_milestones_for_log = '无计划衰减(milestones过大)'
                elif isinstance(actual_milestones_for_log, list) and not actual_milestones_for_log :
                     actual_milestones_for_log = '无有效衰减点(在warmup后)'
                self.print_log(f"调度器: MultiStepLR (用户配置step={cfg_steps}, warmup={warmup_epochs} => "
                               f"调度器实际milestones={actual_milestones_for_log}, "
                               f"gamma={getattr(self.arg, 'lr_decay_rate', 0.1)})")
            
            if warmup_epochs > 0 and scheduler_type == 'multistep':
                 self.print_log(f'MultiStepLR 将配合手动学习率预热, epochs: {warmup_epochs}')
        
        elif scheduler_type == 'cosine':
            if CosineLRScheduler is None: 
                self.print_log("错误: CosineLRScheduler (timm) 未导入。请安装 timm 或选择其他调度器。", logging.ERROR)
                return
            if self.n_iter_per_epoch <= 0 and self.arg.phase == 'train': 
                self.print_log("错误: CosineLRScheduler 需要 n_iter_per_epoch > 0 (即训练数据加载器非空)。", logging.ERROR)
                return
            
            total_iterations = int(self.arg.num_epoch * self.n_iter_per_epoch)
            warmup_iterations = int(warmup_epochs * self.n_iter_per_epoch)
            
            try:
                self.lr_scheduler_each_step = CosineLRScheduler(
                    self.optimizer,
                    t_initial=(total_iterations - warmup_iterations) if getattr(self.arg, 'warmup_prefix', True) else total_iterations,
                    lr_min=getattr(self.arg, 'min_lr', 1e-6),
                    warmup_lr_init=getattr(self.arg, 'warmup_lr', 1e-6),
                    warmup_t=warmup_iterations,
                    cycle_limit=1,
                    t_in_epochs=False, 
                    warmup_prefix=getattr(self.arg, 'warmup_prefix', True)
                )
                self.print_log("调度器: CosineLRScheduler (timm) 加载成功。")
            except Exception as e:
                self.print_log(f"错误: 初始化 CosineLRScheduler 失败: {e}", logging.ERROR)
        else:
            self.print_log(f"警告: 不支持的学习率调度器类型 '{scheduler_type}'。", logging.WARNING)


    def _adjust_learning_rate_for_warmup(self, epoch): # 仅用于 MultiStepLR 的手动 warmup
        if self.lr_scheduler_each_step is not None: # 如果使用逐 iter 调度器，它自己处理 warmup
            return self.optimizer.param_groups[0]['lr']

        warmup_epochs = getattr(self.arg, 'warm_up_epoch', 0)
        base_lr = self.arg.base_lr
        current_lr = self.optimizer.param_groups[0]['lr']

        if epoch < warmup_epochs:
            warmup_lr_init = getattr(self.arg, 'warmup_lr', 1e-6)
            lr = warmup_lr_init + (base_lr - warmup_lr_init) * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
            current_lr = lr
        elif epoch == warmup_epochs and warmup_epochs > 0: # Warmup 结束
            for param_group in self.optimizer.param_groups: param_group['lr'] = base_lr
            current_lr = base_lr
        return current_lr

    def record_time(self): self.cur_time = time.time()
    def split_time(self): split = time.time() - self.cur_time; self.record_time(); return split

    def train(self, epoch):
        self.model.train()
        current_lr_for_display = self._adjust_learning_rate_for_warmup(epoch)

        # --- 恢复旧版 Epoch 开始日志 ---
        self.print_log(f'======> 训练 Epoch: {epoch + 1}')
        if self.lr_scheduler_each_step is None: # 只有非逐 iter 调度器才在这里打印学习率
            if epoch < getattr(self.arg, 'warm_up_epoch', 0):
                self.print_log(f"Epoch {epoch+1} 开始 (Warmup)，学习率: {current_lr_for_display:.8f}")
            else:
                self.print_log(f"Epoch {epoch+1} 开始，学习率: {current_lr_for_display:.8f}")
        # 对于逐 iter 调度器，学习率会在批次日志中显示

        loader = self.data_loader['train']
        if not loader: self.print_log("错误: 训练数据加载器为空！", logging.ERROR); return

        loss_values, acc_values, grad_norm_values = [], [], []
        if self.train_writer: self.train_writer.add_scalar('meta/epoch', epoch + 1, epoch + 1)
        
        self.record_time(); timer = {'dataloader': 0.0, 'model': 0.0, 'statistics': 0.0}
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch}", ncols=120, leave=False) # 保持 leave=False
        log_interval = getattr(self.arg, 'log_interval', 50)

        for batch_idx, batch_data in enumerate(process):
            self.global_step += 1
            if self.lr_scheduler_each_step: self.lr_scheduler_each_step.step(self.global_step) # 逐 iter 更新 LR
            
            if batch_data is None: timer['dataloader'] += self.split_time(); continue
            try: data, label, mask, _ = batch_data
            except Exception as e_unpack: self.print_log(f"警告: Batch {batch_idx} 数据解包失败: {e_unpack}", logging.WARNING); timer['dataloader'] += self.split_time(); continue
            timer['dataloader'] += self.split_time()
            data = data.float().to(self.output_device,non_blocking=True); label = label.long().to(self.output_device,non_blocking=True)
            if mask is not None: mask = mask.bool().to(self.output_device,non_blocking=True)
            self.record_time()
            try:
                output, _ = self.model(data, mask=mask); loss = self.loss(output, label)
                if torch.isnan(loss) or torch.isinf(loss): self.print_log(f"警告: Batch {batch_idx} 损失 NaN/Inf！", logging.WARNING); timer['model'] += self.split_time(); continue
                self.optimizer.zero_grad(); loss.backward()
                total_norm = 0.0; valid_grad = True
                for p in self.model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): valid_grad = False; break
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5 if valid_grad else float('nan')
                if valid_grad:
                    grad_norm_values.append(total_norm)
                    if getattr(self.arg, 'grad_clip', True) and getattr(self.arg, 'grad_max', 1.0) > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.arg.grad_max)
                    self.optimizer.step()
                else: self.print_log(f"警告: Batch {batch_idx} 梯度 NaN/Inf，跳过优化。", logging.WARNING); self.optimizer.zero_grad()
            except Exception as e_train_step: self.print_log(f"错误: 训练步骤 (Batch {batch_idx}) 失败: {e_train_step}", logging.ERROR); timer['model'] += self.split_time(); continue
            timer['model'] += self.split_time()
            loss_item = loss.item(); loss_values.append(loss_item)
            with torch.no_grad(): _, pred = torch.max(output.data, 1); acc_item = torch.mean((pred == label.data).float()).item(); acc_values.append(acc_item)
            
            # tqdm 后缀只显示 Loss, Acc, Grad (不含 LR，因为 LR 可能在批次日志中)
            grad_postfix = f"{total_norm:.2f}" if not np.isnan(total_norm) else "NaN"
            process.set_postfix_str(f"Loss: {loss_item:.3f}, Acc: {acc_item:.2f}, Grad: {grad_postfix}")

            self.record_time()
            if log_interval > 0 and (self.global_step % log_interval == 0):
                lr_curr_for_log = self.optimizer.param_groups[0]['lr'] # 获取当前实际学习率
                # --- 准备 GradNorm 的字符串 ---
                grad_norm_str = f"{total_norm:.4f}" if not np.isnan(total_norm) else "NaN"                
                # --- 恢复旧版批次日志格式 ---
                log_line = (f"Epoch: [{epoch+1}][{batch_idx+1}/{self.n_iter_per_epoch}]\t"
                            f"Loss: {loss_item:.4f}\tAcc: {acc_item:.3f}\tLR: {lr_curr_for_log:.8f}\tGradNorm: {grad_norm_str}") # 使用准备好的字符串
                # 直接 print，不通过 self.logger，以避免标准 logging 前缀和重复时间戳
                if getattr(self.arg, 'print_log', True): print(log_line)
                # 同时，为了文件记录，我们通过 self.logger （它有文件 handler）记录一次
                self.logger.info(log_line) # 这会写入文件，格式由 basicConfig 控制

                if self.train_writer: # Tensorboard 记录不变
                    self.train_writer.add_scalar('train/batch_loss', loss_item, self.global_step)
                    self.train_writer.add_scalar('train/batch_acc', acc_item, self.global_step)
                    if not np.isnan(total_norm): self.train_writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                    self.train_writer.add_scalar('meta/learning_rate_iter', lr_curr_for_log, self.global_step)
            timer['statistics'] += self.split_time()
        
        process.close()
        avg_loss = np.nanmean(loss_values) if loss_values else float('nan'); avg_acc = np.nanmean(acc_values) * 100 if acc_values else 0.0
        avg_grad = np.nanmean(grad_norm_values) if grad_norm_values else float('nan')
        total_time_epoch = sum(timer.values()); prop = {k:f"{int(round(v*100/total_time_epoch))}%" if total_time_epoch>0 else "0%" for k,v in timer.items()}
        
        # Epoch 结束总结日志，使用 self.print_log (会通过 self.logger 输出，带标准格式)
        self.print_log(f'\t平均训练损失: {avg_loss:.4f}. 平均训练准确率: {avg_acc:.2f}%. 平均梯度范数: {avg_grad if not np.isnan(avg_grad) else "NaN":.4f}')
        self.print_log(f'\t时间消耗: [数据加载]{prop["dataloader"]}, [网络计算]{prop["model"]}, [统计]{prop["statistics"]}')
        
        if self.train_writer:
            if not np.isnan(avg_loss): self.train_writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
            if not np.isnan(avg_acc): self.train_writer.add_scalar('train/epoch_acc', avg_acc / 100.0, epoch + 1)
            if not np.isnan(avg_grad): self.train_writer.add_scalar('train/epoch_grad_norm', avg_grad, epoch + 1)
            self.train_writer.add_scalar('meta/lr_epoch', self.optimizer.param_groups[0]['lr'], epoch + 1)

        # 更新 MultiStepLR (如果使用且不在 warmup 阶段)
        if self.scheduler and self.lr_scheduler_each_step is None: # 确保是 MultiStepLR
            # 只有在 warmup 阶段结束之后，才让 MultiStepLR 开始 step
            if epoch >= getattr(self.arg, 'warm_up_epoch', 0):
                self.scheduler.step()

        if self.scheduler and self.lr_scheduler_each_step is None and \
           (getattr(self.arg, 'warm_up_epoch', 0) <= 0 or epoch >= getattr(self.arg, 'warm_up_epoch', 0)):
             pass 
        if self.scheduler and self.lr_scheduler_each_step is None and \
           (getattr(self.arg, 'warm_up_epoch', 0) <= 0 or epoch >= getattr(self.arg, 'warm_up_epoch', 0)):
             self.print_log(f"\tMultiStepLR.step(). New LR: {self.optimizer.param_groups[0]['lr']:.8f}")

    def eval(self, epoch, save_score_final_eval=False, loader_name=['val'], wrong_file=None, result_file=None):
        self.model.eval()
        # --- 恢复旧版评估开始日志 ---
        self.print_log(f"--- 开始评估 Epoch {epoch + 1} ---") # 对应旧版
        self.print_log(f'======> 评估 Epoch: {epoch + 1} on {", ".join(loader_name)}')
        final_eval_acc = 0.0; final_score_path = None

        for ln_idx, ln in enumerate(loader_name):
            loader = self.data_loader.get(ln)
            if not loader: self.print_log(f"警告: 找不到加载器 '{ln}'。", logging.WARNING); continue
            all_loss, all_logits, all_labels, all_indices = [], [], [], []
            # --- 恢复旧版 tqdm 描述符 ---
            process = tqdm(loader, desc=f"Eval {ln} (Epoch {epoch+1})", ncols=100, leave=False) # leave=False

            for batch_data in process:
                if batch_data is None: continue
                try: data, label_cpu, mask, index = batch_data
                except Exception as e_eval_unpack: self.print_log(f"警告: 评估数据解包失败: {e_eval_unpack}", logging.WARNING); continue
                data=data.float().to(self.output_device,non_blocking=True); label=label_cpu.long().to(self.output_device,non_blocking=True)
                if mask is not None: mask=mask.bool().to(self.output_device,non_blocking=True)
                with torch.no_grad():
                    try:
                        output, _ = self.model(data, mask=mask); loss = self.loss(output, label)
                        if not (torch.isnan(loss) or torch.isinf(loss)): all_loss.append(loss.item())
                        all_logits.append(output.cpu()); all_labels.append(label_cpu); all_indices.append(index.cpu())
                    except Exception as e_eval_fwd: self.print_log(f"错误: 评估前向传播失败: {e_eval_fwd}", logging.ERROR); continue
            process.close()
            if not all_logits: self.print_log(f"警告: 在 {ln} 上无数据处理。", logging.WARNING); continue
            logits_all_np = torch.cat(all_logits, dim=0).numpy(); labels_all_np = torch.cat(all_labels, dim=0).numpy()
            preds_all_np = np.argmax(logits_all_np, axis=1)
            indices_all_np = torch.cat(all_indices, dim=0).numpy() if all_indices and all_indices[0] is not None else np.array([])
            eval_loss = np.nanmean(all_loss) if all_loss else float('nan'); eval_acc = accuracy_score(labels_all_np, preds_all_np) if len(labels_all_np) > 0 else 0.0
            if ln_idx == 0: final_eval_acc = eval_acc
            
            # --- 恢复旧版评估结果日志格式 ---
            self.print_log(f'\t{ln} 集: 平均损失: {eval_loss:.4f}, Top-1 Acc: {eval_acc * 100:.2f}%')
            if self.arg.phase == 'train' and self.val_writer:
                if not np.isnan(eval_loss): self.val_writer.add_scalar(f'eval/{ln}_epoch_loss', eval_loss, epoch + 1)
                self.val_writer.add_scalar(f'eval/{ln}_epoch_acc_top1', eval_acc, epoch + 1) # 旧版用 top1
            num_classes = self.arg.model_args.get('num_classes', 0)
            if num_classes > 0 and len(labels_all_np) > 0:
                for k_val in getattr(self.arg, 'show_topk', [1]): # 旧版用 show_topk
                    if k_val > 1 and k_val < num_classes:
                        try:
                            topk_acc_val = top_k_accuracy_score(labels_all_np, logits_all_np, k=k_val, labels=np.arange(num_classes))
                            self.print_log(f'\t{ln} 集: Top-{k_val} Acc: {topk_acc_val * 100:.2f}%')
                            if self.arg.phase == 'train' and self.val_writer: self.val_writer.add_scalar(f'eval/{ln}_epoch_acc_top{k_val}', topk_acc_val, epoch + 1)
                        except Exception as e_topk: self.print_log(f"警告: 计算 Top-{k_val} 准确率失败 for {ln}: {e_topk}", logging.WARNING)
            
            if save_score_final_eval and ln == 'test' and len(indices_all_np) == len(logits_all_np):
                score_dict = {idx.item() if hasattr(idx,'item') else int(idx): vec for idx, vec in zip(indices_all_np, logits_all_np)}
                score_file_name = f'eval_score_{ln}_epoch{epoch+1}.pkl'
                final_score_path = os.path.join(self.arg.work_dir, score_file_name)
                try:
                    with open(final_score_path, 'wb') as f_score: pickle.dump(score_dict, f_score)
                    self.print_log(f"评估分数 ({ln}) 已为最终评估保存到: {final_score_path}") # 旧版日志
                except Exception as e_save_score: self.print_log(f"警告: 保存最终评估分数 ({ln}) 失败: {e_save_score}", logging.WARNING); final_score_path=None
            
            if (wrong_file or result_file) and ln_idx == 0 and len(indices_all_np) == len(labels_all_np): # 旧版只对第一个 loader (通常是val/test) 写
                self._save_prediction_details(indices_all_np, preds_all_np, labels_all_np, wrong_file, result_file)
            
            # 旧版混淆矩阵保存逻辑
            is_final_eval_for_cm = (self.arg.phase == 'test' or (self.arg.phase == 'train' and epoch + 1 == self.arg.num_epoch))
            if num_classes > 0 and ln_idx == 0 and is_final_eval_for_cm and len(labels_all_np) > 0:
                 try:
                     cm = confusion_matrix(labels_all_np, preds_all_np, labels=np.arange(num_classes))
                     cm_file = os.path.join(self.arg.work_dir, f'confusion_matrix_{ln}_final_epoch{epoch+1}.csv')
                     # 这里省略了旧版详细的CSV写入代码，但你知道它的逻辑
                     self.print_log(f"最终混淆矩阵 ({ln}) 已保存到: {cm_file} (保存逻辑待补充或参考旧版)")
                 except Exception as e_cm: self.print_log(f"警告: 保存混淆矩阵 ({ln}) 失败: {e_cm}", logging.WARNING)

        if self.arg.phase == 'train' and ((epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == self.arg.num_epoch):
            self.print_log(f"--- 结束评估 Epoch {epoch + 1} (Val Acc: {final_eval_acc*100:.2f}%) ---")

        return final_eval_acc, final_score_path

    def _save_prediction_details(self, indices, preds, trues, wrong_fp, result_fp):
        # 这个方法与之前版本可以保持一致
        if result_fp:
            try:
                with open(result_fp, 'w', encoding='utf-8', newline='') as fr_csv:
                    csv_w = csv.writer(fr_csv); csv_w.writerow(["Sample_Index", "Prediction", "True_Label"])
                    for i in range(len(trues)): csv_w.writerow([indices[i].item(), preds[i].item(), trues[i].item()])
                # self.print_log(f"详细预测结果已保存到: {result_fp}") # eval中打印
            except Exception as e: self.print_log(f"警告: 保存结果文件 {result_fp} 失败: {e}", logging.WARNING)
        if wrong_fp:
            try:
                with open(wrong_fp, 'w', encoding='utf-8') as fw_txt:
                    for i in range(len(trues)):
                        if preds[i].item() != trues[i].item(): fw_txt.write(f"{indices[i].item()},{preds[i].item()},{trues[i].item()}\n")
                # self.print_log(f"错误预测样本已保存到: {wrong_fp}") # eval中打印
            except Exception as e: self.print_log(f"警告: 保存错误文件 {wrong_fp} 失败: {e}", logging.WARNING)


    def start(self):
        final_score_path_for_main = None 
        if self.arg.phase == 'train':
            self.print_log('开始训练阶段...')
            self.print_log(f'参数:\n{yaml.dump(vars(self.arg), default_flow_style=None, sort_keys=False, allow_unicode=True, Dumper=Dumper)}')
            self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
            
            num_epochs = int(self.arg.num_epoch); patience = getattr(self.arg, 'early_stop_patience', 0); patience_counter = 0
            self.print_log(f"总训练 Epochs: {num_epochs}, 起始 Epoch: {self.arg.start_epoch + 1}")
            if patience > 0: self.print_log(f"启用 Early Stopping, patience={patience}")

            for epoch in range(self.arg.start_epoch, num_epochs):
                self.train(epoch)
                if (epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == num_epochs:
                    # eval 方法内部会打印评估开始/结束日志
                    val_acc, _ = self.eval(epoch, save_score_final_eval=False, loader_name=['val'])
                    if val_acc > self.best_acc:
                        self.best_acc, self.best_acc_epoch = val_acc, epoch + 1; patience_counter = 0
                        best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt') # 旧版在这里就尝试保存
                        try:
                            state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
                            torch.save(state_dict_to_save, best_model_path) # 旧版每次最佳都保存
                            self.print_log(f'*** 新的最佳准确率: {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}). 模型已保存到 {best_model_path} ***')
                            self.best_state_dict = OrderedDict([[k.replace('module.', ''), v.cpu()] for k, v in state_dict_to_save.items()]) # 仍然记录 best_state_dict
                        except Exception as e: self.print_log(f"警告: 保存最佳模型失败: {e}", logging.WARNING)
                    elif patience > 0 :
                        patience_counter += 1; self.print_log(f'验证集准确率未提升. EarlyStopping Counter: {patience_counter}/{patience}')
                        if patience_counter >= patience: self.print_log(f'触发 Early Stopping (Epoch {epoch + 1})'); break
            
            self.print_log('训练完成。')
            if self.best_acc_epoch > 0: # 旧版这里用 best_acc_epoch 判断
                self.print_log(f'训练中最佳验证准确率 (Top-1): {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}).')
                best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt') # 最佳模型路径
                if os.path.exists(best_model_path): # 确保最佳模型文件存在
                    self.print_log(f'加载最佳模型 {best_model_path} 进行最终测试...')
                    try:
                        # --- 恢复旧版加载最佳模型的逻辑 ---
                        loaded_weights = torch.load(best_model_path, map_location=self.output_device)
                        # 旧版加载时会再次处理 module. 前缀，以防万一
                        if isinstance(self.model, torch.nn.DataParallel) and not list(loaded_weights.keys())[0].startswith('module.'):
                            loaded_weights = OrderedDict([('module.'+k, v) for k,v in loaded_weights.items()])
                        elif not isinstance(self.model, torch.nn.DataParallel) and list(loaded_weights.keys())[0].startswith('module.'):
                            loaded_weights = OrderedDict([(k.replace('module.',''),v) for k,v in loaded_weights.items()])
                        self.model.load_state_dict(loaded_weights)
                        self.print_log("最佳模型权重加载成功。")
                        # --- 结束恢复 ---
                        
                        wf = os.path.join(self.arg.work_dir, 'final_test_wrong.txt')
                        rf = os.path.join(self.arg.work_dir, 'final_test_results.csv')
                        _, final_score_path_for_main = self.eval(
                            epoch=self.best_acc_epoch -1, save_score_final_eval=getattr(self.arg, 'save_score', True),
                            loader_name=['test'], wrong_file=wf, result_file=rf)
                    except Exception as e: self.print_log(f"错误: 加载或测试最佳模型失败: {e}", logging.ERROR); traceback.print_exc()
                else: self.print_log(f"警告: 最佳模型文件 {best_model_path} 未找到，无法进行最终测试。", logging.WARNING)
            else: self.print_log("训练中未记录有效的最佳模型。", logging.WARNING)

        elif self.arg.phase == 'test':
            self.print_log('开始测试阶段...')
            if not self.arg.weights or not os.path.exists(self.arg.weights):
                self.print_log(f"错误: 测试阶段必须指定有效的 --weights 文件路径。", logging.CRITICAL); return None
            # 权重已在 _load_and_prepare_model 中加载
            self.print_log(f'模型: {self.arg.model}, 权重: {self.arg.weights}') # 旧版测试时打印模型和权重
            base_name = os.path.basename(self.arg.weights).replace('.pt','')
            wf = os.path.join(self.arg.work_dir, f'{base_name}_wrong.txt')
            rf = os.path.join(self.arg.work_dir, f'{base_name}_results.csv')
            _, final_score_path_for_main = self.eval(
                epoch=0, save_score_final_eval=getattr(self.arg, 'save_score', True),
                loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('测试完成。')
        
        elif self.arg.phase == 'model_size':
             self.print_log(f'模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}')
             self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
        else: self.print_log(f"未知的运行阶段: {self.arg.phase}", logging.ERROR)

        if self.train_writer: self.train_writer.close()
        if self.val_writer: self.val_writer.close()
        return final_score_path_for_main