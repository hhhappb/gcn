# 文件名: processor/processor.py (修改版 - 适应后期融合的分数保存)
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import yaml
try:
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
# import csv # 如果 _save_prediction_details 中用到，则保留
import traceback
import random
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
import torch.nn.functional as F

try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)
    # 确保你的 utils.py 中的 LabelSmoothingCrossEntropy 和 FocalSmoothCE (如果使用) 可用
    from utils import init_seed, import_class, LabelSmoothingCrossEntropy, collate_fn_filter_none 
except ImportError as e:
    print(f"CRITICAL: Failed to import from utils: {e}")
    def collate_fn_filter_none(batch): batch = [item for item in batch if item is not None]; return torch.utils.data.dataloader.default_collate(batch) if batch else None # noqa
    def init_seed(seed, source_description=""): pass
    # --- 根据你的 loss_type 选择默认损失 ---
    # 假设你的 YAML 中 loss_type 可能为 'CE', 'SmoothCE', 'FocalSmoothCE'
    # 为了简单，这里只定义了 CrossEntropyLoss 作为后备
    DefaultLoss = nn.CrossEntropyLoss 
    # 你可能需要在这里定义 FocalSmoothCrossEntropy，如果 utils 导入失败
    # class FocalSmoothCrossEntropy(nn.Module): ...
    # LabelSmoothingCrossEntropy = DefaultLoss 
    import_class = lambda x: None


# CosineLRScheduler 的导入尝试
try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    CosineLRScheduler = None

# --- 你自定义的 FocalSmoothCrossEntropy 损失函数 ---
class FocalSmoothCrossEntropy(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalSmoothCrossEntropy, self).__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing 值 ({smoothing}) 必须在 [0, 1) 范围内")
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.as_tensor([alpha] * num_classes)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        else:
            raise TypeError('Unsupported type for alpha')
        
        if self.alpha is not None and len(self.alpha) != num_classes:
            raise ValueError(f"alpha长度 ({len(self.alpha)}) 与 num_classes ({num_classes}) 不匹配")

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.size(0) != target.size(0):
            raise ValueError(f"输入和目标样本数不匹配: x ({x.size(0)}), target ({target.size(0)})")
        if x.dim() != 2 or target.dim() != 1:
            raise ValueError(f"输入维度应为2 (N,C)，目标维度应为1 (N)。得到: x {x.dim()}, target {target.dim()}")

        logprobs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1) # 用于focal loss的pt

        # --- CE (NLL) part ---
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # --- Smooth CE part ---
        smooth_loss = -logprobs.mean(dim=-1)
        loss_ce_smooth = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # --- Focal part ---
        # gather a_t * (1-p_t)^gamma * log(p_t)
        # Here, we apply focal modulation to the smoothed CE loss
        pt = probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1) # p_t for the true class
        focal_modulator = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(target.device).gather(0, target)
            focal_modulator = alpha_t * focal_modulator
            
        loss = focal_modulator * loss_ce_smooth

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.global_step = 0

        self._setup_device_and_logging() # 包含保存配置
        self.print_log("Processor 初始化开始...")
        self._load_and_prepare_data()
        self._load_and_prepare_model() # loss 在这里初始化

        if self.arg.phase == 'train':
            self.n_iter_per_epoch = len(self.data_loader['train']) if 'train' in self.data_loader and self.data_loader['train'] else 0
            if self.n_iter_per_epoch == 0: self.print_log("警告: 训练数据加载器为空或长度为0。", logging.WARNING)
            self.global_step = getattr(self.arg, 'start_epoch', 0) * self.n_iter_per_epoch
            self._load_optimizer() # 简化后的优化器加载
        else: # 测试或评估阶段，也需要优化器实例（即使不进行step）
            self._load_optimizer()

        self.lr = self.arg.base_lr # 初始化lr，将在每个epoch更新
        self.print_log("Processor 初始化完成。")

    def _setup_device_and_logging(self):
        if not hasattr(self.arg, 'device') or self.arg.device is None:
            self.arg.device = [0] if torch.cuda.is_available() else [-1]
        if not isinstance(self.arg.device, list): self.arg.device = [self.arg.device]

        if self.arg.device[0] == -1 or not torch.cuda.is_available():
            self.output_device = torch.device("cpu")
            self.arg.device_actual = ["cpu"]
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
        
        work_dir = self.arg.work_dir
        os.makedirs(work_dir, exist_ok=True)
        log_file_path = os.path.join(work_dir, 'log.txt')
        filemode = 'w' if self.arg.phase == 'train' and getattr(self.arg, 'start_epoch', 0) == 0 else 'a'
        
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file_path, mode=filemode, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("Processor")

        if self.output_device.type == 'cpu': self.print_log("将在 CPU 上运行。")
        else: self.print_log(f"使用 GPU: {self.arg.device_actual}。主输出设备: {self.output_device}")
        self.print_log(f'工作目录: {work_dir}')
        self.print_log(f"日志文件: {log_file_path} (模式: {filemode})")

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
        if getattr(self.arg, 'print_log', True):
            if hasattr(self, 'logger') and self.logger: self.logger.log(level, msg)
            else: logging.log(level, msg)

    def _save_config(self):
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

    def _create_dataloader(self, feeder_args, batch_size, shuffle, is_train=False):
        from torch.utils.data import DataLoader
        if not self.arg.feeder: raise ValueError("'feeder' 参数未设置。")
        Feeder = import_class(self.arg.feeder)
        
        # 对于单流后期融合，feeder_args['data_path'] 应该是单一模态名
        # main.py 中已经处理了将顶层 modalities (单元素列表) 转换为 data_path 字符串
        
        # 确保 NewFeederUCLA (或你修改后的版本) 的 __init__ 参数与 feeder_args 匹配
        # 这里假设 feeder_args 已经由 main.py 正确准备
        dataset = Feeder(**feeder_args)
        
        num_worker = getattr(self.arg, 'num_worker', 0)
        current_worker_init_fn = None
        if num_worker > 0:
            current_worker_init_fn = lambda worker_id: self._initialize_worker_rng(worker_id)

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_worker,
            drop_last=getattr(self.arg, 'drop_last', True) if is_train else False,
            worker_init_fn=current_worker_init_fn,
            pin_memory=True, collate_fn=collate_fn_filter_none
        )
        log_modality_info = feeder_args.get('data_path', feeder_args.get('modalities', '未知模态'))
        self.print_log(f"{'训练' if is_train else '验证/测试'}数据加载器 '{os.path.basename(self.arg.feeder)}' "
                       f"(模态: {log_modality_info}) 加载成功。样本数: {len(dataset)}")
        return loader

    def _load_and_prepare_data(self):
        self.print_log("开始加载和准备数据...")
        self.data_loader = {}
        try:
            if self.arg.phase == 'train':
                self.data_loader['train'] = self._create_dataloader(
                    self.arg.train_feeder_args, self.arg.batch_size, shuffle=True, is_train=True
                )
            
            val_args_copy = self.arg.test_feeder_args.copy()
            val_args_copy['split'] = 'val' # 验证集用 'val' split
            self.data_loader['val'] = self._create_dataloader(
                val_args_copy, self.arg.test_batch_size, shuffle=False
            )

            if self.arg.phase == 'test': # 如果是测试阶段，额外创建一个 'test' loader
                test_args_copy = self.arg.test_feeder_args.copy()
                test_args_copy['split'] = 'test'
                self.data_loader['test'] = self._create_dataloader(
                     test_args_copy, self.arg.test_batch_size, shuffle=False
                )
            else: # 训练阶段，'test' loader 通常不存在或与 'val' 相同 (取决于是否需要独立测试集评估)
                 # 为了简化，这里可以不创建独立的 test loader，如果评估总是在 val 上进行
                 # 如果你的流程中，训练后会在一个独立的test集上评估，则需要创建它
                 # 假设最终评估在 'val' loader 上进行，或者在 phase='test' 时在 'test' loader 上进行
                 pass

        except Exception as e:
            self.print_log(f"错误: 加载数据时发生严重错误: {e}", logging.CRITICAL); traceback.print_exc(); raise
        self.print_log("数据加载和准备完成。")

    def _load_and_prepare_model(self):
        from torch.nn.parallel import DataParallel
        self.print_log(f"模型将运行在设备: {self.output_device}")
        if not self.arg.model: raise ValueError("'model' 参数未设置。")
        Model = import_class(self.arg.model)
        try: 
            model_file_path = inspect.getfile(Model)
            if os.path.exists(model_file_path) and os.path.isfile(model_file_path): shutil.copy2(model_file_path, self.arg.work_dir)
        except Exception: pass
        if not self.arg.model_args: raise ValueError("'model_args' 参数未设置或为空。")
        # 对于单流，model_args['num_input_dim'] 应该已经是3 (由main.py设置)
        self.model = Model(model_cfg=self.arg.model_args) 
        
        loss_type = getattr(self.arg, 'loss_type', 'CE').upper()
        if loss_type == 'SMOOTHCE':
            smoothing_val = getattr(self.arg, 'label_smoothing', 0.1)
            self.loss = LabelSmoothingCrossEntropy(smoothing=smoothing_val).to(self.output_device)
            self.print_log(f"损失函数: SmoothCE (平滑: {smoothing_val})")
        elif loss_type == 'FOCALSMOOTHCE':
            smoothing_val = getattr(self.arg, 'label_smoothing', 0.1)
            gamma_val = getattr(self.arg, 'focal_gamma', 2.0)
            # alpha_val 可以从 self.arg.focal_alpha 获取，如果需要的话
            self.loss = FocalSmoothCrossEntropy(
                num_classes=self.arg.model_args['num_classes'],
                smoothing=smoothing_val, 
                gamma=gamma_val
            ).to(self.output_device)
            self.print_log(f"损失函数: FocalSmoothCE (平滑: {smoothing_val}, Gamma: {gamma_val})")
        else: # 默认 CE
            self.loss = nn.CrossEntropyLoss().to(self.output_device)
            self.print_log(f"损失函数: CrossEntropyLoss")


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
            self.model = DataParallel(self.model, device_ids=self.arg.device_actual, output_device=self.output_device)
            self.print_log(f'模型已在 GPUs {self.arg.device_actual} 上启用 DataParallel。')

    def _load_optimizer(self):
        """加载优化器 (仅支持 SGD 和 AdamW)。"""
        optimizer_type = self.arg.optimizer.lower() # 假设 optimizer 参数总是在 arg 中
        lr = self.arg.base_lr
        wd = self.arg.weight_decay
        
        # TD-GCN 直接获取所有参数，不区分是否 requires_grad，这在多数情况下没问题
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        if not params_to_optimize:
            self.print_log("警告: 模型中没有可训练的参数。", logging.WARNING)
            # 创建一个无参数的优化器以避免后续代码出错
            self.optimizer = optim.AdamW([], lr=lr, weight_decay=wd)
            return
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                params_to_optimize,
                lr=lr,
                momentum=self.arg.momentum, # 假设 momentum 总是在 arg 中
                nesterov=self.arg.nesterov, # 假设 nesterov 总是在 arg 中
                weight_decay=wd)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                params_to_optimize,
                lr=lr,
                weight_decay=wd)
        else:
            self.print_log(f"错误: 不支持的优化器 '{optimizer_type}'. 请使用 'sgd' 或 'adamw'.", logging.CRITICAL)
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        self.print_log(f"优化器: {optimizer_type.upper()} (初始lr={lr:.2e}, wd={wd:.1e})")
        # 由于是手动管理学习率，不再需要 PyTorch 的 scheduler
        self.scheduler = None
        self.lr_scheduler_each_step = None

    def _adjust_learning_rate_for_warmup(self, epoch): # 保持原名，实现手动warmup和decay
        """手动调整学习率，包括预热和基于step的衰减。在每个训练epoch开始时调用。"""
        base_lr = self.arg.base_lr
        warmup_epochs = getattr(self.arg, 'warm_up_epoch', 0)
        calculated_lr = base_lr

        if epoch < warmup_epochs:
            warmup_lr_init = getattr(self.arg, 'warmup_lr', 1e-6)
            if warmup_epochs > 0: # 避免除以零
                calculated_lr = warmup_lr_init + (base_lr - warmup_lr_init) * (epoch + 1) / warmup_epochs
        else:
            # MultiStep 衰减逻辑
            decay_steps = getattr(self.arg, 'step', [])
            num_decays = np.sum(epoch >= np.array(decay_steps)) # 假设step是0-based epoch索引
            calculated_lr = base_lr * (getattr(self.arg, 'lr_decay_rate', 0.1) ** num_decays)

        # 应用计算出的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = calculated_lr
        
        self.lr = calculated_lr # 更新 self.lr 属性
        return calculated_lr

    def record_time(self): self.cur_time = time.time()
    def split_time(self): split = time.time() - self.cur_time; self.record_time(); return split

    def train(self, epoch, save_model=False):
        self.model.train()
        # 在epoch开始时调用手动的学习率调整函数
        current_lr_for_display = self._adjust_learning_rate_for_warmup(epoch) # <<< 调用修改后的函数

        # 日志打印调整，避免重复打印warmup信息
        if epoch < getattr(self.arg, 'warm_up_epoch', 0) :
             self.print_log(f'======> 训练 Epoch: {epoch + 1} (Warmup)，当前学习率: {current_lr_for_display:.8f}')
        else:
             self.print_log(f'======> 训练 Epoch: {epoch + 1}，当前学习率: {current_lr_for_display:.8f}')


        loader = self.data_loader['train']
        if not loader: self.print_log("错误: 训练数据加载器为空！", logging.ERROR); return

        loss_values, acc_values, grad_norm_values = [], [], []
        if self.train_writer: self.train_writer.add_scalar('meta/epoch', epoch + 1, epoch + 1) # 使用 epoch + 1
        
        self.record_time(); timer = {'dataloader': 0.0, 'model': 0.0, 'statistics': 0.0}
        if self.n_iter_per_epoch == 0 and loader: 
            try: self.n_iter_per_epoch = len(loader)
            except: pass # 避免在某些iterable上失败

        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch}", ncols=120, leave=False) 
        log_interval = getattr(self.arg, 'log_interval', 50)
        if self.n_iter_per_epoch > 0 and log_interval > self.n_iter_per_epoch : # 确保至少在epoch末尾打印一次
            log_interval = self.n_iter_per_epoch


        for batch_idx, batch_data in enumerate(process):
            self.global_step += 1
            # 移除了 self.lr_scheduler_each_step.step() 因为我们不再使用它
            
            # ... (数据加载和前向、反向传播逻辑保持你之前的版本) ...
            if batch_data is None: timer['dataloader'] += self.split_time(); continue
            try: 
                if len(batch_data) == 4: data, label, mask_from_batch, index = batch_data 
                elif len(batch_data) == 3: data, label, index = batch_data; mask_from_batch = None
                else: raise ValueError(f"批次数据元素数量不为3或4: {len(batch_data)}")
            except Exception as e_unpack: 
                self.print_log(f"警告: Batch {batch_idx} 数据解包失败: {e_unpack}", logging.WARNING)
                timer['dataloader'] += self.split_time(); continue
            timer['dataloader'] += self.split_time()
            data = data.float().to(self.output_device,non_blocking=True)
            label = label.long().to(self.output_device,non_blocking=True)
            mask_to_model = mask_from_batch.bool().to(self.output_device,non_blocking=True) if mask_from_batch is not None else None
            self.record_time()
            try:
                output, _ = self.model(data, mask=mask_to_model)
                loss = self.loss(output, label)
                if torch.isnan(loss) or torch.isinf(loss): 
                    self.print_log(f"警告: Batch {batch_idx} 损失 NaN/Inf！跳过此批次。", logging.WARNING)
                    timer['model'] += self.split_time(); continue
                self.optimizer.zero_grad(); loss.backward()
                total_norm = 0.0; valid_grad = True
                for p in self.model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): valid_grad = False; break
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5 if valid_grad else float('nan')
                if valid_grad:
                    if not np.isnan(total_norm): grad_norm_values.append(total_norm)
                    if getattr(self.arg, 'grad_clip', True) and getattr(self.arg, 'grad_max', 1.0) > 0: 
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.arg.grad_max)
                    self.optimizer.step()
                else: 
                    self.print_log(f"警告: Batch {batch_idx} 梯度 NaN/Inf，跳过优化步骤。", logging.WARNING)
                    self.optimizer.zero_grad()
            except Exception as e_train_step: 
                self.print_log(f"错误: 训练步骤 (Batch {batch_idx}) 失败: {e_train_step}", logging.ERROR); traceback.print_exc()
                timer['model'] += self.split_time(); continue
            timer['model'] += self.split_time()
            loss_item = loss.item(); loss_values.append(loss_item)
            with torch.no_grad(): 
                _, pred = torch.max(output.data, 1)
                acc_item = torch.mean((pred == label.data).float()).item()
                acc_values.append(acc_item)
            grad_postfix_str = f"{total_norm:.2f}" if not np.isnan(total_norm) else "NaN"
            process.set_postfix_str(f"Loss: {loss_item:.3f}, Acc: {acc_item:.2f}, Grad: {grad_postfix_str}")
            self.record_time()
            if log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == self.n_iter_per_epoch):
                lr_curr_for_log = self.optimizer.param_groups[0]['lr'] 
                grad_norm_batch_str = f"{total_norm:.4f}" if not np.isnan(total_norm) else "NaN"                
                log_line = (f"Epoch: [{epoch+1}][{batch_idx+1}/{self.n_iter_per_epoch}]\t"
                            f"Loss: {loss_item:.4f}\tAcc: {acc_item:.3f}\tLR: {lr_curr_for_log:.8f}\tGradNorm: {grad_norm_batch_str}")
                # self.print_log(log_line, print_time=False) # tqdm 已经打印时间了
                if self.train_writer: 
                    self.train_writer.add_scalar('批次训练/损失', loss_item, self.global_step)
                    self.train_writer.add_scalar('批次训练/准确率', acc_item, self.global_step)
                    if not np.isnan(total_norm): self.train_writer.add_scalar('批次训练/梯度范数', total_norm, self.global_step)
                    self.train_writer.add_scalar('学习率/迭代', lr_curr_for_log, self.global_step)
            timer['statistics'] += self.split_time()
        
        process.close()
        avg_loss = np.nanmean(loss_values) if loss_values else float('nan')
        avg_acc = np.nanmean(acc_values) * 100 if acc_values else 0.0
        avg_grad_epoch_val = np.nanmean(grad_norm_values) if grad_norm_values else float('nan')
        avg_grad_epoch_str = f"{avg_grad_epoch_val:.4f}" if not np.isnan(avg_grad_epoch_val) else "NaN"
        total_time_epoch = sum(timer.values())
        prop = {k:f"{int(round(v*100/total_time_epoch))}%" if total_time_epoch>0 else "0%" for k,v in timer.items()}
        self.print_log(f'\t平均训练损失: {avg_loss:.4f}. 平均训练准确率: {avg_acc:.2f}%. 平均梯度范数: {avg_grad_epoch_str}')
        self.print_log(f'\t时间消耗: [数据加载]{prop["dataloader"]}, [网络计算]{prop["model"]}, [统计]{prop["statistics"]}')
        if self.train_writer:
            if not np.isnan(avg_loss): self.train_writer.add_scalar('Epoch训练/平均损失', avg_loss, epoch + 1)
            if not np.isnan(avg_acc): self.train_writer.add_scalar('Epoch训练/平均准确率', avg_acc / 100.0, epoch + 1)
            if not np.isnan(avg_grad_epoch_val): self.train_writer.add_scalar('Epoch训练/平均梯度范数', avg_grad_epoch_val, epoch + 1)
            self.train_writer.add_scalar('学习率/Epoch', self.optimizer.param_groups[0]['lr'], epoch + 1)

        # 移除了 self.scheduler.step() 的调用，因为现在是手动调整
        # 如果需要打印学习率衰减的日志，可以在 _adjust_learning_rate_for_warmup 中当衰减发生时打印
        if epoch >= getattr(self.arg, 'warm_up_epoch', 0):
            # 检查是否在衰减点
            decay_steps = getattr(self.arg, 'step', [])
            if epoch +1 in decay_steps : # epoch是0-based, step中的通常是1-based epoch num
                 self.print_log(f"\t学习率在 Epoch {epoch+1} 后可能已衰减。新学习率: {self.optimizer.param_groups[0]['lr']:.8f}")

    def eval(self, epoch, save_score_final_eval=False, loader_name=['val'], wrong_file=None, result_file=None):
        self.model.eval()
        self.print_log(f"--- 开始评估 Epoch {epoch + 1} ---") 
        self.print_log(f'======> 评估 Epoch: {epoch + 1} 于数据集: {", ".join(loader_name)}')
        final_eval_acc = 0.0; final_score_path_for_return = None # 修改变量名以区分

        for ln_idx, ln in enumerate(loader_name): 
            loader = self.data_loader.get(ln)
            if not loader: self.print_log(f"警告: 找不到名为 '{ln}' 的数据加载器。", logging.WARNING); continue
            
            all_loss, all_logits, all_labels, all_indices = [], [], [], []
            process_desc = f"评估 {ln} (Epoch {epoch+1})"
            process = tqdm(loader, desc=process_desc, ncols=100, leave=False)

            for batch_data in process:
                if batch_data is None: self.print_log(f"警告: 在评估数据集 '{ln}' 时遇到空批次，已跳过。", logging.WARNING); continue
                try: 
                    # 确保与 train 方法中的解包逻辑一致
                    if len(batch_data) == 4:
                        data, label_cpu, mask_from_batch, index = batch_data 
                    elif len(batch_data) == 3:
                        data, label_cpu, index = batch_data
                        mask_from_batch = None
                    else:
                        raise ValueError(f"批次数据元素数量不为3或4: {len(batch_data)}")
                except Exception as e_eval_unpack: self.print_log(f"警告: 评估数据批次解包失败 ({ln}): {e_eval_unpack}", logging.WARNING); continue
                
                data=data.float().to(self.output_device,non_blocking=True)
                label_gpu=label_cpu.long().to(self.output_device,non_blocking=True)
                mask_to_model = None
                if mask_from_batch is not None: mask_to_model=mask_from_batch.bool().to(self.output_device,non_blocking=True)
                
                with torch.no_grad():
                    try:
                        output, _ = self.model(data, mask=mask_to_model)
                        loss = self.loss(output, label_gpu)
                        if not (torch.isnan(loss) or torch.isinf(loss)): all_loss.append(loss.item())
                        all_logits.append(output.cpu())
                        all_labels.append(label_cpu)   
                        all_indices.append(index.cpu()) 
                    except Exception as e_eval_fwd: self.print_log(f"错误: 模型在评估数据集 '{ln}'上前向传播失败: {e_eval_fwd}", logging.ERROR); continue
            process.close()

            if not all_logits: self.print_log(f"警告: 在数据集 '{ln}' 上没有处理任何有效的模型输出 logits。", logging.WARNING); continue
            
            logits_all_np = torch.cat(all_logits, dim=0).numpy()
            labels_all_np = torch.cat(all_labels, dim=0).numpy()
            preds_all_np = np.argmax(logits_all_np, axis=1)
            indices_all_np = torch.cat(all_indices, dim=0).numpy() if all_indices and all_indices[0] is not None else np.array([])
            
            eval_loss = np.nanmean(all_loss) if all_loss else float('nan')
            eval_acc = accuracy_score(labels_all_np, preds_all_np) if len(labels_all_np) > 0 else 0.0
            
            if ln_idx == 0: final_eval_acc = eval_acc
            self.print_log(f'\t数据集 [{ln}]: 平均损失: {eval_loss:.4f}, Top-1 准确率: {eval_acc * 100:.2f}%')
            
            if self.arg.phase == 'train' and self.val_writer:
                if not np.isnan(eval_loss): self.val_writer.add_scalar(f'评估/{ln}_epoch_loss', eval_loss, epoch + 1)
                self.val_writer.add_scalar(f'评估/{ln}_epoch_acc_top1', eval_acc, epoch + 1)
            
            num_classes = self.arg.model_args.get('num_classes', 0)
            if num_classes > 0 and len(labels_all_np) > 0:
                for k_val in getattr(self.arg, 'show_topk', [1]): 
                    if k_val == 1: continue
                    if k_val > 1 and k_val < num_classes:
                        try:
                            top_k_acc_val = top_k_accuracy_score(labels_all_np, logits_all_np, k=k_val, labels=np.arange(num_classes))
                            self.print_log(f'\t数据集 [{ln}]: Top-{k_val} 准确率: {top_k_acc_val * 100:.2f}%')
                            if self.arg.phase == 'train' and self.val_writer: self.val_writer.add_scalar(f'评估/{ln}_epoch_acc_top{k_val}', top_k_acc_val, epoch + 1)
                        except Exception as e_topk: self.print_log(f"警告: 计算 Top-{k_val} 准确率失败 (数据集: {ln}): {e_topk}", logging.WARNING)
            
            # --- 修改保存分数的文件名和逻辑 ---
            # save_score_final_eval 通常在 Processor.start() 的最终测试阶段设为 True
            # ln 通常是 'test' (如果 test_loader 存在) 或 'val' (如果用 val_loader 做最终评估)
            if save_score_final_eval and ln in ['test', 'val'] and len(indices_all_np) == len(logits_all_np):
                # 固定文件名为 'epoch1_test_score.pkl' 以便 ensemble.py 查找
                # 注意： "epoch1" 这个名字是 Hyperformer ensemble.py 的硬编码，可能不代表实际epoch
                # 如果你的 ensemble.py 被修改为接受其他文件名，这里可以相应调整
                score_file_name = 'epoch1_test_score.pkl' 
                current_score_path = os.path.join(self.arg.work_dir, score_file_name)
                
                # 保存的格式是 {'indices': ..., 'scores': ..., 'labels': ...}
                # 确保你的 ensemble.py 能够读取这个格式中的 'scores' 和 'labels'
                score_data_to_save = {
                    'indices': indices_all_np,
                    'scores': logits_all_np, 
                    'labels': labels_all_np
                }
                try:
                    with open(current_score_path, 'wb') as f_score: 
                        pickle.dump(score_data_to_save, f_score)
                    self.print_log(f"评估分数 ({ln}) 已保存到: {current_score_path}")
                    if ln_idx == 0 : # 通常第一个loader是主要的，将其路径返回给main
                        final_score_path_for_return = current_score_path
                except Exception as e_save_score: 
                    self.print_log(f"警告: 保存评估分数 ({ln}) 失败: {e_save_score}", logging.WARNING)
                    if ln_idx == 0: final_score_path_for_return = None
            # --- 结束修改 ---
            
            if (wrong_file or result_file) and ln_idx == 0 and len(indices_all_np) == len(labels_all_np): 
                self._save_prediction_details(indices_all_np, preds_all_np, labels_all_np, wrong_file, result_file)
            
            is_final_eval_for_cm = (self.arg.phase == 'test' or (self.arg.phase == 'train' and epoch + 1 == self.arg.num_epoch))
            if num_classes > 0 and ln_idx == 0 and is_final_eval_for_cm and len(labels_all_np) > 0:
                 try:
                     cm = confusion_matrix(labels_all_np, preds_all_np, labels=np.arange(num_classes))
                     cm_file = os.path.join(self.arg.work_dir, f'混淆矩阵_{ln}_epoch{epoch+1}_final.csv')
                     np.savetxt(cm_file, cm, delimiter=',', fmt='%d')
                     self.print_log(f"最终混淆矩阵 ({ln}) 已保存到: {cm_file}")
                 except Exception as e_cm: self.print_log(f"警告: 保存混淆矩阵 ({ln}) 失败: {e_cm}", logging.WARNING)

        if self.arg.phase == 'train' and ((epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == self.arg.num_epoch):
            self.print_log(f"--- 结束评估 Epoch {epoch + 1} (验证集准确率: {final_eval_acc*100:.2f}%) ---")

        return final_eval_acc, final_score_path_for_return # 返回路径

    def _save_prediction_details(self, indices, preds, trues, wrong_fp, result_fp):
        # 这个方法保持不变
        if result_fp:
            try:
                import csv # 确保导入
                with open(result_fp, 'w', encoding='utf-8', newline='') as fr_csv:
                    csv_w = csv.writer(fr_csv); csv_w.writerow(["Sample_Index", "Prediction", "True_Label"])
                    for i in range(len(trues)): csv_w.writerow([indices[i].item(), preds[i].item(), trues[i].item()])
            except Exception as e: self.print_log(f"警告: 保存结果文件 {result_fp} 失败: {e}", logging.WARNING)
        if wrong_fp:
            try:
                with open(wrong_fp, 'w', encoding='utf-8') as fw_txt:
                    for i in range(len(trues)):
                        if preds[i].item() != trues[i].item(): fw_txt.write(f"{indices[i].item()},{preds[i].item()},{trues[i].item()}\n")
            except Exception as e: self.print_log(f"警告: 保存错误文件 {wrong_fp} 失败: {e}", logging.WARNING)

    def start(self):
        final_score_path_for_main = None
        if self.arg.phase == 'train':
            self.print_log('开始训练阶段...')
            self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

            num_epochs = int(self.arg.num_epoch)
            patience = getattr(self.arg, 'early_stop_patience', 0)
            patience_counter = 0
            self.print_log(f"总训练 Epochs: {num_epochs}, 起始 Epoch: {self.arg.start_epoch + 1}")
            if patience > 0: self.print_log(f"启用 Early Stopping, patience={patience}")

            for epoch in range(self.arg.start_epoch, num_epochs):
                self.train(epoch)
                perform_eval = (epoch + 1) % self.arg.eval_interval == 0 or (epoch + 1) == num_epochs

                if perform_eval:
                    # 始终在 'val' loader 上评估来更新最佳模型
                    val_acc, _ = self.eval(epoch, save_score_final_eval=False, loader_name=['val'])
                    if val_acc > self.best_acc:
                        self.best_acc, self.best_acc_epoch = val_acc, epoch + 1
                        patience_counter = 0
                        best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                        try:
                            state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
                            torch.save(state_dict_to_save, best_model_path)
                            self.print_log(f'*** 新的最佳准确率: {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}). 模型已保存到 {best_model_path} ***')
                        except Exception as e:
                            self.print_log(f"警告: 保存最佳模型失败: {e}", logging.WARNING)
                    elif patience > 0:
                        patience_counter += 1
                        self.print_log(f'验证集准确率未提升. EarlyStopping Counter: {patience_counter}/{patience}')
                        if patience_counter >= patience:
                            self.print_log(f'触发 Early Stopping (Epoch {epoch + 1})')
                            break
            
            self.print_log('训练完成。')

            # --- 训练结束后，加载最佳模型进行最终评估并保存分数 ---
            if self.best_acc_epoch > 0:
                self.print_log(f'训练中最佳验证准确率 (Top-1): {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}).')
                best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                if os.path.exists(best_model_path):
                    self.print_log(f'加载最佳模型 {best_model_path} 进行最终测试/分数保存...')
                    try:
                        # 1. 加载权重到CPU以安全处理前缀
                        loaded_weights = torch.load(best_model_path, map_location='cpu')

                        # 2. 更稳健地处理 DataParallel 前缀
                        # 当前的 self.model 实例在训练后应该是 DataParallel 包装的（如果你用了多GPU）
                        is_current_model_dp = isinstance(self.model, torch.nn.DataParallel)
                        
                        # 检查加载的权重是否带有 'module.' 前缀
                        # 需要注意，如果 loaded_weights 为空或不是字典，这里会出错，但 torch.load 通常返回字典
                        keys_in_loaded_weight_start_with_module = False
                        if loaded_weights and isinstance(loaded_weights, dict) and len(loaded_weights.keys()) > 0:
                            keys_in_loaded_weight_start_with_module = list(loaded_weights.keys())[0].startswith('module.')

                        final_weights_to_load = OrderedDict()
                        if is_current_model_dp: # 当前模型是 DataParallel 实例
                            if not keys_in_loaded_weight_start_with_module:
                                # 模型是DP, 但权重文件中的键名不带 'module.' -> 给权重键名加上 'module.'
                                self.print_log("  当前模型为 DataParallel，但加载的权重不含 'module.' 前缀。将为权重键名添加前缀。")
                                for k, v in loaded_weights.items():
                                    final_weights_to_load['module.' + k] = v
                            else:
                                # 模型是DP, 权重也是DP -> 直接使用
                                self.print_log("  当前模型为 DataParallel，加载的权重已含 'module.' 前缀。直接使用。")
                                final_weights_to_load = loaded_weights
                        else: # 当前模型不是 DataParallel 实例
                            if keys_in_loaded_weight_start_with_module:
                                # 模型不是DP, 但权重文件中的键名带 'module.' -> 从权重键名移除 'module.'
                                self.print_log("  当前模型非 DataParallel，但加载的权重含 'module.' 前缀。将从权重键名移除前缀。")
                                for k, v in loaded_weights.items():
                                    final_weights_to_load[k[7:]] = v # 移除 'module.'
                            else:
                                # 模型不是DP, 权重也不是DP -> 直接使用
                                self.print_log("  当前模型非 DataParallel，加载的权重不含 'module.' 前缀。直接使用。")
                                final_weights_to_load = loaded_weights
                        
                        # 3. 将处理好的权重加载到模型
                        self.model.load_state_dict(final_weights_to_load)
                        self.print_log("最佳模型权重已成功加载到 self.model。")
                        
                        # 4. !!! 关键：在进行任何评估之前，立即设置模型为评估模式 !!!
                        self.model.eval()
                        self.print_log(f"模型已设置为评估模式: model.training = {self.model.training}") # 应该输出 False

                        # 5. 确定评估用的loader
                        # 为了调试，强制使用 'val' loader，与训练中获取 best_acc 时一致
                        loader_for_final_scores = ['val']
                        self.print_log(f"最终评估将强制使用 '{loader_for_final_scores}' loader 进行准确率对比。")

                        wf = os.path.join(self.arg.work_dir, 'final_eval_best_model_wrong.txt')
                        rf = os.path.join(self.arg.work_dir, 'final_eval_best_model_results.csv')
                        
                        # 调用 eval 进行最终评估
                        # eval 方法内部也会调用 self.model.eval()，但在这里多调用一次是好的实践
                        final_eval_acc_loaded_best, final_score_path_for_main = self.eval(
                            epoch=self.best_acc_epoch -1, # epoch参数主要用于日志
                            save_score_final_eval=getattr(self.arg, 'save_score', True), # 保存分数文件
                            loader_name=loader_for_final_scores,
                            wrong_file=wf, result_file=rf
                        )
                        self.print_log(f"加载最佳模型后，在 {loader_for_final_scores[0]} 上直接评估的准确率: {final_eval_acc_loaded_best*100:.2f}%")

                    except Exception as e:
                        self.print_log(f"错误: 加载或评估最佳模型时失败: {e}", logging.ERROR)
                        traceback.print_exc()
                else:
                    self.print_log(f"警告: 最佳模型文件 {best_model_path} 未找到。", logging.WARNING)
            else:
                self.print_log("训练中未记录有效的最佳模型。", logging.WARNING)

        elif self.arg.phase == 'test':
            self.print_log('开始测试阶段...')
            if not self.arg.weights or not os.path.exists(self.arg.weights):
                self.print_log(f"错误: 测试阶段必须指定有效的 --weights 文件路径。", logging.CRITICAL)
                return None
            self.print_log(f'加载模型: {self.arg.model}, 测试权重: {self.arg.weights}')
            
            try:
                # 1. 加载权重到CPU以安全处理前缀
                loaded_weights = torch.load(self.arg.weights, map_location='cpu')

                # 2. 更稳健地处理 DataParallel 前缀
                is_current_model_dp = isinstance(self.model, torch.nn.DataParallel)
                keys_in_loaded_weight_start_with_module = False
                if loaded_weights and isinstance(loaded_weights, dict) and len(loaded_weights.keys()) > 0:
                    keys_in_loaded_weight_start_with_module = list(loaded_weights.keys())[0].startswith('module.')
                
                final_weights_to_load = OrderedDict()
                if is_current_model_dp:
                    if not keys_in_loaded_weight_start_with_module:
                        self.print_log("  测试阶段：当前模型为 DataParallel，但加载的权重不含 'module.' 前缀。将为权重键名添加前缀。")
                        for k, v in loaded_weights.items(): final_weights_to_load['module.' + k] = v
                    else:
                        self.print_log("  测试阶段：当前模型为 DataParallel，加载的权重已含 'module.' 前缀。直接使用。")
                        final_weights_to_load = loaded_weights
                else:
                    if keys_in_loaded_weight_start_with_module:
                        self.print_log("  测试阶段：当前模型非 DataParallel，但加载的权重含 'module.' 前缀。将从权重键名移除前缀。")
                        for k, v in loaded_weights.items(): final_weights_to_load[k[7:]] = v
                    else:
                        self.print_log("  测试阶段：当前模型非 DataParallel，加载的权重不含 'module.' 前缀。直接使用。")
                        final_weights_to_load = loaded_weights
                
                self.model.load_state_dict(final_weights_to_load)
                self.print_log("测试权重已成功加载到 self.model。")

                # 3. !!! 关键：设置模型为评估模式 !!!
                self.model.eval()
                self.print_log(f"测试阶段：模型已设置为评估模式: model.training = {self.model.training}")

            except Exception as e:
                self.print_log(f"错误: 测试阶段加载权重失败: {e}", logging.CRITICAL)
                traceback.print_exc()
                return None

            base_name = os.path.basename(self.arg.weights).replace('.pt','')
            wf = os.path.join(self.arg.work_dir, f'{base_name}_wrong.txt')
            rf = os.path.join(self.arg.work_dir, f'{base_name}_results.csv')
            
            loader_for_test_phase = ['val'] # 默认使用 'val' loader
            if 'test' in self.data_loader and self.arg.test_feeder_args.get('split') == 'test':
                loader_for_test_phase = ['test']
            self.print_log(f"测试阶段将使用 loader: {loader_for_test_phase}")


            test_acc, final_score_path_for_main = self.eval(
                epoch=self.best_acc_epoch if hasattr(self, 'best_acc_epoch') else 0, # 使用最佳epoch或0
                save_score_final_eval=getattr(self.arg, 'save_score', True),
                loader_name=loader_for_test_phase,
                wrong_file=wf, result_file=rf
            )
            self.print_log(f'测试完成。在 {loader_for_test_phase[0]} 上的准确率: {test_acc*100:.2f}%')
        
        elif self.arg.phase == 'model_size':
             self.print_log(f'模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}')
             self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
        else:
            self.print_log(f"未知的运行阶段: {self.arg.phase}", logging.ERROR)

        if hasattr(self, 'train_writer') and self.train_writer: # 检查属性是否存在
            try: self.train_writer.close()
            except Exception: pass
        if hasattr(self, 'val_writer') and self.val_writer: # 检查属性是否存在
            try: self.val_writer.close()
            except Exception: pass
            
        return final_score_path_for_main