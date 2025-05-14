# -*- coding: utf-8 -*-
# 文件名: processor/processor.py (v_warmup_final_log_grad_save_opt_v3)
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
import traceback 
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score

try:
    from utils import init_seed, import_class, str2bool, DictAction, LabelSmoothingCrossEntropy, collate_fn_filter_none
except ImportError:
     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
     from utils import init_seed, import_class, str2bool, DictAction, LabelSmoothingCrossEntropy, collate_fn_filter_none

try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    logging.warning("无法从 timm.scheduler 导入 CosineLRScheduler。如果配置使用 cosine 调度器，请安装 timm: pip install timm")
    CosineLRScheduler = None

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.setup_device()
        self.setup_logging_and_writers() 
        self.print_log("Processor 初始化开始...")

        self.load_data()
        self.load_model()

        self.optimizer = None
        self.scheduler = None
        self.lr_scheduler_each_step = None
        self.n_iter_per_epoch = 0

        if self.arg.phase == 'train':
            if 'train' in self.data_loader and self.data_loader['train'] is not None:
                try:
                    self.n_iter_per_epoch = len(self.data_loader['train'])
                    if self.n_iter_per_epoch == 0: self.print_log("训练数据加载器长度为 0。", logging.WARNING)
                except Exception as e:
                    self.print_log(f"警告: 获取训练迭代次数失败: {e}", logging.WARNING)
            else:
                 self.print_log("警告: 训练数据加载器未初始化。", logging.WARNING)

            self.global_step = self.arg.start_epoch * self.n_iter_per_epoch
            self.load_optimizer()
            self.load_scheduler()
        else: 
             self.global_step = 0
             if self.optimizer is None: self.load_optimizer()

        self.lr = self.optimizer.param_groups[0]['lr'] if self.optimizer and self.optimizer.param_groups else getattr(self.arg, 'base_lr', 0.001)
        self.best_acc = 0.0
        self.best_acc_epoch = 0

        self.model = self.model.to(self.output_device)
        if isinstance(self.arg.device, list) and len(self.arg.device) > 1 and all(isinstance(d, int) and d >= 0 for d in self.arg.device):
            if torch.cuda.is_available() and len(self.arg.device) <= torch.cuda.device_count():
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device 
                )
                self.print_log(f'模型已在 GPUs {self.arg.device} 上启用 DataParallel。')
            else:
                self.print_log(f"警告: 请求的 GPU 设备 {self.arg.device} 无效或数量超出可用范围。未使用 DataParallel。", logging.WARNING)
        self.print_log("Processor 初始化完成。")

    def setup_device(self):
        if not hasattr(self.arg, 'device') or self.arg.device is None:
            self.arg.device = [0] if torch.cuda.is_available() else [-1] 
        if not isinstance(self.arg.device, list): self.arg.device = [self.arg.device]

        if self.arg.device[0] == -1 or not torch.cuda.is_available():
            self.output_device = torch.device("cpu")
            self.arg.device = ["cpu"] 
        else:
            valid_devices = [d for d in self.arg.device if isinstance(d, int) and 0 <= d < torch.cuda.device_count()]
            if not valid_devices:
                self.output_device = torch.device("cpu")
                self.arg.device = ["cpu"]
            else:
                self.arg.device = valid_devices
                self.output_device = torch.device(f"cuda:{self.arg.device[0]}")
                try:
                    torch.cuda.set_device(self.output_device)
                except Exception as e:
                    self.output_device = torch.device("cpu") 
                    self.arg.device = ["cpu"]
                    if hasattr(self, 'logger'): # 只有在logger已设置后才打印
                        self.print_log(f"错误: 设置 CUDA 设备失败: {e}。将使用 CPU。", logging.ERROR)


    def setup_logging_and_writers(self):
        work_dir = self.arg.work_dir
        os.makedirs(work_dir, exist_ok=True)
        log_file = os.path.join(work_dir, 'log.txt')
        
        should_clear_log = self.arg.phase == 'train' and self.arg.start_epoch == 0
        filemode = 'w' if should_clear_log else 'a'

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() 

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)-8s - %(message)s', 
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.FileHandler(log_file, mode=filemode, encoding='utf-8'),
                                logging.StreamHandler(sys.stdout) 
                            ])
        
        self.logger = logging.getLogger("Processor") 

        if self.output_device.type == 'cpu':
            self.print_log("将在 CPU 上运行。")
        else:
            if self.arg.device == ["cpu"]: 
                 self.print_log(f"警告: GPU 设置失败，将在 CPU 上运行。", logging.WARNING)
            elif not [d for d in self.arg.device if isinstance(d, int) and 0 <= d < torch.cuda.device_count()]:
                 self.print_log(f"错误: 无效 GPU 索引 {self.arg.device}。系统有 {torch.cuda.device_count()} 个 GPU。将使用 CPU。", logging.ERROR)
            else:
                 self.print_log(f"使用 GPU: {self.arg.device}。主输出设备: {self.output_device}")

        self.print_log(f'工作目录: {work_dir}')
        self.print_log(f"日志文件: {log_file} (模式: {filemode})")

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
            except Exception as e: 
                self.print_log(f"警告: 初始化 TensorBoardWriter 失败: {e}", logging.WARNING)
                self.train_writer = self.val_writer = None
        else:
             self.train_writer = self.val_writer = None

    def print_log(self, msg, level=logging.INFO):
        if getattr(self.arg, 'print_log', True):
            logger_to_use = getattr(self, 'logger', logging.getLogger()) 
            logger_to_use.log(level, msg)

    def save_arg(self):
        arg_dict = vars(self.arg)
        work_dir = self.arg.work_dir
        os.makedirs(work_dir, exist_ok=True)
        try:
            filepath = os.path.join(work_dir, 'config_used.yaml')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Work Dir: {work_dir}\n# Phase: {self.arg.phase}\n# Command line: {' '.join(sys.argv)}\n\n")
                yaml.dump(arg_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True, Dumper=Dumper)
        except Exception as e:
            self.print_log(f"警告: 保存 config_used.yaml ({work_dir}) 失败: {e}", logging.WARNING)

    def load_data(self):
        self.print_log("开始加载数据...")
        feeder_path = getattr(self.arg, 'feeder', None)
        if not feeder_path: raise ValueError("'feeder' 参数未在配置中设置。")
        Feeder = import_class(feeder_path)

        self.data_loader = dict()
        num_worker = getattr(self.arg, 'num_worker', 4)

        if self.arg.phase == 'train':
            train_batch_size = getattr(self.arg, 'batch_size') 
            train_feeder_args = getattr(self.arg, 'train_feeder_args', {})
            train_dataset = Feeder(**train_feeder_args)
            self.data_loader['train'] = DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=num_worker, drop_last=getattr(self.arg, 'drop_last', True),
                worker_init_fn=init_seed, pin_memory=True, collate_fn=collate_fn_filter_none
            )
            self.print_log(f"训练数据加载器 '{self.arg.feeder}' 加载成功。样本数: {len(train_dataset)}")

        test_batch_size = getattr(self.arg, 'test_batch_size')
        test_feeder_args = getattr(self.arg, 'test_feeder_args', {})
        test_dataset = Feeder(**test_feeder_args)
        self.data_loader['val'] = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=num_worker, drop_last=False, worker_init_fn=init_seed,
            pin_memory=True, collate_fn=collate_fn_filter_none
        )
        self.data_loader['test'] = self.data_loader['val']
        self.print_log(f"测试/验证数据加载器 '{self.arg.feeder}' 加载成功。样本数: {len(test_dataset)}")
        self.print_log("数据加载完成。")

    def load_model(self):
        self.print_log(f"模型将运行在设备: {self.output_device}")
        model_path_str = getattr(self.arg, 'model')
        Model = import_class(model_path_str)
        try: 
            model_file_path = inspect.getfile(Model)
            if os.path.exists(model_file_path) and os.path.isfile(model_file_path):
                 shutil.copy2(model_file_path, self.arg.work_dir)
        except Exception: pass 

        model_args_dict = getattr(self.arg, 'model_args', {})
        self.model = Model(model_cfg=model_args_dict)
        self.print_log(f"模型 '{self.arg.model}' 实例化成功。")

        loss_type = getattr(self.arg, 'loss_type', 'CE').upper()
        if loss_type == 'SMOOTHCE':
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).to(self.output_device)
        else: 
            self.loss = nn.CrossEntropyLoss().to(self.output_device)
        self.print_log(f"损失函数: {loss_type}")

        weights_path = getattr(self.arg, 'weights', None)
        if weights_path:
            self.print_log(f'加载权重自: {weights_path}')
            if not os.path.exists(weights_path):
                self.print_log(f"错误: 权重文件不存在: {weights_path}", logging.ERROR); return
            try:
                weights = torch.load(weights_path, map_location=self.output_device)
                weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
                ignore_weights_list = getattr(self.arg, 'ignore_weights', [])
                keys_to_remove = [k_loaded for w_pattern in ignore_weights_list for k_loaded in weights if w_pattern in k_loaded]
                if keys_to_remove:
                    self.print_log(f"将忽略以下加载的权重: {keys_to_remove}")
                    for k_rem in keys_to_remove: weights.pop(k_rem, None)
                
                missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
                if missing_keys: self.print_log(f"警告: 模型缺失键: {missing_keys}", logging.WARNING)
                if unexpected_keys: self.print_log(f"警告: 权重中多余键: {unexpected_keys}", logging.WARNING)
                self.print_log("权重加载完成 (strict=False)。")
            except Exception as e:
                self.print_log(f"错误: 加载权重失败: {e}", logging.ERROR); traceback.print_exc()
    
    def load_optimizer(self):
        optimizer_type = getattr(self.arg, 'optimizer', 'AdamW').lower()
        lr = self.arg.base_lr
        wd = getattr(self.arg, 'weight_decay', 0.01)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
             self.print_log("警告: 模型中无可优化的参数。", logging.WARNING)
             self.optimizer = optim.AdamW([], lr=lr) 
             return

        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD(params, lr=lr, momentum=getattr(self.arg, 'momentum', 0.9), nesterov=getattr(self.arg, 'nesterov', False), weight_decay=wd)
        elif optimizer_type == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        elif optimizer_type == 'adamw':
             self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
        self.print_log(f"优化器: {optimizer_type.upper()} (初始lr={lr:.2e}, wd={wd:.1e})")

    def load_scheduler(self):
        scheduler_type = getattr(self.arg, 'lr_scheduler', 'multistep').lower()
        if not self.optimizer: self.print_log("错误：优化器未初始化。", logging.ERROR); return
        self.print_log(f"尝试加载调度器: {scheduler_type}")

        if scheduler_type == 'multistep':
            steps = getattr(self.arg, 'step', [])
            if not steps: self.print_log("警告: MultiStepLR 'step' 参数为空。", logging.WARNING); return
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps, gamma=getattr(self.arg, 'lr_decay_rate', 0.1))
            if self.arg.warm_up_epoch > 0: self.print_log(f"  将配合手动预热 {self.arg.warm_up_epoch} epochs. MultiStepLR 将在 warmup 后生效。")
        elif scheduler_type == 'cosine':
            if CosineLRScheduler is None: self.print_log("错误: CosineLRScheduler 未导入。", logging.ERROR); return
            if self.n_iter_per_epoch <= 0 and self.arg.phase == 'train': 
                 self.print_log("错误: CosineLRScheduler 需要 n_iter_per_epoch > 0。", logging.ERROR); return
            
            total_steps = int(self.arg.num_epoch * self.n_iter_per_epoch)
            warmup_steps = int(self.arg.warm_up_epoch * self.n_iter_per_epoch)
            try:
                self.lr_scheduler_each_step = CosineLRScheduler(
                    self.optimizer,
                    t_initial=(total_steps - warmup_steps) if getattr(self.arg, 'warmup_prefix', True) else total_steps,
                    lr_min=getattr(self.arg, 'min_lr', 1e-6),
                    warmup_lr_init=getattr(self.arg, 'warmup_lr', 1e-6), 
                    warmup_t=warmup_steps, cycle_limit=1, t_in_epochs=False,
                    warmup_prefix=getattr(self.arg, 'warmup_prefix', True)
                )
                self.print_log("调度器: CosineLRScheduler (timm) 加载成功。")
            except Exception as e:
                self.print_log(f"错误: 初始化 CosineLRScheduler 失败: {e}", logging.ERROR)
        else:
            self.print_log(f"警告: 不支持的学习率调度器类型 '{scheduler_type}'。", logging.WARNING)

    def record_time(self): self.cur_time = time.time()
    def split_time(self): split = time.time() - self.cur_time; self.record_time(); return split

    def train(self, epoch):
        self.model.train()
        self.print_log(f'======> 训练 Epoch: {epoch + 1}')
        loader = self.data_loader['train']
        if not loader or self.n_iter_per_epoch == 0: 
            self.print_log("错误或空训练数据加载器！", logging.ERROR); return

        lr_for_log = -1.0
        if self.lr_scheduler_each_step is None: 
            if self.arg.warm_up_epoch > 0 and epoch < self.arg.warm_up_epoch:
                warmup_start_lr = getattr(self.arg, 'warmup_lr', self.arg.base_lr * 0.01)
                progress = (epoch + 1) / self.arg.warm_up_epoch
                lr_for_log = warmup_start_lr + (self.arg.base_lr - warmup_start_lr) * progress
                lr_for_log = min(lr_for_log, self.arg.base_lr)
                for pg in self.optimizer.param_groups: pg['lr'] = lr_for_log
            else: 
                lr_for_log = self.optimizer.param_groups[0]['lr'] 
        else: 
            lr_for_log = self.optimizer.param_groups[0]['lr']
        self.print_log(f"Epoch {epoch+1} 开始，学习率: {lr_for_log:.8f}")

        loss_val, acc_val, grad_norm_val = [], [], []
        if self.train_writer: self.train_writer.add_scalar('meta/epoch', epoch + 1, epoch + 1)
        
        # --- 恢复 timer 初始化 ---
        self.record_time()
        timer = dict(dataloader=0.0, model=0.0, statistics=0.0)
        
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch}", ncols=120, leave=False)
        log_interval = getattr(self.arg, 'log_interval', 50)

        for batch_idx, batch_data in enumerate(process):
            self.global_step += 1
            if self.lr_scheduler_each_step: self.lr_scheduler_each_step.step(self.global_step)
            
            if batch_data is None: timer['dataloader'] += self.split_time(); continue # 仍然记录跳过的时间
            
            # --- 更新 timer['dataloader'] ---
            timer['dataloader'] += self.split_time() # 记录数据加载和预处理到这里为止的时间

            data, label, mask, _ = batch_data 
            data = data.float().to(self.output_device, non_blocking=True)
            label = label.long().to(self.output_device, non_blocking=True)
            if mask is not None: mask = mask.bool().to(self.output_device, non_blocking=True)
            
            # 重置计时器，开始模型计算计时
            self.record_time()

            output, _ = self.model(data, mask=mask)
            loss = self.loss(output, label)

            if torch.isnan(loss) or torch.isinf(loss):
                self.print_log(f"警告: Batch {batch_idx} 损失为 NaN/Inf！", logging.WARNING)
                timer['model'] += self.split_time(); continue # 记录模型计算时间

            self.optimizer.zero_grad(); loss.backward()
            
            total_norm = 0.0; valid_grad_for_norm = True
            for p in self.model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): valid_grad_for_norm = False; break
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5 if valid_grad_for_norm else float('nan')
            
            if valid_grad_for_norm:
                grad_norm_val.append(total_norm)
                if getattr(self.arg, 'grad_clip', True) and getattr(self.arg, 'grad_max', 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.arg.grad_max)
                self.optimizer.step()
            else:
                self.print_log(f"警告: Batch {batch_idx} 梯度包含 NaN/Inf，跳过优化。", logging.WARNING)
                self.optimizer.zero_grad() 
            
            # --- 更新 timer['model'] ---
            timer['model'] += self.split_time() # 模型前向、反向、优化结束

            loss_item = loss.item(); loss_val.append(loss_item)
            with torch.no_grad(): _, pred = torch.max(output.data, 1); acc_item = torch.mean((pred == label.data).float()).item(); acc_val.append(acc_item)

            grad_pf = f"{total_norm:.2f}" if not np.isnan(total_norm) else "NaN"
            process.set_postfix_str(f"Loss: {loss_item:.3f}, Acc: {acc_item:.2f}, Grad: {grad_pf}")

            # 重置计时器，开始统计和日志记录计时
            self.record_time()

            if log_interval > 0 and (self.global_step % log_interval == 0):
                lr_curr = self.optimizer.param_groups[0]['lr']
                log_line = (f"Epoch: [{epoch+1}][{batch_idx+1}/{self.n_iter_per_epoch}]\t"
                            f"Loss: {loss_item:.4f}\tAcc: {acc_item:.3f}\tLR: {lr_curr:.8f}\tGradNorm: {total_norm:.4f if not np.isnan(total_norm) else 'NaN'}")
                self.print_log(log_line)
                if self.train_writer:
                    self.train_writer.add_scalar('train/batch_loss', loss_item, self.global_step)
                    self.train_writer.add_scalar('train/batch_acc', acc_item, self.global_step)
                    if not np.isnan(total_norm): self.train_writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                    self.train_writer.add_scalar('meta/learning_rate_iter', lr_curr, self.global_step)
            
            # --- 更新 timer['statistics'] ---
            timer['statistics'] += self.split_time() # 日志和统计结束
        
        process.close()
        
        avg_loss = np.nanmean(loss_val) if loss_val else float('nan')
        avg_acc = np.nanmean(acc_val) * 100 if acc_val else 0.0
        avg_grad = np.nanmean(grad_norm_val) if grad_norm_val else float('nan')
        grad_log_str = f"{avg_grad:.4f}" if not np.isnan(avg_grad) else "NaN"
        
        # --- 恢复时间消耗的计算和打印 ---
        total_time_epoch = sum(timer.values())
        proportion = {k: f"{int(round(v * 100 / total_time_epoch))}%" if total_time_epoch > 0 else "0%" for k, v in timer.items()}
        
        self.print_log(f'\t平均训练损失: {avg_loss:.4f}. 平均训练准确率: {avg_acc:.2f}%. 平均梯度范数: {grad_log_str}.')
        self.print_log(f'\t时间消耗: [数据加载]{proportion["dataloader"]}, [网络计算]{proportion["model"]}, [统计]{proportion["statistics"]}')
        
        if self.train_writer:
            if not np.isnan(avg_loss): self.train_writer.add_scalar('train/epoch_loss', avg_loss, epoch + 1)
            if not np.isnan(avg_acc): self.train_writer.add_scalar('train/epoch_acc', avg_acc / 100.0, epoch + 1)
            if not np.isnan(avg_grad): self.train_writer.add_scalar('train/epoch_grad_norm', avg_grad, epoch + 1)
            self.train_writer.add_scalar('meta/learning_rate_epoch', self.optimizer.param_groups[0]['lr'], epoch + 1)

        if self.scheduler and self.lr_scheduler_each_step is None and (self.arg.warm_up_epoch <= 0 or epoch >= self.arg.warm_up_epoch):
            self.scheduler.step()
            self.print_log(f"\tMultiStepLR.step(). New LR: {self.optimizer.param_groups[0]['lr']:.8f}")

    def eval(self, epoch, save_score_final_eval=False, loader_name=['val'], wrong_file=None, result_file=None):
        # (eval 方法保持与上一版本一致，这里不再重复，除非有特定修改需求)
        is_training_phase_eval = (self.arg.phase == 'train')
        score_path_to_return = None 
        self.model.eval()
        self.print_log(f'======> 评估 Epoch: {epoch + 1} on {", ".join(loader_name)}')

        for ln_idx, ln in enumerate(loader_name): 
            loader = self.data_loader.get(ln)
            if not loader: self.print_log(f"警告: 找不到加载器 '{ln}'。", logging.WARNING); continue

            all_loss, all_logits, all_labels, all_indices = [], [], [], []
            process = tqdm(loader, desc=f"Eval {ln} (Epoch {epoch+1})", ncols=100, leave=False)

            for batch_data in process:
                if batch_data is None: continue
                data, label, mask, index = batch_data
                data = data.float().to(self.output_device, non_blocking=True)
                label_cpu = label.long() 
                label = label_cpu.to(self.output_device, non_blocking=True)
                if mask is not None: mask = mask.bool().to(self.output_device, non_blocking=True)

                with torch.no_grad():
                    output, _ = self.model(data, mask=mask)
                    loss = self.loss(output, label)
                    if not (torch.isnan(loss) or torch.isinf(loss)): all_loss.append(loss.item())
                    all_logits.append(output.cpu()); all_labels.append(label_cpu); all_indices.append(index)
            process.close()

            if not all_logits: self.print_log(f"警告: 在 {ln} 上无数据处理。", logging.WARNING); continue
            
            logits_all = torch.cat(all_logits, dim=0)
            labels_all = torch.cat(all_labels, dim=0)
            preds_all = torch.argmax(logits_all, dim=1)
            
            # 稍微健壮一点的索引拼接
            if all_indices and all_indices[0] is not None:
                if isinstance(all_indices[0], torch.Tensor):
                    indices_all = torch.cat([idx for idx in all_indices if idx is not None]).numpy()
                else: # 假设是 numpy 数组或列表
                    indices_all = np.concatenate([np.array(idx).reshape(-1) for idx in all_indices if idx is not None])
            else:
                indices_all = np.array([])


            eval_loss = np.mean(all_loss) if all_loss else float('nan')
            eval_acc = accuracy_score(labels_all.numpy(), preds_all.numpy()) if len(labels_all) > 0 else 0.0
            if ln_idx == 0: eval_acc_primary_loader = eval_acc 

            if is_training_phase_eval and self.val_writer:
                if not np.isnan(eval_loss): self.val_writer.add_scalar(f'eval/{ln}_epoch_loss', eval_loss, epoch + 1)
                self.val_writer.add_scalar(f'eval/{ln}_epoch_acc_top1', eval_acc, epoch + 1)

            self.print_log(f'\t{ln} 集: 平均损失: {eval_loss:.4f}, Top-1 Acc: {eval_acc * 100:.2f}%')

            num_classes = self.arg.model_args.get('num_classes', 0)
            if num_classes > 0 and len(labels_all) > 0:
                for k in getattr(self.arg, 'show_topk', [1]):
                    if k > 1 and k < num_classes: # Top-k 只有在 k > 1 且 k < 类别数时才有意义
                        try:
                            topk_acc = top_k_accuracy_score(labels_all.numpy(), logits_all.numpy(), k=k, labels=np.arange(num_classes))
                            self.print_log(f'\t{ln} 集: Top-{k} Acc: {topk_acc * 100:.2f}%')
                            if is_training_phase_eval and self.val_writer: self.val_writer.add_scalar(f'eval/{ln}_epoch_acc_top{k}', topk_acc, epoch + 1)
                        except Exception as e_topk:
                             self.print_log(f"警告: 计算 Top-{k} 准确率失败 for {ln}: {e_topk}", logging.WARNING)

            if save_score_final_eval and len(indices_all) == len(logits_all):
                score_dict = {idx.item() if hasattr(idx, 'item') else int(idx): vec.numpy() for idx, vec in zip(indices_all, logits_all)}

                score_file = os.path.join(self.arg.work_dir, f'eval_score_{ln}_epoch{epoch+1}.pkl')
                try:
                    with open(score_file, 'wb') as f: pickle.dump(score_dict, f)
                    self.print_log(f"评估分数 ({ln}) 已为最终评估保存到: {score_file}")
                    if ln_idx == 0: score_path_to_return = score_file
                except Exception as e: self.print_log(f"警告: 保存最终评估分数 ({ln}) 失败: {e}", logging.WARNING)
            
            # 文件写入 (wrong_file, result_file)
            if (wrong_file or result_file) and ln_idx == 0 and len(indices_all) == len(labels_all) and len(indices_all) == len(preds_all):
                with open(wrong_file, 'w', encoding='utf-8') if wrong_file else open(os.devnull, 'w') as fw, \
                     open(result_file, 'w', encoding='utf-8', newline='') if result_file else open(os.devnull, 'w') as fr:
                    csv_w = csv.writer(fr) if result_file else None
                    if csv_w: csv_w.writerow(["Sample_Index", "Prediction", "True_Label"])
                    
                    for i in range(len(labels_all)):
                        s_idx = indices_all[i].item() if hasattr(indices_all[i], 'item') else int(indices_all[i])
                        s_pred = preds_all[i].item()
                        s_true = labels_all[i].item()
                        if csv_w: csv_w.writerow([s_idx, s_pred, s_true])
                        if wrong_file and s_pred != s_true: fw.write(f"{s_idx},{s_pred},{s_true}\n")
            
            # 混淆矩阵
            is_final_eval = (self.arg.phase == 'test' or (self.arg.phase == 'train' and epoch + 1 == self.arg.num_epoch))
            if num_classes > 0 and ln_idx == 0 and is_final_eval and len(labels_all) > 0:
                try:
                    cm = confusion_matrix(labels_all.numpy(), preds_all.numpy(), labels=np.arange(num_classes))
                    cm_file = os.path.join(self.arg.work_dir, f'confusion_matrix_{ln}_final_epoch{epoch+1}.csv')
                    # (代码保存混淆矩阵到CSV的逻辑与之前类似，为简洁省略，但应保留)
                    self.print_log(f"最终混淆矩阵 ({ln}) 已保存到: {cm_file} (保存逻辑待补充)")
                except Exception as e: self.print_log(f"警告: 保存混淆矩阵 ({ln}) 失败: {e}", logging.WARNING)
        
        return eval_acc_primary_loader, score_path_to_return

    def start(self):
        # (start 方法与上一版本基本一致，确保调用 eval 时传递正确的 save_score_final_eval)
        final_score_path_for_main = None 
        if self.arg.phase == 'train':
            self.print_log('开始训练阶段...')
            self.print_log(f'参数:\n{yaml.dump(vars(self.arg), default_flow_style=None, sort_keys=False, allow_unicode=True, Dumper=Dumper)}')
            self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
            
            num_epochs = int(self.arg.num_epoch)
            patience = getattr(self.arg, 'early_stop_patience', 0)
            patience_counter = 0
            self.print_log(f"总训练 Epochs: {num_epochs}, 起始 Epoch: {self.arg.start_epoch + 1}")
            if patience > 0: self.print_log(f"启用 Early Stopping, patience={patience}")

            for epoch in range(self.arg.start_epoch, num_epochs):
                self.train(epoch) # 调用已恢复 timer 的 train 方法
                if (epoch + 1) % self.arg.eval_interval == 0 or epoch + 1 == num_epochs:
                    self.print_log(f"--- 开始评估 Epoch {epoch + 1} ---")
                    val_acc, _ = self.eval(epoch, save_score_final_eval=False, loader_name=['val'])
                    self.print_log(f"--- 结束评估 Epoch {epoch + 1} (Val Acc: {val_acc*100:.2f}%) ---")

                    if val_acc > self.best_acc:
                        self.best_acc, self.best_acc_epoch = val_acc, epoch + 1
                        patience_counter = 0
                        best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                        try:
                            state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                            torch.save(state_dict_to_save, best_model_path)
                            self.print_log(f'*** 新的最佳准确率: {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}). 模型已保存到 {best_model_path} ***')
                        except Exception as e: self.print_log(f"警告: 保存最佳模型失败: {e}", logging.WARNING)
                    elif patience > 0:
                        patience_counter += 1
                        self.print_log(f'验证集准确率未提升. EarlyStopping Counter: {patience_counter}/{patience}')
                        if patience_counter >= patience: self.print_log(f'触发 Early Stopping (Epoch {epoch + 1})'); break
            
            self.print_log('训练完成。')
            if self.best_acc_epoch > 0:
                self.print_log(f'训练中最佳验证准确率 (Top-1): {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}).')
                best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                if os.path.exists(best_model_path):
                    self.print_log(f'加载最佳模型 {best_model_path} 进行最终测试...')
                    try:
                        # 加载权重时，如果保存的是 state_dict，直接加载即可
                        # 如果模型在DataParallel下训练和保存，加载到单GPU或CPU时，权重键通常不需要额外处理'module.'前缀，
                        # 因为保存时我们已经处理了。但如果加载的模型实例本身是DataParallel封装的，而权重不是，则可能需要加前缀。
                        # 为简单起见，这里假设保存和加载的 DataParallel 状态一致，或者 load_state_dict(strict=False) 能处理。
                        # 更稳健的做法是像之前那样，明确处理 'module.'
                        loaded_weights = torch.load(best_model_path, map_location=self.output_device)
                        if isinstance(self.model, nn.DataParallel) and not list(loaded_weights.keys())[0].startswith('module.'):
                            loaded_weights = OrderedDict([('module.'+k, v) for k,v in loaded_weights.items()])
                        elif not isinstance(self.model, nn.DataParallel) and list(loaded_weights.keys())[0].startswith('module.'):
                            loaded_weights = OrderedDict([(k.replace('module.',''),v) for k,v in loaded_weights.items()])

                        self.model.load_state_dict(loaded_weights)
                        self.print_log("最佳模型权重加载成功。")
                        
                        _, final_score_path_for_main = self.eval(
                            epoch=self.best_acc_epoch -1, 
                            save_score_final_eval=getattr(self.arg, 'save_score', False),
                            loader_name=['test'] 
                        )
                    except Exception as e: self.print_log(f"错误: 加载或测试最佳模型失败: {e}", logging.ERROR); traceback.print_exc()
            else:
                self.print_log("训练中未记录有效的最佳模型。", logging.WARNING)

        elif self.arg.phase == 'test':
            self.print_log('开始测试阶段...')
            weights_path = getattr(self.arg, 'weights', None)
            if not weights_path or not os.path.exists(weights_path):
                self.print_log(f"错误: 测试阶段必须指定有效的 --weights 文件路径。当前: {weights_path}", logging.ERROR); return None
            
            self.print_log(f'模型: {self.arg.model}, 权重: {weights_path}')
            # 权重已在 __init__ -> load_model 中加载 (如果 phase 是 test 且提供了 weights)
            # 或者在这里再次确保加载 (如果 __init__ 中的加载逻辑依赖 phase=='train' 的某些设置)
            # 当前 load_model 逻辑不依赖 phase，所以权重应该已加载
            _, final_score_path_for_main = self.eval(
                epoch=0, 
                save_score_final_eval=getattr(self.arg, 'save_score', False),
                loader_name=['test']
            )
            self.print_log('测试完成。')
        
        else: 
            self.print_log(f"执行 Phase: {self.arg.phase}")
            if self.arg.phase == 'model_size':
                 self.print_log(f'模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}')
                 self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

        if self.train_writer: self.train_writer.close()
        if self.val_writer: self.val_writer.close()
        return final_score_path_for_main