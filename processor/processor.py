# 文件名: processor/processor.py (修改版 - 7.27)
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
import traceback
import random
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, top_k_accuracy_score
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
from utils import init_seed, import_class, LabelSmoothingCrossEntropy 
from timm.scheduler.cosine_lr import CosineLRScheduler

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.best_acc = 0.0
        self.best_acc_epoch = 0
        self.global_step = 0

        self._setup_device_and_logging() # 包含保存配置
        self._save_config()
        self._load_and_prepare_data()
        self._load_and_prepare_model() # loss 在这里初始化

        if self.arg.phase == 'train':
            self.n_iter_per_epoch = len(self.data_loader['train']) if 'train' in self.data_loader and self.data_loader['train'] else 0
            if self.n_iter_per_epoch == 0: self.print_log("警告: 训练数据加载器为空或长度为0。", logging.WARNING)
            self.global_step = 0  # 简化：总是从0开始
            self._load_optimizer() # 简化后的优化器加载
        else: # 测试或评估阶段，也需要优化器实例（即使不进行step）
            self._load_optimizer()

        self.lr = self.arg.base_lr # 初始化lr，将在每个epoch更新

    def _setup_device_and_logging(self):
        # (设备设置部分保持不变)
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

        # 简化判断条件，同时保留您需要的 input() 确认机制
        if self.arg.phase == 'train' and os.path.isfile(os.path.join(work_dir, 'log.txt')):
            print(f"检测到全新训练，但旧工作目录 '{work_dir}' 已存在。")
            while True:
                user_choice = input("是否删除旧目录并重新开始？(y/N): ").strip().lower()
                if user_choice in ['y', 'yes']:
                    try:
                        shutil.rmtree(work_dir)
                        print(f"旧目录已成功删除。")
                        break
                    except OSError as e:
                        print(f"警告: 删除旧目录失败: {e}。将在原目录中继续写入。")
                        break
                elif user_choice in ['n', 'no', '']:
                    print("将在原目录中继续训练，日志文件将追加写入。")
                    break
                else:
                    print("请输入 y 或 n")
        
        # 确保目录存在 (如果被删除，则重新创建)
        os.makedirs(work_dir, exist_ok=True)
        
        log_file_path = os.path.join(work_dir, 'log.txt')
        # 根据文件是否存在决定日志模式 ('w' 或 'a')
        filemode = 'w' if not os.path.exists(log_file_path) else 'a'
        
        # 重置并配置日志记录器
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
        
        # 打印设备信息 (删除了冗余的工作目录和日志文件路径打印)
        if self.output_device.type == 'cpu': self.print_log("将在 CPU 上运行。")
        else: self.print_log(f"使用 GPU: {self.arg.device_actual}。主输出设备: {self.output_device}")

        # TensorBoard 设置 (保持不变)
        self.train_writer = self.val_writer = None
        if self.arg.phase == 'train':
            runs_dir = os.path.join(work_dir, 'runs')
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
        try:
            # 只保存YAML中实际设置的参数，避免保存大量默认参数
            config_to_save = {
                'work_dir': self.arg.work_dir,
                'phase': self.arg.phase,
                'device': getattr(self.arg, 'device', None),
                'command_line': ' '.join(sys.argv)
            }
            
            # 只保存YAML中实际设置的参数
            if hasattr(self.arg, '_yaml_set_params') and self.arg._yaml_set_params:
                for param in self.arg._yaml_set_params:
                    if hasattr(self.arg, param):
                        config_to_save[param] = getattr(self.arg, param)
            
            filepath = os.path.join(self.arg.work_dir, 'config_used.yaml')
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入文件头信息
                f.write(f"# Work Dir: {self.arg.work_dir}\n")
                f.write(f"# Phase: {self.arg.phase}\n")
                f.write(f"# Device: {self.output_device}\n")
                f.write(f"# Command line: {' '.join(sys.argv)}\n")
                f.write(f"# 本文件只包含YAML配置文件中实际设置的参数\n\n")
                
                # 保存简化的配置
                yaml.dump(config_to_save, f, default_flow_style=False, 
                        sort_keys=True, Dumper=Dumper, allow_unicode=True)
            
            self.print_log(f"简化配置已保存到: {filepath}")
            
        except Exception as e:
            self.print_log(f"警告: 保存 config_used.yaml 失败: {e}", logging.WARNING)
            traceback.print_exc()

    def _create_dataloader(self, feeder_args, batch_size, shuffle, is_train=False):
        from torch.utils.data import DataLoader
        if not self.arg.feeder: raise ValueError("'feeder' 参数未设置。")
        Feeder = import_class(self.arg.feeder)
        feeder_constructor_args = feeder_args.copy()
        feeder_constructor_args.pop('max_len', None)      
        
        # 使用清理过的参数来实例化数据集
        dataset = Feeder(**feeder_constructor_args)
        
        num_worker = getattr(self.arg, 'num_worker', 0)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_worker,
            drop_last=getattr(self.arg, 'drop_last', True) if is_train else False,
            worker_init_fn=init_seed,
            pin_memory=True
        )
        log_modality_info = feeder_args.get('data_path', '未知数据路径')
        self.print_log(f"{'训练' if is_train else '验证/测试'}数据加载器 '{os.path.basename(self.arg.feeder)}' "
                       f"(数据路径: {log_modality_info}) 加载成功。样本数: {len(dataset)}")
        return loader
        
    def _load_and_prepare_data(self):
            self.print_log("开始加载和准备数据...")
            self.data_loader = {}
            try:
                # 1. 创建训练加载器 (仅在训练阶段需要)
                if self.arg.phase == 'train':
                    self.data_loader['train'] = self._create_dataloader(
                        self.arg.train_feeder_args, self.arg.batch_size, shuffle=True, is_train=True
                    )
                
                # 2. 创建一个标准的“验证”加载器，用于训练过程中的模型选择
                # 我们拿一份 test_feeder_args 的拷贝，并在存在相应键时强制将其设置为 'val'
                val_args = self.arg.test_feeder_args.copy()
                for _k in ('split', 'subset', 'phase', 'mode'):
                    if _k in val_args:
                        original_v = val_args[_k]
                        val_args[_k] = 'val'
                        self.print_log(f"验证集构造: 将 '{_k}' 从 '{original_v}' 强制为 'val' 以避免用测试集做模型选择。")
                self.data_loader['val'] = self._create_dataloader(
                    val_args, self.arg.test_batch_size, shuffle=False
                )

                # 3. 如果是测试阶段，额外创建一个标准的“测试”加载器
                if self.arg.phase == 'test':
                    test_args = self.arg.test_feeder_args.copy()
                    self.data_loader['test'] = self._create_dataloader(
                        test_args, self.arg.test_batch_size, shuffle=False
                    )

            except Exception as e:
                self.print_log(f"错误: 加载数据时发生严重错误: {e}", logging.CRITICAL); traceback.print_exc(); raise
            self.print_log("数据加载和准备完成。")


    def _load_and_prepare_model(self):
        from torch.nn.parallel import DataParallel
        self.print_log("开始加载和准备模型...")

        Model = import_class(self.arg.model)
        
        # 直接复制模型定义文件
        model_source_file = inspect.getfile(Model)
        shutil.copy2(model_source_file, self.arg.work_dir)

        self.model = Model(model_cfg=self.arg.model_args)
        
        loss_type = getattr(self.arg, 'loss_type', 'CE').upper()
        if loss_type == 'LSCE':
            smoothing_val = getattr(self.arg, 'label_smoothing', 0.1)
            self.loss = LabelSmoothingCrossEntropy(smoothing=smoothing_val).to(self.output_device)
            self.print_log(f"损失函数: LabelSmoothingCrossEntropy (smoothing={smoothing_val})")
        else:
            self.loss = nn.CrossEntropyLoss().to(self.output_device)
            self.print_log(f"损失函数: CrossEntropyLoss")

        if self.arg.weights:
            self.print_log(f'加载权重自: {self.arg.weights}')
            weights = torch.load(self.arg.weights, map_location=self.output_device)
            weights = OrderedDict([[k.replace('module.', ''), v] for k, v in weights.items()])
            
            if self.arg.ignore_weights:
                for p in self.arg.ignore_weights:
                    weights = {k: v for k, v in weights.items() if p not in k}
                self.print_log(f"已忽略包含关键字 {self.arg.ignore_weights} 的权重。")

            missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
            if missing_keys: self.print_log(f"警告: 模型中缺失的键: {missing_keys}", logging.WARNING)
            if unexpected_keys: self.print_log(f"警告: 权重文件中多余的键: {unexpected_keys}", logging.WARNING)
            self.print_log("权重加载完成。")
        
        self.model.to(self.output_device)
        if isinstance(self.arg.device_actual, list) and len(self.arg.device_actual) > 1 and self.output_device.type == 'cuda':
            self.model = DataParallel(self.model, device_ids=self.arg.device_actual, output_device=self.output_device)
            self.print_log(f'模型已在 GPUs {self.arg.device_actual} 上启用 DataParallel。')

        self.print_log("模型加载和准备完成。")

    def _load_optimizer(self):
        """加载优化器 (仅支持 SGD 和 AdamW)。"""
        optimizer_type = self.arg.optimizer.lower() # 假设 optimizer 参数总是在 arg 中
        lr = self.arg.base_lr
        wd = self.arg.weight_decay
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

    def adjust_learning_rate(self, epoch): 
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' :
            if epoch < self.arg.warm_up_epoch: 
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch 
            else: 
                lr = self.arg.base_lr * ( 
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step))) 
            for param_group in self.optimizer.param_groups: 
                param_group['lr'] = lr 
            self.lr = lr
            return lr 
        else: 
            raise ValueError()

    def record_time(self): self.cur_time = time.time()
    def split_time(self): split = time.time() - self.cur_time; self.record_time(); return split

    def train(self, epoch):
        self.model.train()
        current_lr = self.adjust_learning_rate(epoch)
        
        # --- 日志与进度条设置 ---
        log_prefix = f'Epoch {epoch + 1}/{self.arg.num_epoch}'
        if epoch < getattr(self.arg, 'warm_up_epoch', 0):
            log_prefix += " (Warmup)"
        self.print_log(f'======> {log_prefix}，当前学习率: {current_lr:.8f}')
        
        loader = self.data_loader['train']
        if not loader:
            self.print_log("错误: 训练数据加载器为空！", logging.ERROR)
            return

        # --- 用于累积指标的变量 ---
        total_loss = 0.0
        total_correct = 0
        total_samples = 0 

        process = tqdm(loader, desc=log_prefix, ncols=120, leave=False)

        # --- 标准的全精度训练循环 ---
        for batch_data in process:
            
            # 1. 数据准备
            data, label, index = batch_data
            data = data.float().to(self.output_device, non_blocking=True)
            label = label.long().to(self.output_device, non_blocking=True)
            
            # 2. 前向传播
            model_output = self.model(data)
            if isinstance(model_output, tuple):
                output = model_output[0]  # 只取logits
            else:
                output = model_output
            loss = self.loss(output, label)

            if torch.isnan(loss) or torch.isinf(loss):
                self.print_log(f"警告: 损失变为 NaN/Inf，跳过此批次。", logging.WARNING)
                continue

            # 3. 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()

            # 4. 权重更新（已移除梯度裁剪）

            self.optimizer.step()
            
            # 6. 累积指标
            with torch.no_grad():
                total_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, 1)
                total_correct += (pred == label).sum().item()
                total_samples += data.size(0)
            
            # 7. 更新tqdm进度条
            if total_samples > 0:
                process.set_postfix_str(
                    f"Loss: {loss.item():.3f}, "
                    f"Acc: {(pred == label).float().mean().item():.3f}"
                )

        process.close()
        
        # --- Epoch结束后的总结与日志 ---
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_acc = total_correct / total_samples * 100.0 if total_samples > 0 else 0.0
        
        self.print_log(f'\t平均训练损失: {avg_loss:.4f}. 平均训练准确率: {avg_acc:.2f}%.')

        if self.train_writer:
            self.train_writer.add_scalar('Epoch训练/平均损失', avg_loss, epoch + 1)
            self.train_writer.add_scalar('Epoch训练/平均准确率', avg_acc / 100.0, epoch + 1)
            self.train_writer.add_scalar('学习率/Epoch', current_lr, epoch + 1)
    def eval(self, epoch, save_score_final_eval=False, loader_name=['val'], wrong_file=None, result_file=None):      
            self.model.eval()
            self.print_log(f'======> 评估 Epoch: {epoch + 1} 于数据集: {", ".join(loader_name)}')
            final_eval_acc = 0.0
            final_score_path_for_return = None

            for ln_idx, ln in enumerate(loader_name): 
                loader = self.data_loader.get(ln)
                if not loader:
                    self.print_log(f"警告: 找不到名为 '{ln}' 的数据加载器。", logging.WARNING)
                    continue
                
                all_loss, all_logits, all_labels, all_indices = [], [], [], []
                
                process = tqdm(loader, desc=f"评估 {ln} (Epoch {epoch+1})", ncols=120, leave=False)

                for batch_data in process:
                    
                    data, label_cpu, index = batch_data
                    data = data.float().to(self.output_device, non_blocking=True)
                    label_gpu = label_cpu.long().to(self.output_device, non_blocking=True)
                    
                    with torch.no_grad():
                        try:
                            model_output = self.model(data)
                            output = model_output[0] if isinstance(model_output, tuple) else model_output
                            loss = self.loss(output, label_gpu)
                            if not (torch.isnan(loss) or torch.isinf(loss)): all_loss.append(loss.item())
                            all_logits.append(output.cpu())
                            all_labels.append(label_cpu)   
                            all_indices.append(index.cpu())
                        except Exception as e: 
                            self.print_log(f"错误: 主模型在评估数据集 '{ln}'上前向传播失败: {e}", logging.ERROR); continue
                process.close()

                if not all_logits:
                    self.print_log(f"警告: 在数据集 '{ln}' 上没有处理任何有效的模型输出 logits。", logging.WARNING); continue
                
                # --- 主模型指标计算 (保持不变) ---
                logits_all_np = torch.cat(all_logits, dim=0).numpy()
                labels_all_np = torch.cat(all_labels, dim=0).numpy()
                preds_all_np = np.argmax(logits_all_np, axis=1)
                indices_all_np = torch.cat(all_indices, dim=0).numpy() if all_indices and all_indices[0] is not None else np.array([])
                eval_loss = np.mean(all_loss) if all_loss else 0.0
                eval_acc = accuracy_score(labels_all_np, preds_all_np) if len(labels_all_np) > 0 else 0.0
                
                if ln_idx == 0: final_eval_acc = eval_acc
                self.print_log(f'\t数据集 [{ln}]: 平均损失: {eval_loss:.4f}, Top-1 准确率: {eval_acc * 100:.2f}%')

                # --- Tensorboard 日志 (清理EMA部分) ---
                if self.arg.phase == 'train' and self.val_writer:
                    if not np.isnan(eval_loss): self.val_writer.add_scalar(f'评估/{ln}_epoch_loss', eval_loss, epoch + 1)
                    self.val_writer.add_scalar(f'评估/{ln}_epoch_acc_top1', eval_acc, epoch + 1)

                # --- 文件保存逻辑 (清理EMA部分) ---
                if save_score_final_eval and ln_idx == 0:
                    num_class = self.arg.model_args.get('num_class', 0)
                    if num_class > 0 and len(labels_all_np) > 0:
                        # --- 1. 保存主模型的结果 (保持您原来的逻辑) ---
                        try:
                            report_title = (f"最终测试评估报告 (基于权重: {os.path.basename(self.arg.weights)})" if self.arg.phase == 'test'
                                        else f"最终评估报告 (基于Epoch {self.best_acc_epoch} 的最佳模型)")
                            report_filename = 'final_test_report.txt' if self.arg.phase == 'test' else 'final_evaluation_report_best_model.txt'
                            report_path = os.path.join(self.arg.work_dir, report_filename)
                            
                            report_content = [f"{report_title}\n", f"最终评估准确率 (Top-1): {eval_acc * 100:.2f}%\n", "分类报告:"]
                            class_names = [f'C{i}' for i in range(num_class)]
                            report_content.append(classification_report(labels_all_np, preds_all_np, target_names=class_names, labels=np.arange(num_class), zero_division=0))
                            report_content.append("\n混淆矩阵:")
                            report_content.append(np.array2string(confusion_matrix(labels_all_np, preds_all_np, labels=np.arange(num_class)), separator=', '))
                            
                            with open(report_path, 'w', encoding='utf-8') as f: f.write('\n'.join(report_content))
                            self.print_log(f"完整的最终评估报告已保存到: {report_path}")
                        except Exception as e_report: self.print_log(f"警告: 生成或保存最终评估报告失败: {e_report}", logging.WARNING)

                        score_file_name = 'epoch1_test_score.pkl'
                        current_score_path = os.path.join(self.arg.work_dir, score_file_name)
                        score_data = {'indices': indices_all_np, 'scores': logits_all_np, 'labels': labels_all_np}
                        try:
                            with open(current_score_path, 'wb') as f: pickle.dump(score_data, f)
                            self.print_log(f"评估分数已保存到: {current_score_path}")
                            if ln_idx == 0: final_score_path_for_return = current_score_path
                        except Exception as e: self.print_log(f"警告: 保存评估分数失败: {e}", logging.WARNING)

                if (wrong_file or result_file) and ln_idx == 0:
                    self._save_prediction_details(indices_all_np, preds_all_np, labels_all_np, wrong_file, result_file)

            self.print_log(f"--- 结束评估 Epoch {epoch + 1} (评估集准确率: {final_eval_acc*100:.2f}%) ---")
            
            return final_eval_acc, final_score_path_for_return
    
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
                self.print_log(f"总训练 Epochs: {num_epochs}")

                # --- 训练主循环 (这部分逻辑是正确的，保持不变) ---
                for epoch in range(num_epochs):
                    self.train(epoch)
                    
                    perform_eval = (epoch + 1) % self.arg.eval_interval == 0 or (epoch + 1) == num_epochs
                    if perform_eval:
                        val_acc, _ = self.eval(
                            epoch,
                            save_score_final_eval=False,
                            loader_name=['val']
                        )
                        
                        if val_acc > self.best_acc:
                            self.best_acc, self.best_acc_epoch = val_acc, epoch + 1
                            best_model_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                            state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
                            torch.save(state_dict_to_save, best_model_path)
                            self.print_log(f'*** 新的最佳准确率: {self.best_acc*100:.2f}% (Epoch {self.best_acc_epoch}). 模型已保存到 {best_model_path} ***')
                
                self.print_log('训练完成。')

                # --- 训练结束后，加载最佳模型进行最终评估 (全新简化逻辑) ---
                self.print_log(f'\n--- 最终评估 ---')
                final_loader = ['test'] if 'test' in self.data_loader else ['val']
                self.print_log(f"将在 {final_loader[0]} 数据集上进行最终评估。")

                # --- 直接加载最佳模型权重并进行最终评估 ---
                best_epoch = self.best_acc_epoch
                best_acc = self.best_acc
                best_weights_path = os.path.join(self.arg.work_dir, 'best_model.pt')
                self.print_log(f"最佳验证准确率: {best_acc*100:.2f}% (Epoch {best_epoch})")

                if os.path.exists(best_weights_path):
                    self.print_log(f'加载最佳模型权重: {best_weights_path} ...')
                    loaded_weights = torch.load(best_weights_path, map_location='cpu')
                    model_to_load = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                    model_to_load.load_state_dict(loaded_weights)

                    # --- 执行一次评估，生成最终报告和分数文件 ---
                    _, final_score_path_for_main = self.eval(
                        epoch=best_epoch - 1, # 传递正确的epoch号给报告
                        save_score_final_eval=True, 
                        loader_name=final_loader
                    )
                else:
                    self.print_log(f"警告: 找不到最佳模型的权重文件: {best_weights_path}", logging.WARNING)

            elif self.arg.phase == 'test':
                self.print_log('开始测试阶段...')
                if not self.arg.weights:
                    raise ValueError("错误: 测试阶段必须通过 --weights 指定权重文件。")
                self.print_log(f'测试权重: {self.arg.weights}')
                
                loaded_weights = torch.load(self.arg.weights, map_location='cpu')
                model_to_load = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                
                is_dp_weights = list(loaded_weights.keys())[0].startswith('module.')
                if is_dp_weights:
                    loaded_weights = OrderedDict((k[7:], v) for k, v in loaded_weights.items())
                model_to_load.load_state_dict(loaded_weights)
                
                _, final_score_path_for_main = self.eval(
                    epoch=0,
                    save_score_final_eval=True,
                    loader_name=['test']
                )
                self.print_log(f'测试完成。')

            elif self.arg.phase == 'model_size':
                self.print_log(f'模型总参数量: {sum(p.numel() for p in self.model.parameters()):,}')
                self.print_log(f'模型可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
            
            # --- 清理工作 ---
            if self.train_writer: self.train_writer.close()
            if self.val_writer: self.val_writer.close()
                
            return final_score_path_for_main
