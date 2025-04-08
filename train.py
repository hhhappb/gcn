# 文件名: train.py (骨骼手势识别专用版)
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import time
import traceback # 用于打印详细错误信息
from collections import OrderedDict

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# --- 导入分类指标库 ---
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# --- 导入你的模型和可能的工具函数 ---
# 假设模型类在 model 目录下
# from model.SDT_GRUs_Gesture_Runnable_CN import SDT_GRU_Classifier # 确保路径正确
# 假设工具函数在 utils 目录下
# from utils import move2device, import_class # 等

# --- 为确保独立性，如果需要 import_class，在此处定义 ---
def import_class(import_str):
    """动态导入 Python 类"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    if not mod_str: # 如果是顶级模块
        mod_str = class_str
        class_str = None
    try:
        __import__(mod_str)
        if class_str is None:
            return sys.modules[mod_str]
        else:
            return getattr(sys.modules[mod_str], class_str)
    except ImportError as e:
         raise ImportError(f"无法导入模块 {mod_str}: {e}")
    except AttributeError:
        raise ImportError('在模块 %s 中无法找到类 %s (%s)' % (mod_str, class_str, traceback.format_exception(*sys.exc_info())))
    except Exception as e:
         raise ImportError(f"导入 {import_str} 时发生错误: {e}")

# --- 同样，如果需要 move2device，在此处定义或从 utils 导入 ---
def move2device(obj, device):
    """将 Tensor 或包含 Tensor 的列表/字典移动到指定设备"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, list):
        return [move2device(item, device) for item in obj]
    elif isinstance(obj, tuple):
         # 注意：tuple 是不可变的，需要创建新的 tuple
         return tuple(move2device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: move2device(v, device) for k, v in obj.items()}
    else:
        # 对于非 Tensor 类型或不支持的容器，直接返回
        return obj


# --- 数据加载函数 (适配骨骼 Feeder) ---
def gen_data(cfg, split):
    """
    根据配置动态加载数据加载器。
    Args:
        cfg (dict): 包含所有配置的字典。
        split (str): 'train', 'val', 或 'test'。
    Returns:
        tuple: (Dataset 对象, DataLoader 对象)
    """
    feeder_path = cfg.get('feeder')
    if not feeder_path:
        raise ValueError("配置文件 (cfg) 中缺少 'feeder' 参数，无法确定数据加载器类。")

    # 根据 split 获取对应的 feeder 参数
    if split == 'train':
        feeder_args = cfg.get('train_feeder_args', {})
        shuffle = True
        drop_last = cfg['train'].get('drop_last', True) # 从 train 配置获取 drop_last
        batch_size = cfg['train'].get('batch_size', 32) # 从 train 配置获取 batch_size
    elif split == 'val':
        feeder_args = cfg.get('test_feeder_args', {}) # 验证集通常使用测试集参数
        shuffle = False
        drop_last = False
        batch_size = cfg.get('test_batch_size', 64) # 从全局配置获取 test batch_size
    elif split == 'test':
        feeder_args = cfg.get('test_feeder_args', {})
        shuffle = False
        drop_last = False
        batch_size = cfg.get('test_batch_size', 64)
    else:
         raise ValueError(f"不支持的 split 类型: {split}")

    # --- 适配 feeder_ucla.py: 需要根据 split 调整 label_path ---
    # 检查 feeder_path 是否指向 feeder_ucla (或其他需要此逻辑的 feeder)
    # 注意: 这个检查可能需要根据你的实际 feeder 路径调整
    is_feeder_ucla_or_similar = 'feeder_ucla' in feeder_path.lower()

    if is_feeder_ucla_or_similar and 'label_path' not in feeder_args:
        print(f"警告: Feeder '{feeder_path}' 可能需要 'label_path' 参数，但未在 {split}_feeder_args 中找到。将使用默认标识 '{split}'。")
        feeder_args['label_path'] = split # 使用 'train' 或 'val'/'test' 作为标识

    dataset_args = cfg.get('dataset', {}) # 获取通用的数据集参数
    num_workers = dataset_args.get('num_workers', 4)

    print(f"加载数据加载器: {feeder_path}, split: {split}")
    print(f"Feeder 参数: {feeder_args}")

    try:
        Feeder = import_class(feeder_path)
        data_set = Feeder(**feeder_args)
    except ImportError as e:
         print(f"错误: 无法导入 Feeder 类 '{feeder_path}'. 检查路径和文件名。 {e}")
         raise e
    except Exception as e:
         print(f"错误: 实例化 Feeder 类 '{feeder_path}' 失败，参数: {feeder_args}。错误: {e}")
         traceback.print_exc()
         raise e

    # --- 创建 DataLoader ---
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=True, # 通常可以加速 GPU 训练
                             drop_last=drop_last,
                             # worker_init_fn=init_seed # 如果需要保证 worker 随机性一致
                             )
    print(f"{split} 数据加载器创建成功，样本数: {len(data_set)}，批次数: {len(data_loader)}")
    return data_set, data_loader


# --- 训练一个 Epoch 的函数 (适配骨骼数据和分类) ---
def train_one_epoch(model, data_loader, criterion, optimizer, max_grad_norm, device, current_epoch, logger=None):
    """
    执行单个训练 epoch。
    Args:
        model (nn.Module): 要训练的模型。
        data_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数 (例如 nn.CrossEntropyLoss)。
        optimizer (Optimizer): 优化器 (例如 Adam)。
        max_grad_norm (float): 最大梯度范数，用于梯度裁剪 (0 表示不裁剪)。
        device (torch.device): 运行设备 (cuda 或 cpu)。
        current_epoch (int): 当前的 epoch 编号 (用于日志)。
        logger (logging.Logger, optional): 用于记录日志的对象。
    Returns:
        tuple: (平均损失, 平均准确率)
    """
    model.train() # 设置模型为训练模式
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # 使用 desc 显示更丰富的进度条信息
    dl = tqdm(data_loader, desc=f"Epoch {current_epoch+1} Training", leave=False, ncols=100)

    for idx, batch in enumerate(dl):
        optimizer.zero_grad() # 每个 batch 开始前清空梯度

        # --- 数据解包、形状调整、移动 (适配 feeder_ucla) ---
        try:
            # 假设 feeder 返回 (data, label, index) 或 (data, label)
            if len(batch) == 3:
                x, labels, index = batch
            elif len(batch) == 2:
                x, labels = batch
            else:
                raise ValueError(f"数据加载器返回了未知格式的 batch (包含 {len(batch)} 个元素)")

            # x 初始形状: (B, C, T, V, M=1) -> (B, T, N=V, C)
            x = x.squeeze(-1) # 移除最后的维度 M=1 -> (B, C, T, V)
            x = x.permute(0, 2, 3, 1).contiguous() # 调整维度 -> (B, T, V, C)
            mask = None # feeder_ucla 不返回 mask

        except Exception as e:
            if logger: logger.error(f"处理 Batch {idx} 数据时出错: {e}")
            continue # 跳过这个错误的 batch

        # --- 移动到设备 ---
        try:
            x = x.float().to(device) # 确保是 float 类型
            labels = labels.long().to(device) # 标签通常是 Long 类型
        except Exception as e:
            if logger: logger.error(f"Batch {idx} 移动到设备 {device} 时出错: {e}")
            continue

        # --- 模型前向传播 ---
        try:
            # 假设模型接收 (B, T, N, C) 和可选 mask
            logits, _ = model(x, mask=mask) # mask is None here
        except Exception as e:
            if logger: logger.error(f"Batch {idx} 模型前向传播失败: {e}")
            if logger: logger.error(f"输入 x 形状: {x.shape}")
            continue

        # --- 计算损失 ---
        try:
            loss = criterion(logits, labels)
        except Exception as e:
            if logger: logger.error(f"Batch {idx} 计算损失失败: {e}")
            if logger: logger.error(f"Logits 形状: {logits.shape}, Labels 形状: {labels.shape}")
            continue

        # --- 反向传播与优化 ---
        loss.backward()
        if max_grad_norm > 0:
             clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # --- 记录统计信息 ---
        loss_item = loss.item()
        total_loss += loss_item
        with torch.no_grad(): # 准确率计算不需要梯度
             preds = torch.argmax(logits, dim=1)
             correct_batch = (preds == labels).sum().item()
             total_correct += correct_batch
             total_samples += labels.size(0)
             batch_acc = correct_batch / labels.size(0) if labels.size(0) > 0 else 0.0

        # 更新 tqdm 进度条后缀
        dl.set_postfix_str(f'Loss: {loss_item:.4f}, Acc: {batch_acc:.3f}')

    # --- 计算 Epoch 平均指标 ---
    avg_loss = total_loss / len(dl) if len(dl) > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


# --- 评估函数 (适配骨骼数据和分类) ---
def evaluate_model(eval_type, model, data_loader, device, logger=None):
    """
    评估模型在指定数据集上的表现。
    Args:
        eval_type (str): 'val' 或 'test'，用于日志记录。
        model (nn.Module): 要评估的模型。
        data_loader (DataLoader): 验证或测试数据加载器。
        device (torch.device): 运行设备。
        logger (logging.Logger, optional): 用于记录日志的对象。
    Returns:
        tuple: (平均损失, Top-1 准确率)
    """
    model.eval() # 设置模型为评估模式
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss() # 用于计算评估损失

    dl = tqdm(data_loader, desc=f"Evaluating {eval_type}", leave=False, ncols=100)
    for idx, batch in enumerate(dl):
        # --- 数据解包、形状调整、移动 ---
        try:
            if len(batch) == 3:
                x, labels, index = batch
            elif len(batch) == 2:
                x, labels = batch
            else:
                 raise ValueError("评估数据加载器返回了未知格式。")

            x = x.squeeze(-1).permute(0, 2, 3, 1).contiguous()
            mask = None
        except Exception as e:
            if logger: logger.error(f"处理评估 Batch {idx} 数据时出错: {e}")
            continue

        try:
            x = x.float().to(device)
            labels = labels.long().to(device)
        except Exception as e:
            if logger: logger.error(f"移动评估 Batch {idx} 到设备 {device} 时出错: {e}")
            continue

        # --- 模型前向传播 (无梯度) ---
        with torch.no_grad():
            try:
                logits, _ = model(x, mask=mask)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # 收集结果 (移动到 CPU 以节省 GPU 内存)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
            except Exception as e:
                 if logger: logger.error(f"评估 Batch {idx} 模型前向/损失计算失败: {e}")
                 if logger: logger.error(f"输入 x 形状: {x.shape}")
                 continue

    # --- 计算整体指标 ---
    avg_loss = total_loss / len(dl) if len(dl) > 0 else 0
    if not all_logits:
        if logger: logger.warning(f"评估 {eval_type} 时没有处理任何数据！")
        return avg_loss, 0.0

    try:
        # 拼接所有批次的结果
        logits_all = torch.cat(all_logits, dim=0).numpy() # 转 NumPy (N, C)
        labels_all = torch.cat(all_labels, dim=0).numpy() # (N,)
        preds_all = np.argmax(logits_all, axis=1)         # (N,)

        # 计算 Top-1 准确率
        accuracy = accuracy_score(labels_all, preds_all)

        # --- (可选) 计算 Top-K 准确率 ---
        # topk_accuracies = {}
        # num_classes = logits_all.shape[1]
        # class_labels = np.arange(num_classes)
        # for k in [1, 5]: # 或从配置读取 show_topk
        #     if k <= num_classes:
        #         try:
        #             topk_acc = top_k_accuracy_score(labels_all, logits_all, k=k, labels=class_labels)
        #             topk_accuracies[f'top{k}'] = topk_acc
        #         except Exception as e:
        #              if logger: logger.warning(f"计算 Top-{k} 准确率失败: {e}")

    except Exception as e:
        if logger: logger.error(f"组合评估结果或计算指标时出错: {e}")
        accuracy = 0.0 # 或返回 NaN

    if logger:
        log_msg = f'Evaluation_{eval_type}_Results: Loss: {avg_loss:.4f}, Accuracy@1: {accuracy:.4f}'
        # if topk_accuracies:
        #      log_msg += ", ".join([f", Acc@{k}: {v:.4f}" for k, v in topk_accuracies.items() if k != 'top1'])
        logger.info(log_msg)

    # 返回损失和 Top-1 准确率
    return avg_loss, accuracy


# --- 主训练函数 (适配骨骼分类) ---
def train_model(cfg, logger, log_dir, seed=None):
    """
    执行完整的模型训练流程。
    Args:
        cfg (dict): 包含所有配置的字典。
        logger (logging.Logger): 用于记录日志的对象。
        log_dir (str): 保存日志和模型的工作目录。
        seed (int, optional): 随机种子。
    Returns:
        dict or None: 包含最终测试结果的字典，如果训练成功。
    """
    if seed is not None:
        print(f"设置随机种子为: {seed}")
        # init_seed(seed) # 假设在 main.py 设置

    # --- 设备选择 ---
    try:
        # 优先使用 YAML/命令行指定的 device，然后自动检测
        if 'device' in cfg and cfg['device'] is not None:
            # 如果是列表，取第一个作为主设备
            dev_id = cfg['device'][0] if isinstance(cfg['device'], list) else cfg['device']
            device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
    except Exception as e:
        logger.error(f"设置设备时出错: {e}")
        return None

    # --- 加载数据 ---
    try:
        # 使用修改后的 gen_data
        train_set, train_loader = gen_data(cfg, 'train')
        val_set, val_loader = gen_data(cfg, 'val')     # 加载验证集
        test_set, test_loader = gen_data(cfg, 'test')   # 加载测试集
    except Exception as e:
        logger.error(f"创建数据加载器失败: {e}")
        return None

    # --- 移除 Scaler ---
    scaler = None
    logger.info("StandardScaler 已移除，假设归一化在 Dataset 内完成。")

    # --- 实例化模型 ---
    try:
        model_path = cfg.get('model') # 从 cfg 获取模型导入路径
        if not model_path: raise ValueError("配置文件中缺少 'model' 参数。")
        ModelClass = import_class(model_path)

        model_args = cfg.get('model_args', {}) # 获取模型参数
        if not model_args: raise ValueError("配置文件中缺少 'model_args' 参数。")
        logger.info(f"模型参数: {model_args}")

        model = ModelClass(model_args) # 确保 model_args 是字典
        model.to(device)
        logger.info("模型实例化成功。")
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型可训练参数量: {param_count:,}")
    except Exception as e:
        logger.error(f"实例化模型 '{model_path}' 失败: {e}")
        traceback.print_exc()
        return None

    # --- 加载预训练权重 (如果指定) ---
    load_weights_path = cfg['train'].get('load_param')
    if load_weights_path and load_weights_path.lower() != 'none':
        if os.path.exists(load_weights_path):
            logger.info(f"加载预训练权重从: {load_weights_path}")
            try:
                state_dict = torch.load(load_weights_path, map_location=device)
                # 处理 'module.' 前缀
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                # 使用 strict=False 允许加载部分权重或忽略不匹配的键
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                if missing_keys: logger.warning(f"加载权重时发现模型中缺失的键: {missing_keys}")
                if unexpected_keys: logger.warning(f"加载权重时发现权重文件中多余的键: {unexpected_keys}")
                logger.info("权重部分或完全加载成功 (strict=False)。")
            except Exception as e:
                logger.error(f"加载权重文件 '{load_weights_path}' 失败: {e}")
        else:
            logger.warning(f"指定的预训练权重文件不存在: {load_weights_path}")

    # --- 定义损失函数 ---
    criterion = nn.CrossEntropyLoss().to(device)

    # --- 定义优化器 ---
    optimizer_type = cfg['train'].get('optimizer', 'Adam').lower()
    lr = cfg['train'].get('base_lr', 0.001)
    weight_decay = cfg['train'].get('weight_decay', 0.0001)
    logger.info(f"使用优化器: {optimizer_type.upper()}, 初始学习率: {lr:.6f}, 权重衰减: {weight_decay:.6f}")

    optimizer_params = model.parameters() # 默认优化所有参数

    if optimizer_type == 'adam':
        optimizer = optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay, eps=cfg['train'].get('epsilon', 1e-8))
    elif optimizer_type == 'adamw':
         optimizer = optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay, eps=cfg['train'].get('epsilon', 1e-8))
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(optimizer_params, lr=lr, momentum=cfg['train'].get('momentum', 0.9), weight_decay=weight_decay, nesterov=cfg['train'].get('nesterov', False))
    else:
        logger.error(f"不支持的优化器类型: {optimizer_type}")
        return None

    # --- 定义学习率调度器 ---
    scheduler_type = cfg['train'].get('scheduler', 'multisteplr').lower() # 默认为 MultiStepLR
    logger.info(f"使用学习率调度器: {scheduler_type.upper()}")

    if scheduler_type == 'multisteplr':
         scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg['train']['steps'],
                                                   gamma=cfg['train']['lr_decay_ratio'])
    elif scheduler_type == 'reducelronplateau':
         # ReduceLROnPlateau 需要监控一个指标，通常是验证损失或准确率
         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode='max', # 监控准确率，越高越好
                                                          factor=cfg['train']['lr_decay_ratio'],
                                                          patience=cfg['train'].get('lr_patience', 10), # 等待多少个 epoch 没有提升再降 LR
                                                          verbose=True)
    else:
        logger.warning(f"不支持的学习率调度器类型: {scheduler_type}。将不使用调度器。")
        scheduler = None # 不使用调度器

    # --- 训练参数 ---
    max_grad_norm = cfg['train'].get('max_grad_norm', 0) # 0 表示不裁剪
    start_epoch = cfg['train'].get('start_epoch', 0)
    num_epochs = cfg['train']['epoch']
    warmup_epochs = cfg['train'].get('warm_up_epoch', 0)
    base_lr = cfg['train']['base_lr']
    save_interval = cfg['train'].get('save_every_n_epochs', 0) # 0 表示不周期保存
    save_start_epoch = cfg.get('save_epoch', 0) # 从哪个 epoch 开始保存

    best_val_metric = 0.0 # 记录最佳验证准确率
    best_epoch = -1

    logger.info(f"开始训练，共 {num_epochs} 个 Epochs...")
    # --- 主训练循环 ---
    for epoch in range(start_epoch, num_epochs):
        begin_time = time.perf_counter()

        # --- 手动 Warmup ---
        if epoch < warmup_epochs:
             # 线性增加学习率
             warmup_lr = base_lr * (epoch + 1) / warmup_epochs
             for param_group in optimizer.param_groups:
                  param_group['lr'] = warmup_lr
             current_lr = warmup_lr
             logger.info(f"Warmup Epoch {epoch+1}/{warmup_epochs}: 设置 LR 为 {current_lr:.6f}")
        else:
             current_lr = optimizer.param_groups[0]['lr'] # 获取当前调度器设置的 LR

        # --- 训练 ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm, device, epoch, logger)

        # --- 评估 ---
        val_loss, val_acc = evaluate_model('val', model, val_loader, device, logger)

        logger.info(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')

        # --- 更新学习率调度器 ---
        if scheduler:
            if scheduler_type == 'reducelronplateau':
                scheduler.step(val_acc) # ReduceLROnPlateau 需要传入监控的指标
            elif epoch >= warmup_epochs: # 其他调度器在 warmup 后 step
                scheduler.step()

        # --- 保存最佳模型 (基于验证准确率) ---
        current_val_metric = val_acc
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch + 1
            best_model_path = os.path.join(log_dir, 'best_model.pt')
            logger.info(f'*** 新的最佳准确率: {best_val_metric*100:.2f}% (Epoch: {best_epoch}). 保存模型到 {best_model_path} ***')
            try:
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, best_model_path)
            except Exception as e:
                logger.error(f"保存最佳模型失败: {e}")

        # --- 周期性保存模型 ---
        if save_interval > 0 and (epoch + 1) >= save_start_epoch and (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f'epoch-{epoch+1}.pt')
            logger.info(f"达到保存间隔，保存 checkpoint 到: {checkpoint_path}")
            try:
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, checkpoint_path)
            except Exception as e:
                logger.error(f"保存 checkpoint 失败: {e}")

        time_elapsed = time.perf_counter() - begin_time
        logger.info(f'Epoch {epoch+1} 耗时: {time_elapsed:.2f}s')

    # --- 训练结束 ---
    logger.info('训练完成。')
    final_results = None
    if best_epoch != -1:
        logger.info(f'最佳验证准确率: {best_val_metric:.4f} 在 Epoch {best_epoch}')
        logger.info('加载最佳模型进行最终测试...')
        best_model_path = os.path.join(log_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            try:
                # 加载最佳权重
                state_dict = torch.load(best_model_path, map_location=device)
                # 移除可能的 'module.' 前缀
                new_state_dict = OrderedDict()
                for k, v in state_dict.items(): name = k[7:] if k.startswith('module.') else k; new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                logger.info("最佳模型加载成功。")

                # 在测试集上评估
                test_loss, test_acc = evaluate_model("test", model, test_loader, device, logger)
                logger.info(f'最终测试结果: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
                final_results = {"test_loss": test_loss, "test_accuracy": test_acc}
            except Exception as e:
                logger.error(f"加载或测试最佳模型时出错: {e}")
        else:
            logger.warning(f"找不到最佳模型文件: {best_model_path}，无法进行最终测试。")
    else:
        logger.warning("训练过程中没有找到最佳模型（验证准确率未提升）。可能需要调整超参数或增加训练轮数。")

    return final_results