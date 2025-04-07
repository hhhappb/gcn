# 文件名: train.py (修改版)

# --- 导入新模型和可能的 gesture dataloader ---
# from loader.HZMetro import HZMetro # 移除
# from loader.SHMetro import SHMetro # 移除
# from loader.BJMetro import BJMetro # 移除
# 导入你为手势数据集编写的加载器，例如:
# from loader.gesture_datasets import NTUDataset, DHGDataset # (你需要创建这些)

from model.SDT_GRUs_Gesture import SDT_GRU_Classifier # <--- 修改导入
from trainer import metrics # 需要修改或替换 metrics
from utils import StandardScaler, move2device, StepLR2

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import os
import time
from tqdm import tqdm
import yaml
# --- 导入分类指标计算库 ---
# 例如: from torchmetrics.classification import Accuracy
from sklearn.metrics import accuracy_score # 或者使用 sklearn

# ... (Loader, Dumper 不变) ...

def gen_data(cfg, loader_type):
    # --- 修改数据加载逻辑 ---
    dataset_name = cfg.get('dataset_name', 'ntu') # 从配置获取数据集名称
    dataset_args = cfg['dataset'] # 获取数据集相关配置

    print(f"Loading dataset: {dataset_name}, split: {loader_type}")

    if dataset_name == 'ntu':
        # data_set = NTUDataset(dataset_args, split=loader_type) # 示例
        raise NotImplementedError("NTU Dataloader not implemented yet!")
    elif dataset_name == 'dhg':
        # data_set = DHGDataset(dataset_args, split=loader_type) # 示例
        raise NotImplementedError("DHG Dataloader not implemented yet!")
    elif dataset_name == 'shrec':
        # data_set = SHRECDataset(dataset_args, split=loader_type) # 示例
        raise NotImplementedError("SHREC Dataloader not implemented yet!")
    elif dataset_name == 'nw_ucla':
        # data_set = NWUCLADataset(dataset_args, split=loader_type) # 示例
        raise NotImplementedError("NW-UCLA Dataloader not implemented yet!")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if loader_type == 'train':
        shuffle = True
        drop_last = True # 或者 False，取决于你的策略
    else: # 'val' or 'test'
        shuffle = False
        drop_last = False

    data_loader = DataLoader(data_set,
                             batch_size=cfg['train']['batch_size'], # 使用 train 下的 batch_size
                             shuffle=shuffle,
                             num_workers=dataset_args.get('num_workers', 4), # 从 dataset 配置获取
                             pin_memory=True,
                             drop_last=drop_last)
    return data_set, data_loader


def train_one_epoch(model, data_loader, criterion, optimizer, scaler, max_grad_norm, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    dl = tqdm(data_loader, desc="Training")

    for idx, batch in enumerate(dl):
        optimizer.zero_grad()

        # --- 修改数据解包和移动 ---
        # 假设 batch 包含: (骨骼数据, 标签, [可选 mask])
        x, labels, *mask = batch # 根据你的 dataloader 输出调整
        mask = mask[0] if mask else None

        # --- 数据预处理 (归一化) 和移动 ---
        # 注意：scaler 现在只用于 x
        x = scaler.transform(x) # 假设 scaler 适用于 (B, T, N, C)
        x, labels = move2device([x, labels], device)
        if mask is not None:
            mask = move2device(mask, device)

        # --- 修改模型调用 ---
        # y_pred 在分类任务中是 logits
        logits, _ = model(x, mask) # y 不再是输入

        # --- 计算损失 ---
        # criterion 是 nn.CrossEntropyLoss
        loss = criterion(logits, labels) # labels 应该是 LongTensor 类型
        loss.backward()

        # --- 梯度裁剪和优化器步骤 ---
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # --- 记录损失和准确率 ---
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        dl.set_postfix_str('Loss: {:.4f} | Acc: {:.3f}'.format(loss.item(), (preds == labels).float().mean().item()))

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate_model(eval_type, model, data_loader, scaler, device, logger):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss() # 需要损失函数来计算评估损失 (可选)

    dl = tqdm(data_loader, desc=f"Evaluating {eval_type}")
    for idx, batch in enumerate(dl):
        # --- 数据解包、预处理、移动 (同 train_one_epoch) ---
        x, labels, *mask = batch
        mask = mask[0] if mask else None
        x = scaler.transform(x)
        x, labels = move2device([x, labels], device)
        if mask is not None:
            mask = move2device(mask, device)

        with torch.no_grad():
            # --- 模型调用 ---
            logits, _ = model(x, mask)
            loss = criterion(logits, labels) # 计算验证/测试损失
            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # --- 组合所有批次的结果 ---
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_preds_tensor = torch.argmax(all_logits_tensor, dim=1)

    # --- 计算评估指标 ---
    avg_loss = total_loss / len(data_loader)
    # 使用 sklearn 计算准确率
    accuracy = accuracy_score(all_labels_tensor.numpy(), all_preds_tensor.numpy())
    # 你可以在这里添加 F1-score, Precision, Recall 等其他指标

    if logger: # 避免在没 logger 时报错
        logger.info(f'Evaluation_{eval_type}_Results: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 返回平均损失和准确率 (以及其他你需要的指标)
    # 注意：原始返回格式是针对多步预测的 MAE/RMSE/MAPE 列表，这里改为标量指标
    return avg_loss, accuracy


def train_model(cfg, logger, log_dir, seed):
    if seed is not None:
        # ... (设置随机种子不变) ...

    device = cfg['device']

    # --- 修改数据加载 ---
    train_set, train_loader = gen_data(cfg, 'train')
    val_set, val_loader = gen_data(cfg, 'val')
    test_set, test_loader = gen_data(cfg, 'test')

    # --- Scaler ---
    # 假设 train_set 提供了计算 mean/std 的方法或属性
    # 需要确认你的 Gesture Dataset 类如何提供 mean/std
    try:
        # scaler = StandardScaler(mean=train_set.mean, std=train_set.std)
        # 或者如果你的数据集直接提供归一化数据，就不需要 scaler
        # 暂时假设需要 scaler
        scaler = StandardScaler(mean=train_set.calculate_mean(), std=train_set.calculate_std()) # 假设有这些方法
        print("Scaler created.")
    except AttributeError:
        print("Warning: Dataset does not provide mean/std calculation. Scaler not used or needs manual setup.")
        # 创建一个虚拟 scaler，不做任何操作
        class DummyScaler:
            def transform(self, data): return data
            def inverse_transform(self, data): return data
        scaler = DummyScaler()


    # --- 实例化新模型 ---
    model = SDT_GRU_Classifier(cfg['model']) # <--- 使用新模型
    model.to(device)

    # ... (打印模型大小、加载预训练参数逻辑不变) ...

    # --- 修改损失函数 ---
    criterion = nn.CrossEntropyLoss() # <--- 使用交叉熵

    # --- 优化器和学习率调度器 (保持不变，但参数可能需要调整) ---
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['base_lr'], ...)
    scheduler = StepLR2(optimizer=optimizer, milestones=cfg['train']['steps'], ...)

    max_grad_norm = cfg['train']['max_grad_norm']
    best_val_metric = 0.0 # 对于准确率，越高越好
    best_epoch = -1

    for epoch in range(cfg['train']['epoch']):
        begin_time = time.perf_counter()

        # --- 调用修改后的训练函数 ---
        train_loss, train_acc = train_one_epoch(model=model,
                                                data_loader=train_loader,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                scaler=scaler, # 传入 scaler
                                                max_grad_norm=max_grad_norm,
                                                device=device)

        # --- 调用修改后的评估函数 ---
        val_loss, val_acc = evaluate_model(eval_type='val',
                                           model=model,
                                           data_loader=val_loader,
                                           scaler=scaler, # 传入 scaler
                                           device=device,
                                           logger=logger)

        logger.info(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # --- 修改保存最佳模型的逻辑 ---
        # 保存验证集准确率最高的模型
        current_val_metric = val_acc
        if current_val_metric > best_val_metric:
            logger.info(f'Validation accuracy improved from {best_val_metric:.4f} to {current_val_metric:.4f}')
            best_val_metric = current_val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pt'))
            logger.info(f"Best model saved at epoch {epoch}")

        # --- 模型周期性保存 (可选，逻辑不变) ---
        if (epoch + 1) % cfg['train']['save_every_n_epochs'] == 0:
            # ... (保存 epoch checkpoint 不变) ...

        scheduler.step()

        time_elapsed = time.perf_counter() - begin_time
        logger.info(f'Time elapsed for epoch {epoch}: {time_elapsed:.2f}s')


    # --- 训练结束后加载最佳模型并测试 ---
    logger.info(f'Loading best model from epoch {best_epoch} with validation accuracy {best_val_metric:.4f}')
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best.pt')))

    # --- 调用修改后的评估函数进行最终测试 ---
    test_loss, test_acc = evaluate_model(eval_type="test",
                                         model=model,
                                         data_loader=test_loader,
                                         scaler=scaler, # 传入 scaler
                                         device=device,
                                         logger=logger)
    logger.info(f'Final Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')

    # 返回测试结果 (可以根据需要调整返回格式)
    return {"test_loss": test_loss, "test_accuracy": test_acc}