# 文件名: evaluation.py (重写版)
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
logger = logging.getLogger(__name__)

# --- 导入你的新模型和数据加载器 ---
from model.SDT_GRUs_Gesture import SDT_GRU_Classifier
# from loader.gesture_datasets import YourGestureDatasetLoader # 导入你的加载器
from utils import StandardScaler, move2device, get_logger # 保留需要的 utils

def evaluate_gesture_model(model_cfg_path, dataset_cfg_path, weight_path, log_dir, device='cuda'):
    logger = get_logger(log_dir, name='evaluation') # 创建 logger
    logger.info(f"Evaluating model specified in {model_cfg_path} with weights {weight_path}")
    logger.info(f"Using dataset specified in {dataset_cfg_path}")

    # --- 加载配置 ---
    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)['model'] # 获取模型配置部分
    with open(dataset_cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) # 获取完整配置 (包含 dataset)

    # --- 加载数据 ---
    # 实例化测试数据集和加载器
    # 注意：需要从 cfg 获取数据集名称和参数
    # test_set = YourGestureDatasetLoader(cfg['dataset'], split='test')
    # test_loader = DataLoader(test_set, batch_size=cfg['train']['batch_size'], ...)
    # scaler = StandardScaler(mean=test_set.calculate_mean(), std=test_set.calculate_std()) # 或虚拟 scaler
    # ！！！你需要替换下面的占位符！！！
    print("错误：请在此处替换为实际的数据加载代码！")
    test_loader = [] # 占位符
    class DummyScaler: # 占位符
        def transform(self, data): return data
    scaler = DummyScaler()

    # --- 加载模型 ---
    model = SDT_GRU_Classifier(model_cfg)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    dl = tqdm(test_loader, desc="Testing")
    for batch in dl:
        # --- 数据解包、预处理、移动 ---
        x, labels, *mask = batch
        mask = mask[0] if mask else None
        x = scaler.transform(x)
        x, labels = move2device([x, labels], device)
        if mask is not None:
            mask = move2device(mask, device)

        with torch.no_grad():
            logits, _ = model(x, mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # --- 计算指标 ---
    if not all_logits:
         logger.error("No data processed during evaluation. Check dataloader.")
         return

    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_preds_tensor = torch.argmax(all_logits_tensor, dim=1)

    accuracy = accuracy_score(all_labels_tensor.numpy(), all_preds_tensor.numpy())
    report = classification_report(all_labels_tensor.numpy(), all_preds_tensor.numpy(), zero_division=0)
    cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy())

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:\n" + report)
    logger.info("Confusion Matrix:\n" + np.array2string(cm))

    # 可以将结果保存到文件
    results_df = pd.DataFrame({'true': all_labels_tensor.numpy(), 'pred': all_preds_tensor.numpy()})
    results_df.to_csv(os.path.join(log_dir, 'test_predictions.csv'), index=False)
    with open(os.path.join(log_dir, 'test_report.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

if __name__ == "__main__":
    # --- 设置评估所需的参数 ---
    log_dir = './log/Gesture_Eval' # 指定评估日志目录
    model_yaml_file = './config/gesture_model_config.yaml' # 指向你的模型配置文件
    dataset_yaml_file = './config/gesture_dataset_config.yaml' # 指向你的数据集配置文件
    weight_path = './log/Gesture_Train/best.pt' # 指向训练好的模型权重
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    evaluate_gesture_model(model_yaml_file, dataset_yaml_file, weight_path, log_dir, device)