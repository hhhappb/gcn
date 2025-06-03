import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import torch
import sys 
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late fusion for skeleton action recognition") # 添加描述
    parser.add_argument('--dataset', required=True, 
                        choices={'NW-UCLA', 'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset'}, # 根据你的实际数据集调整
                        help='Dataset name for determining default alphas and label loading (if applicable)')
    parser.add_argument('--joint_dir', type=str, help='Directory for joint scores .pkl file')
    parser.add_argument('--bone_dir', type=str, help='Directory for bone scores .pkl file')
    parser.add_argument('--joint_motion_dir', type=str, default=None, help='Directory for joint_motion scores .pkl file')
    parser.add_argument('--bone_motion_dir', type=str, default=None, help='Directory for bone_motion scores .pkl file')
    parser.add_argument('--score_file_name', default='epoch1_test_score.pkl', 
                        help='Name of the .pkl file containing scores in each directory (e.g., epoch1_test_score.pkl or test_scores.pkl)')
    parser.add_argument('--alphas', type=float, nargs='+', default=None, 
                        help='List of fusion weights corresponding to the order: joint, bone, joint_motion, bone_motion. Only provide for active streams.')
    parser.add_argument('--label_file', type=str, default=None,
                        help='Path to the ground truth label file (e.g., val_label.pkl for NW-UCLA if not in score files)')

    arg = parser.parse_args()

    # --- 确定激活的流和对应的目录 ---
    stream_configs = [
        ('joint', arg.joint_dir),
        ('bone', arg.bone_dir),
        ('joint_motion', arg.joint_motion_dir),
        ('bone_motion', arg.bone_motion_dir)
    ]
    
    active_streams_info = [(name, path) for name, path in stream_configs if path is not None]
    
    if not active_streams_info:
        print("错误：没有提供任何模态流的目录。请至少指定一个 --*_dir 参数。")
        sys.exit(1)

    active_stream_names = [info[0] for info in active_streams_info]
    active_stream_paths = [info[1] for info in active_streams_info]

    # --- 确定融合权重 ---
    alphas = arg.alphas
    if alphas is None:
        print("信息：未通过 --alphas 提供融合权重，将根据数据集和激活流数量使用预设或平均值。")
        if 'UCLA' in arg.dataset:
            # 示例权重，你需要根据实际情况调整或通过实验获得
            if len(active_stream_names) == 4:
                alphas = [0.4, 0.4, 0.3, 0.2] # J, B, JM, BM
                # alphas = [0.3, 0.3, 0.25, 0.25]
            elif len(active_stream_names) == 3: # J, B, JM
                if all(s in active_stream_names for s in ['joint', 'bone', 'joint_motion']): alphas = [0.5, 0.3, 0.2]
                else: alphas = [1.0 / len(active_stream_names)] * len(active_stream_names)
            elif len(active_stream_names) == 2: # J, B
                if all(s in active_stream_names for s in ['joint', 'bone']): alphas = [0.6, 0.4]
                else: alphas = [1.0 / len(active_stream_names)] * len(active_stream_names)
            elif len(active_stream_names) == 1: alphas = [1.0]
            else: alphas = [1.0 / len(active_stream_names)] * len(active_stream_names) if active_stream_names else []
        # TODO: 为其他数据集 (ntu/xsub 等) 添加类似的预设权重逻辑
        else: # 默认平均分配
            alphas = [1.0 / len(active_stream_names)] * len(active_stream_names) if active_stream_names else []

    print(f"数据集: {arg.dataset}")
    print(f"待加载的分数文件名: {arg.score_file_name}")
    print(f"将要融合的模态流: {active_stream_names}")
    print(f"使用的融合权重 (alphas): {alphas}")

    # --- 加载分数和标签 ---
    all_scores_tensors = []
    label_np_array = None
    num_samples_from_first_file = None

    for i, stream_name in enumerate(active_stream_names):
        stream_dir = active_stream_paths[i]
        score_path = os.path.join(stream_dir, arg.score_file_name)
        print(f"正在加载分数: {score_path} (流: '{stream_name}')")
        try: 
            with open(score_path, 'rb') as f:
                data_dict = pickle.load(f) 
            
            if 'scores' not in data_dict or 'labels' not in data_dict:
                print(f"错误: 分数文件 {score_path} 缺少 'scores' 或 'labels' 键。跳过此流。")
                print("请确保所有指定流的分数文件格式正确。")
                sys.exit(1)

            current_stream_scores = torch.from_numpy(data_dict['scores']).float()
            current_stream_labels = data_dict['labels']

            if num_samples_from_first_file is None:
                num_samples_from_first_file = current_stream_scores.shape[0]
            elif current_stream_scores.shape[0] != num_samples_from_first_file:
                print(f"错误: 流 '{stream_name}' 的样本数 ({current_stream_scores.shape[0]}) 与第一个流 ({num_samples_from_first_file}) 不匹配。")
                sys.exit(1)

            all_scores_tensors.append(current_stream_scores)

            if label_np_array is None: # 从第一个成功加载的文件中获取标签
                label_np_array = current_stream_labels
                print(f"标签已从 {score_path} 加载。样本数: {len(label_np_array)}")
            elif not np.array_equal(current_stream_labels, label_np_array):
                print(f"严重错误: 文件 {score_path} 中的标签与之前加载的标签不匹配！请确保所有分数文件对应相同的样本和顺序。")
                sys.exit(1)

        except FileNotFoundError:
            print(f"错误: 分数文件未找到: {score_path}。请检查路径和文件名。")
            sys.exit(1)
           
    label = torch.from_numpy(label_np_array).long()

    # 确保加载的分数数量与alpha权重数量一致 (在循环外再次检查，因为可能有流加载失败)
    if len(all_scores_tensors) != len(alphas):
        print(f"错误: 成功加载的分数集数量 ({len(all_scores_tensors)}) 与有效 alpha 权重数量 ({len(alphas)}) 不匹配。")
        sys.exit(1)

    # --- 融合分数 ---
    fused_score = torch.zeros_like(all_scores_tensors[0]) # 以第一个加载的分数形状为基准
    for i, stream_scores_tensor in enumerate(all_scores_tensors):
        if stream_scores_tensor.shape[0] != fused_score.shape[0]:
            print(f"错误: 第 {i} 个流的分数张量样本数 ({stream_scores_tensor.shape[0]}) 与预期 ({fused_score.shape[0]}) 不符。")
            sys.exit(1)
        fused_score += stream_scores_tensor * alphas[i]
    
    # --- 计算准确率 ---
    right_num = 0
    total_num = 0
    right_num_5 = 0

    print("开始计算融合后的准确率...")
    for i in tqdm(range(len(label)), desc="Calculating Accuracy"):
        l = label[i].item() 
        s = fused_score[i]  
        
        _, topk_preds = torch.topk(s, k=5) 
        if l in topk_preds.tolist(): # 转换为 list 进行判断
            right_num_5 += 1
        
        pred_label = torch.argmax(s).item()
        if pred_label == l:
            right_num += 1
        total_num += 1
        
    acc = right_num / total_num if total_num > 0 else 0
    acc5 = right_num_5 / total_num if total_num > 0 else 0

    print(f"\n--- 融合结果 ({arg.dataset}) ---")
    print(f"融合的流: {active_stream_names}")
    print(f"使用的权重: {alphas}")
    print('Top1 Acc: {:.2f}%'.format(acc * 100))
    print('Top5 Acc: {:.2f}%'.format(acc5 * 100))