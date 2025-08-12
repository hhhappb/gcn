# ensemble.py (修改版 - 优先外部标签，保留权重搜索，专注Top-1)
import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm # 用于显示进度条
import torch
import sys
import traceback
import itertools # 用于网格搜索生成权重组合

# --- 计算 Top-1 准确率的函数 (保持不变) ---
def calculate_top1_accuracy(scores_tensor, labels_tensor):
    right_num = 0
    total_num = 0
    if len(labels_tensor) != scores_tensor.shape[0]:
        print(f"错误: 标签数量 ({len(labels_tensor)}) 与融合后的分数数量 ({scores_tensor.shape[0]}) 不匹配。")
        return 0
    for i in range(len(labels_tensor)):
        l = labels_tensor[i].item()
        s = scores_tensor[i]
        pred_label = torch.argmax(s).item()
        if pred_label == l:
            right_num += 1
        total_num += 1
    acc = right_num / total_num if total_num > 0 else 0
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于预测分数的后期融合，用于骨骼动作识别，并搜索最佳Top-1权重 (权重和为1)")
    parser.add_argument('--dataset', required=True, 
                        choices={'shrec17', 'nw-ucla', 'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset'},
                        help='数据集名称，用于可能的默认行为或日志记录')
    parser.add_argument('--joint_dir', type=str, default=None, help='包含 "joint" 分数 .pkl 文件的目录')
    parser.add_argument('--bone_dir', type=str, default=None, help='包含 "bone" 分数 .pkl 文件的目录')
    parser.add_argument('--joint_motion_dir', type=str, default=None, help='包含 "joint_motion" 分数 .pkl 文件的目录')
    parser.add_argument('--bone_motion_dir', type=str, default=None, help='包含 "bone_motion" 分数 .pkl 文件的目录')
    parser.add_argument('--score_file_name', default='epoch1_test_score.pkl', 
                        help='在每个模态目录中要加载的 .pkl 文件的名称')
    parser.add_argument('--alphas', type=float, nargs='+', default=None, 
                        help='融合权重列表。如果进行搜索，则此参数被忽略。')
    parser.add_argument('--label_file', type=str, default=None, # <<< 外部标签文件路径
                        help='可选的外部真实标签文件路径。如果提供，将优先使用此文件中的标签。')
    
    parser.add_argument('--find_best_alphas', action='store_true',
                        help='如果设置此项，则搜索最佳alpha权重。')
    parser.add_argument('--alpha_search_granularity', type=int, default=10,
                        help='用于网格搜索：将1分为N份，即步长约1/N。')
    parser.add_argument('--alpha_search_method', type=str, default='grid', choices=['grid', 'random'],
                        help='搜索alphas的方法。')
    parser.add_argument('--random_search_iterations', type=int, default=1000,
                        help='随机搜索的迭代次数。')

    arg = parser.parse_args()

    # --- 确定激活的流和对应的目录 (与之前相同) ---
    stream_configs = [
        ('joint', arg.joint_dir), ('bone', arg.bone_dir),
        ('joint_motion', arg.joint_motion_dir), ('bone_motion', arg.bone_motion_dir)
    ]
    active_streams_info = [(name, path) for name, path in stream_configs if path is not None]
    if not active_streams_info: print("错误：没有提供任何模态流的目录。"); sys.exit(1)
    active_stream_names = [info[0] for info in active_streams_info]
    active_stream_paths = [info[1] for info in active_streams_info]
    num_active_streams = len(active_stream_names)

    print(f"数据集: {arg.dataset}")
    print(f"待加载的分数文件名: {arg.score_file_name}")
    print(f"将要融合的模态流: {active_stream_names}")

    # --- 步骤1: 优先加载外部标签文件 (如果提供) ---
    label_np_array = None
    num_samples_expected_from_labels = None # 用于后续与分数文件样本数校验

    if arg.label_file:  # 检查用户是否提供了 --label_file 参数
        if os.path.exists(arg.label_file):
            print(f"信息：检测到 --label_file 参数，将优先从 {arg.label_file} 加载标签。")
            try:
                # 根据文件扩展名选择加载策略
                if arg.label_file.endswith('.pkl'):
                    with open(arg.label_file, 'rb') as f:
                        label_data_external = pickle.load(f)
                    
                    if isinstance(label_data_external, np.ndarray):
                        label_np_array = label_data_external
                    elif isinstance(label_data_external, list):
                        if label_data_external and isinstance(label_data_external[0], dict) and 'label' in label_data_external[0]:
                            label_np_array = np.array([int(info['label']) - 1 for info in label_data_external])
                        else:
                            label_np_array = np.array(label_data_external)
                    elif isinstance(label_data_external, dict) and 'labels' in label_data_external:
                        label_np_array = np.array(label_data_external['labels'])
                    else:
                        print(f"警告: 外部.pkl标签文件 {arg.label_file} 格式未知。")

                elif arg.label_file.endswith('.txt'):
                    print(f"信息：正在加载 .txt 格式的标签文件: {arg.label_file}")
                    try:
                        # 方案一：尝试直接加载为纯数字标签
                        label_np_array = np.loadtxt(arg.label_file, dtype=int)
                        print("  文件被成功解析为纯数字标签。")
                    except ValueError:
                        # 方案二：如果失败，则尝试按TD-GCN样本名格式解析
                        print(f"  直接加载为数字失败，尝试按TD-GCN样本名格式解析...")
                        try:
                            with open(arg.label_file, 'r') as f:
                                sample_names = [line.strip() for line in f.readlines() if line.strip()]
                            
                            labels = [int(name[name.find('A') + 1:name.find('A') + 4]) - 1 for name in sample_names]
                            
                            if not labels:
                                print(f"警告: 在 {arg.label_file} 中没有找到任何可以成功解析的样本名。")
                                label_np_array = None
                            else:
                                label_np_array = np.array(labels)
                                print(f"  成功从 {len(labels)} 个样本名中解析出标签。")
                        except Exception as e_parse:
                            print(f"警告: 按TD-GCN样本名格式解析 '{arg.label_file}' 时也失败了: {e_parse}")
                            label_np_array = None
                else:
                    print(f"警告: 不支持的外部标签文件格式 {arg.label_file}。")

                # 加载完成后的统一检查
                if label_np_array is not None:
                    num_samples_expected_from_labels = len(label_np_array)
                    print(f"标签已从外部文件 {arg.label_file} 加载。样本数: {num_samples_expected_from_labels}")
                else:
                    print(f"警告: 尝试从 {arg.label_file} 加载标签失败或未提取到标签。将尝试从分数文件加载。")

            except Exception as e_label_load:
                print(f"从外部标签文件 {arg.label_file} 加载时发生错误: {e_label_load}。将尝试从分数文件加载标签。")
                label_np_array = None  # 确保回退
        else:
            print(f"警告: 指定的外部标签文件 {arg.label_file} 未找到。将尝试从分数文件加载标签。")
    else:
        print("信息: 未提供 --label_file 参数。将尝试从第一个分数文件中加载标签。")


    # --- 步骤2: 加载分数，并进行标签处理/校验 ---
    all_scores_tensors = []
    
    for i, stream_name in enumerate(active_stream_names):
        stream_dir = active_stream_paths[i]
        score_path = os.path.join(stream_dir, arg.score_file_name)
        print(f"正在加载分数: {score_path} (流: '{stream_name}')")
        try:
            with open(score_path, 'rb') as f: data_dict = pickle.load(f)
            
            if 'scores' not in data_dict:
                print(f"错误: 分数文件 {score_path} 缺少 'scores' 键。"); sys.exit(1)
            current_stream_scores = torch.from_numpy(data_dict['scores']).float()

            # 校验样本数量
            if num_samples_expected_from_labels is not None: # 如果标签已从外部加载，用其样本数校验
                if current_stream_scores.shape[0] != num_samples_expected_from_labels:
                    print(f"错误: 流 '{stream_name}' 的样本数 ({current_stream_scores.shape[0]}) "
                          f"与从标签文件加载的样本数 ({num_samples_expected_from_labels}) 不匹配。")
                    sys.exit(1)
            elif i == 0 : # 如果是第一个流，并且标签未从外部加载，则以此流的样本数为基准
                num_samples_expected_from_labels = current_stream_scores.shape[0]
            elif current_stream_scores.shape[0] != num_samples_expected_from_labels: # 后续流与第一个流的样本数比较
                print(f"错误: 流 '{stream_name}' 的样本数 ({current_stream_scores.shape[0]}) "
                      f"与第一个流 ({num_samples_expected_from_labels}) 不匹配。")
                sys.exit(1)

            all_scores_tensors.append(current_stream_scores)

            # 如果标签还未从外部文件加载 (label_np_array is None)，则尝试从当前分数文件加载
            if label_np_array is None:
                if 'labels' in data_dict:
                    current_pkl_labels = data_dict['labels']
                    if not isinstance(current_pkl_labels, np.ndarray): # 确保是numpy array
                        current_pkl_labels = np.array(current_pkl_labels)

                    if i == 0: # 如果是第一个流，则将其标签作为基准
                        label_np_array = current_pkl_labels
                        # 再次确认样本数
                        if num_samples_expected_from_labels != len(label_np_array):
                             print(f"错误：从 {score_path} 加载的标签数 ({len(label_np_array)}) "
                                   f"与分数样本数 ({num_samples_expected_from_labels}) 不匹配。"); sys.exit(1)
                        print(f"标签已从第一个分数文件 {score_path} 加载。样本数: {len(label_np_array)}")
                    elif not np.array_equal(current_pkl_labels, label_np_array): # 后续流，校验与基准是否一致
                        print(f"严重错误: 文件 {score_path} 中的标签与之前从分数文件加载的标签不匹配！"); sys.exit(1)
                elif i == 0: # 如果是第一个流，但它也没有标签，且外部标签也没加载
                    print(f"错误: 第一个分数文件 {score_path} 缺少 'labels' 键，并且未提供有效的外部标签文件。")
                    sys.exit(1)
            elif 'labels' in data_dict: # 如果已经从外部加载了标签，但分数文件里也有，则进行校验
                # (可选的额外校验)
                if not np.array_equal(data_dict['labels'], label_np_array):
                    print(f"警告: 文件 {score_path} 中的内部标签与从外部文件加载的标签不匹配！将优先使用外部文件标签。")
        
        except FileNotFoundError: print(f"错误: 分数文件未找到: {score_path}。"); sys.exit(1)
        except Exception as e: print(f"加载或处理分数文件 {score_path} 时发生错误: {e}"); traceback.print_exc(); sys.exit(1)

    if not all_scores_tensors: print("错误: 未能成功加载任何分数文件。"); sys.exit(1)
    if label_np_array is None: print("致命错误: 最终未能加载任何标签信息。"); sys.exit(1)
            
    label_tensor = torch.from_numpy(label_np_array).long()

    # --- 权重处理与融合评估 (这部分逻辑与你之前的版本完全相同) ---
    best_top1_acc = 0.0
    best_alphas_for_top1 = None
    final_evaluation_alphas = None

    if arg.find_best_alphas:
        print(f"\n--- 开始搜索最佳融合权重 (目标: Top-1 Acc, 方法: {arg.alpha_search_method}) ---")
        alphas_to_test_list = []
        if num_active_streams == 1:
            print("信息：只有一个激活流，权重固定为 [1.0]。")
            alphas_to_test_list = [np.array([1.0])]
        elif arg.alpha_search_method == 'grid':
            granularity = arg.alpha_search_granularity
            if granularity < 1: print("错误: 网格搜索粒度必须 >= 1。"); sys.exit(1)
            print(f"网格搜索粒度: {granularity} (步长约 {1.0/granularity:.3f})")
            possible_k_values = range(granularity + 1)
            temp_combinations = itertools.product(possible_k_values, repeat=num_active_streams)
            for combo_k in temp_combinations:
                if sum(combo_k) == granularity:
                    alphas_to_test_list.append(np.array(combo_k) / granularity)
            if not alphas_to_test_list:
                 possible_values = np.linspace(0, 1, granularity + 1)
                 candidate_alphas_unnormalized = list(itertools.product(*([possible_values] * num_active_streams)))
                 for combo in candidate_alphas_unnormalized:
                     s = sum(combo)
                     if s > 1e-6: alphas_to_test_list.append(np.array([c / s for c in combo]))
                 if alphas_to_test_list: alphas_to_test_list = np.unique(np.array(alphas_to_test_list).round(decimals=5), axis=0).tolist()
        elif arg.alpha_search_method == 'random':
            print(f"随机搜索迭代次数: {arg.random_search_iterations}")
            for _ in range(arg.random_search_iterations):
                alphas_to_test_list.append(np.random.dirichlet(np.ones(num_active_streams), size=1)[0])
        if not alphas_to_test_list: print("错误：未能生成任何用于测试的权重组合。"); sys.exit(1)
        print(f"总共将测试 {len(alphas_to_test_list)} 种权重组合...")
        for current_alphas_np in tqdm(alphas_to_test_list, desc="搜索权重"):
            current_alphas = current_alphas_np.tolist()
            if len(current_alphas) != num_active_streams: continue
            fused_score = torch.zeros_like(all_scores_tensors[0])
            for i, stream_scores_tensor in enumerate(all_scores_tensors):
                fused_score += stream_scores_tensor * current_alphas[i]
            current_top1_acc = calculate_top1_accuracy(fused_score, label_tensor)
            if current_top1_acc > best_top1_acc:
                best_top1_acc = current_top1_acc
                best_alphas_for_top1 = current_alphas
        print(f"\n搜索完成。")
        final_evaluation_alphas = best_alphas_for_top1

    if final_evaluation_alphas is None:
        alphas_cmd = arg.alphas
        if alphas_cmd is not None:
            if len(alphas_cmd) != num_active_streams: print(f"错误: 命令行 alphas 数量 ({len(alphas_cmd)}) 与激活流数量 ({num_active_streams}) 不匹配。"); sys.exit(1)
            print(f"信息：将直接使用用户指定的权重，总和为 {sum(alphas_cmd):.4f}，不进行归一化。")
            final_evaluation_alphas = alphas_cmd
        else:
            print("信息：未通过 --alphas 提供权重，且未搜索（或无结果）。使用预设/平均权重。")
            if 'ucla' in arg.dataset:
                if num_active_streams == 4: final_evaluation_alphas = [0.6, 0.6, 0.4, 0.4]; final_evaluation_alphas = [a/sum(final_evaluation_alphas) for a in final_evaluation_alphas] # 确保和为1
                elif num_active_streams == 2 and all(s in active_stream_names for s in ['joint', 'bone']): final_evaluation_alphas = [0.6,0.4]
                elif num_active_streams == 1: final_evaluation_alphas = [1.0]
                else: final_evaluation_alphas = [1.0 / num_active_streams] * num_active_streams if num_active_streams > 0 else []
            else: final_evaluation_alphas = [1.0 / num_active_streams] * num_active_streams if num_active_streams > 0 else []
        if not final_evaluation_alphas : print("错误：最终未能确定有效权重。"); sys.exit(1)
        print(f"使用命令行提供或默认的融合权重: {final_evaluation_alphas}")
        fused_score_final = torch.zeros_like(all_scores_tensors[0])
        for i, stream_scores_tensor in enumerate(all_scores_tensors):
            fused_score_final += stream_scores_tensor * final_evaluation_alphas[i]
        best_top1_acc = calculate_top1_accuracy(fused_score_final, label_tensor)

    # --- 最终结果输出 ---
    print(f"\n--- 最终融合结果 ({arg.dataset}) ---")
    print(f"融合的流: {active_stream_names}")
    weights_to_print = best_alphas_for_top1 if arg.find_best_alphas and best_alphas_for_top1 is not None else final_evaluation_alphas
    print(f"使用的权重: {[round(a, 4) for a in weights_to_print]} (总和: {sum(weights_to_print):.4f})") # 确保打印和为1
    print(f'最佳 Top1 Acc: {best_top1_acc * 100:.2f}%')



    # python ensemble.py \
    # --dataset shrec17 \
    # --label_file ensemble/shrec17_14.txt \
    # --joint_dir ./work_dir/shrec17/14joint \
    # --bone_dir ./work_dir/shrec17/14bone \
    # --joint_motion_dir ./work_dir/shrec17/14joint_motion \
    # --find_best_alphas \
    # --alpha_search_granularity 40

    # python ensemble.py \
    # --dataset nw-ucla \
    # --label_file data/nw-ucla/val_label.pkl \
    # --joint_dir ./work_dir/nw-ucla/joint \
    # --bone_dir ./work_dir/nw-ucla/bone \
    # --joint_motion_dir ./work_dir/nw-ucla/joint_motion \
    # --bone_motion_dir ./work_dir/nw-ucla/bone_motion \
    # --find_best_alphas \
    # --alpha_search_granularity 40

    # python ensemble.py \
    # --dataset ntu/xsub \
    # --label_file ./ensemble/NTU60_XSub_Val.txt \
    # --joint_dir ./work_dir/ntu60/xsub/joint \
    # --bone_dir ./work_dir/ntu60/xsub/bone \
    # --joint_motion_dir ./work_dir/ntu60/xsub/joint_motion \
    # --bone_motion_dir ./work_dir/ntu60/xsub/bone_motion \
    # --find_best_alphas \
    # --alpha_search_granularity 20
