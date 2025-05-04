# -*- coding: utf-8 -*-
# 文件名: main.py (v15.1 - Early Fusion Multi-Modal Training, Optimized Label Loading)
from __future__ import print_function

import argparse
import os
import sys
import traceback
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import copy
import pickle
import torch
import numpy as np
import logging
import time
import shutil
import inspect
import glob
import csv

# --- 导入重构后的模块 ---
try:
    # 确保可以从 utils 导入必要的函数和类
    from utils import init_seed, str2bool, DictAction, import_class, LabelSmoothingCrossEntropy
    # 仍然需要 Processor 类来执行训练/评估
    from processor.processor import Processor
except ModuleNotFoundError as e:
    print(f"错误: 无法导入必要的模块 ({e})。请确保 utils.py 和 processor/processor.py 文件存在且路径正确。")
    # 尝试自动添加上级目录（如果 main.py 在子目录中）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        print(f"尝试将上级目录 '{parent_dir}' 添加到 sys.path")
        sys.path.insert(0, parent_dir)
        try:
            from utils import init_seed, str2bool, DictAction, import_class, LabelSmoothingCrossEntropy
            from processor.processor import Processor
            print("成功从上级目录导入模块。")
        except ModuleNotFoundError:
            print("错误：即使添加了上级目录，仍然无法导入必要的模块。请检查文件结构或 PYTHONPATH。")
            sys.exit(1)
    else:
         print("错误：无法导入必要的模块，且上级目录已在 sys.path 中。请检查文件结构或 PYTHONPATH。")
         sys.exit(1)


# --- 其他必要的导入 ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # 添加 classification_report

# 获取 logger 实例
logger = logging.getLogger(__name__) # 使用 __name__ 获取当前模块 logger

# --- 参数解析器 (保持不变) ---
def get_parser():
    """创建并配置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='骨骼动作识别训练器 (早期融合多模态)')
    # ... (参数定义与 ID 47 回复中的版本一致) ...
    parser.add_argument('--work-dir', default='./work_dir/default_run', help='主工作目录')
    parser.add_argument('--config', default=None, help='YAML 配置文件的路径')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help='运行阶段')
    parser.add_argument('--modalities', type=str, default=['joint'], nargs='+', help='要融合的模态列表')
    parser.add_argument('--ensemble-weights', type=float, default=None, nargs='+', help='(早期融合未使用)')
    parser.add_argument('--label-file', type=str, default=None, help='测试集真实标签文件路径')
    parser.add_argument('--device', type=int, default=None, nargs='+', help='GPU 索引列表')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--model', default=None, help='模型类的导入路径')
    parser.add_argument('--model-args', action=DictAction, default={}, help='模型初始化参数')
    parser.add_argument('--weights', default=None, help='预训练权重路径')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='忽略的层名')
    parser.add_argument('--feeder', default=None, help='数据加载器类的导入路径')
    parser.add_argument('--train-feeder-args', action=DictAction, default={}, help='训练数据加载器参数')
    parser.add_argument('--test-feeder-args', action=DictAction, default={}, help='测试数据加载器参数')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小')
    parser.add_argument('--test-batch-size', type=int, default=None, help='测试批次大小')
    parser.add_argument('--optimizer', default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='优化器类型')
    parser.add_argument('--base-lr', type=float, default=0.001, help='基础学习率')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='SGD Nesterov')
    parser.add_argument('--grad-clip', type=str2bool, default=True, help='启用梯度裁剪')
    parser.add_argument('--grad-max', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--lr-scheduler', default='multistep', choices=['cosine', 'multistep'], help='学习率调度器')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+', help='MultiStepLR 衰减节点')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='MultiStepLR 衰减率')
    parser.add_argument('--warm_up_epoch', type=int, default=15, help='Warmup epoch 数')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warmup 起始学习率')
    parser.add_argument('--warmup-prefix', type=str2bool, default=True, help='Warmup 计入总 step')
    parser.add_argument('--num-epoch', type=int, default=100, help='总训练 epoch 数')
    parser.add_argument('--start-epoch', type=int, default=0, help='起始 epoch')
    parser.add_argument('--loss-type', type=str, default='CE', choices=['CE', 'SmoothCE'], help='损失函数类型')
    parser.add_argument('--early-stop-patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--log-interval', type=int, default=50, help='日志记录间隔')
    parser.add_argument('--eval-interval', type=int, default=1, help='评估间隔')
    parser.add_argument('--save-interval', type=int, default=0, help='周期性保存间隔')
    parser.add_argument('--save-epoch', type=int, default=0, help='开始周期性保存的 epoch')
    parser.add_argument('--print-log', type=str2bool, default=True, help='打印日志')
    parser.add_argument('--save-score', type=str2bool, default=True, help='测试时保存分数')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='显示 Top-K 准确率')
    parser.add_argument('--base-channel', type=int, default=3, help='单个模态的基础通道数')
    return parser

# --- 加载标签函数 (优化 PKL 处理) ---
def load_labels(label_file_path, feeder_class_str=None, feeder_args=None, work_dir='.'): # 添加 work_dir
    """加载真实标签 (用于最终评估)，优化 PKL 处理逻辑"""
    print(f"尝试加载标签...")
    loaded_labels = None
    actual_path_used = None

    # 1. 优先尝试命令行/YAML直接指定的 --label-file
    path_to_try = label_file_path
    source_msg = f"命令行/YAML (--label-file)"

    # 2. 如果上面无效，尝试 feeder_args 中的路径
    if not path_to_try or not os.path.exists(path_to_try):
        if feeder_args:
            path_to_try = feeder_args.get('val_pkl_path', feeder_args.get('label_path')) # 优先 val_pkl_path
            source_msg = f"feeder_args ('val_pkl_path' 或 'label_path')"
        else:
            path_to_try = None

    # 3. 尝试补全相对路径
    if path_to_try and not os.path.isabs(path_to_try) and not os.path.exists(path_to_try):
         # 尝试在 work_dir 和 feeder 的 root_dir 下寻找
         base_dirs_to_try = [work_dir, feeder_args.get('root_dir', '.')]
         for base_dir in base_dirs_to_try:
             potential_path = os.path.join(base_dir, path_to_try)
             if os.path.exists(potential_path):
                 path_to_try = potential_path
                 print(f"  推断标签文件路径为: {path_to_try}")
                 break

    # 4. 如果找到有效路径，尝试加载
    if path_to_try and os.path.exists(path_to_try):
        actual_path_used = path_to_try
        print(f"  尝试从以下路径加载标签 (来源: {source_msg}): {actual_path_used}")
        try:
            if actual_path_used.endswith('.txt'):
                 labels = np.loadtxt(actual_path_used, dtype=int)
                 # 假设 TXT 存的是 0-based 或 1-based? 需要与数据对齐
                 # 这里假设 feeder 内部已经是 0-based, pkl 也是 1-based 转 0-based
                 # 为了安全，最好明确 txt 格式。如果 txt 是 1-based，这里要减 1
                 # loaded_labels = torch.from_numpy(labels - 1).long() # 如果 txt 是 1-based
                 loaded_labels = torch.from_numpy(labels).long() # 假设 txt 是 0-based
                 print(f"  成功从 TXT 文件加载 {len(loaded_labels)} 个标签。")
            elif actual_path_used.endswith('.pkl'):
                 with open(actual_path_used, 'rb') as f:
                     label_info = pickle.load(f)
                 print(f"  成功加载 PKL 文件，尝试解析标签... 数据类型: {type(label_info)}")
                 labels_list = []
                 if isinstance(label_info, list) and label_info and isinstance(label_info[0], dict) and 'label' in label_info[0]:
                     print("    检测到 PKL 格式: list of dicts with 'label' key.")
                     # 假设 PKL 中的 label 是 1-based，转换为 0-based
                     labels_list = [item['label'] - 1 for item in label_info if isinstance(item.get('label'), int)]
                 elif isinstance(label_info, dict):
                     # 尝试 {'file_name': label_int}
                     if all(isinstance(v, int) for v in label_info.values()):
                         print("    检测到 PKL 格式: dict of filename -> label (int).")
                         # 警告：字典顺序不保证，可能与分数对不上！
                         logger.warning("从 dict[str, int] 格式的 PKL 加载标签，顺序可能不保证！")
                         labels_list = [v - 1 for v in label_info.values()] # 假设 1-based
                     # 尝试 {'file_name': {'label': label_int}}
                     elif label_info and isinstance(list(label_info.values())[0], dict) and 'label' in list(label_info.values())[0]:
                         print("    检测到 PKL 格式: dict of filename -> dict with 'label'.")
                         logger.warning("从 dict[str, dict] 格式的 PKL 加载标签，顺序可能不保证！")
                         labels_list = [v['label'] - 1 for v in label_info.values() if isinstance(v.get('label'), int)] # 假设 1-based
                     else: print(f"    警告: 未知的 PKL 字典格式。")
                 else: print(f"    警告: 未知的 PKL 文件顶层格式 ({type(label_info)})。")

                 if labels_list:
                      # 检查标签范围
                      num_classes = feeder_args.get('num_classes', 10) if feeder_args else 10
                      valid_labels = [lbl for lbl in labels_list if 0 <= lbl < num_classes]
                      if len(valid_labels) != len(labels_list):
                          print(f"    警告: PKL 文件中包含无效或超出范围 [0, {num_classes-1}] 的标签！过滤前 {len(labels_list)} 个，过滤后 {len(valid_labels)} 个。")
                      if valid_labels:
                          loaded_labels = torch.tensor(valid_labels, dtype=torch.long)
                          print(f"    成功从 PKL 文件解析并转换 {len(loaded_labels)} 个标签。")
                      else: print(f"    错误: 从 PKL 解析后没有有效的标签。")
                 else: print(f"    错误: 无法从 PKL 文件内容中提取标签列表。")
            else: print(f"警告: 不支持的标签文件格式: {actual_path_used}")
        except Exception as e: print(f"错误: 加载或处理标签文件 {actual_path_used} 失败: {e}"); traceback.print_exc()

    # 5. 如果所有文件加载方法都失败，尝试实例化 Feeder (最后手段)
    if loaded_labels is None:
        if feeder_class_str and feeder_args:
            print("  所有文件加载方法失败，尝试实例化 Feeder 获取标签...")
            # ... (实例化 Feeder 获取标签的逻辑，同 ID 47) ...
            try:
                Feeder = import_class(feeder_class_str)
                temp_feeder_args = feeder_args.copy()
                temp_feeder_args['split'] = 'val' # 强制用 val
                temp_feeder_args['repeat'] = 1
                temp_feeder_args['debug'] = False
                # feeder 可能需要 data_path，即使只是为了加载标签
                if 'data_path' not in temp_feeder_args: temp_feeder_args['data_path'] = 'joint'
                if 'root_dir' not in temp_feeder_args: raise ValueError("Feeder 需要 root_dir")
                feeder_instance = Feeder(**temp_feeder_args)
                if hasattr(feeder_instance, 'label') and isinstance(feeder_instance.label, list):
                    print("    成功从 Feeder 实例获取标签。")
                    loaded_labels = torch.tensor(feeder_instance.label).long() # 假设 Feeder 内部是 0-based
                else: print("    Feeder 实例没有 'label' 列表属性或类型不符。")
            except Exception as e: print(f"    实例化 Feeder 或获取标签失败: {e}")

    if loaded_labels is None:
        print("错误: 所有方法都无法加载有效的标签！评估将无法进行。")
    return loaded_labels


# --- 主程序入口 ---
if __name__ == '__main__':
    parser = get_parser()
    # 解析命令行参数，覆盖 YAML 中的默认值
    p, unknown_args = parser.parse_known_args()
    if unknown_args: print(f"警告: 未知命令行参数: {unknown_args}")

    # --- 加载 YAML 配置 ---
    config_path = p.config
    default_arg = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f: default_arg = yaml.load(f, Loader=Loader)
            if default_arg is None: default_arg = {}
            print(f"--- 已加载配置文件: {config_path} ---")
            # 将 YAML 配置设置为默认值，命令行参数可以覆盖它们
            parser.set_defaults(**default_arg)
        except Exception as e: print(f"错误: 解析配置文件 {config_path} 失败: {e}"); sys.exit(1)
    else:
        if p.config: print(f"警告: 找不到配置文件: {config_path}。")
        else: print("未指定配置文件。")
        print("将仅使用命令行参数和代码内置默认值。")

    # 再次解析参数，以应用 YAML 的默认值和命令行的覆盖
    arg = parser.parse_args()

    # --- 合并字典参数 (命令行再次覆盖 YAML 中的字典) ---
    # 这里确保命令行中通过 DictAction 指定的字典项能更新 YAML 中对应的字典
    cmd_args_dict = vars(p) # 获取只包含命令行指定参数的字典
    for k, v_cmd in cmd_args_dict.items():
        # 如果命令行提供了这个参数 (v_cmd is not None)
        # 并且 arg 对象中有这个属性
        # 并且命令行值与默认值不同 (意味着用户确实在命令行指定了)
        if v_cmd is not None and hasattr(arg, k) and v_cmd != parser.get_default(k):
             v_arg = getattr(arg, k, None) # 获取 arg 对象中对应的值
             # 如果两者都是字典，则用命令行字典更新 arg 字典
             if isinstance(v_arg, dict) and isinstance(v_cmd, dict):
                 print(f"合并命令行字典参数 '{k}': {v_cmd}")
                 v_arg.update(v_cmd)
             # 否则，命令行参数已经在 arg 中覆盖了 YAML 值
    arg.config = config_path if config_path else arg.config # 记录使用的配置文件路径

    # --- 检查关键参数 ---
    required_general = ['work_dir', 'model', 'feeder', 'batch_size', 'test_batch_size', 'modalities', 'base_channel']
    required_train = ['optimizer', 'base_lr', 'num_epoch']
    missing = []
    for k in required_general:
         if getattr(arg, k, None) is None: missing.append(k)
    if arg.phase == 'train':
         for k in required_train:
              if getattr(arg, k, None) is None: missing.append(k)
         if not arg.model_args: missing.append('model_args')
         if not arg.train_feeder_args: missing.append('train_feeder_args')
         if not arg.test_feeder_args: missing.append('test_feeder_args')
    if missing: print(f"错误：缺少必要的配置参数: {missing}。"); sys.exit(1)
    if not arg.modalities: print("错误: `modalities` 列表不能为空。"); sys.exit(1)
    # 检查 model_args 中 num_classes 是否存在
    if 'num_classes' not in arg.model_args or not isinstance(arg.model_args['num_classes'], int) or arg.model_args['num_classes'] <= 0:
        print(f"错误: model_args 中必须包含有效的 'num_classes' 参数。当前值: {arg.model_args.get('num_classes')}")
        sys.exit(1)


    # --- 初始化随机种子 ---
    init_seed(arg.seed)

    # --- 准备融合配置 ---
    modalities_to_fuse = sorted(list(set(arg.modalities)))
    if not modalities_to_fuse: print("错误: 未指定有效的模态进行融合。"); sys.exit(1)
    print(f"--- 将融合以下模态进行训练/测试: {modalities_to_fuse} ---")

    combined_data_path = ','.join(modalities_to_fuse)

    # 动态计算并设置模型输入维度
    base_channel = arg.base_channel
    target_num_input_dim = base_channel * len(modalities_to_fuse)
    if 'num_input_dim' in arg.model_args and arg.model_args['num_input_dim'] != base_channel:
        print(f"警告: model_args 中的 num_input_dim ({arg.model_args['num_input_dim']}) 将被根据 modalities 和 base_channel ({base_channel}) 动态计算的值 ({target_num_input_dim}) 覆盖。")
    arg.model_args['num_input_dim'] = target_num_input_dim
    print(f"基础通道数: {base_channel}, 融合模态数: {len(modalities_to_fuse)}")
    print(f"模型期望的总输入维度 (num_input_dim) 已设置为: {target_num_input_dim}")

    # 确定融合训练的工作目录
    fused_modality_name = '_vs_'.join(modalities_to_fuse)
    combined_work_dir = os.path.join(arg.work_dir, fused_modality_name)
    if not os.path.exists(combined_work_dir):
        try:
            os.makedirs(combined_work_dir)
            print(f"创建融合工作目录: {combined_work_dir}")
        except OSError as e:
             print(f"错误: 创建工作目录 '{combined_work_dir}' 失败: {e}"); sys.exit(1)

    # --- 创建最终配置对象 (用于 Processor) ---
    fused_arg = copy.deepcopy(arg)
    fused_arg.work_dir = combined_work_dir
    # 更新 feeder 的 data_path
    fused_arg.train_feeder_args['data_path'] = combined_data_path
    fused_arg.test_feeder_args['data_path'] = combined_data_path
    # 确保 feeder 知道类别数和最大长度
    fused_arg.train_feeder_args['num_classes'] = fused_arg.model_args['num_classes']
    fused_arg.test_feeder_args['num_classes'] = fused_arg.model_args['num_classes']
    fused_arg.train_feeder_args['max_len'] = fused_arg.model_args.get('max_seq_len', fused_arg.train_feeder_args.get('window_size', 64))
    fused_arg.test_feeder_args['max_len'] = fused_arg.model_args.get('max_seq_len', fused_arg.test_feeder_args.get('window_size', 64))
    # 确保模型配置和 feeder 配置的 max_len 一致
    fused_arg.model_args['max_seq_len'] = fused_arg.train_feeder_args['max_len']

    # 清理不再需要的参数
    fused_arg.modalities = modalities_to_fuse # 只保留实际融合的列表
    fused_arg.ensemble_weights = None

    # 打印最终使用的融合配置
    print("\n--- 最终融合配置 (部分关键参数) ---")
    print(f"  Work Dir: {fused_arg.work_dir}")
    print(f"  Phase: {fused_arg.phase}")
    print(f"  Fused Modalities: {fused_arg.modalities}")
    print(f"  Feeder Data Path: {combined_data_path}")
    print(f"  Model Class: {fused_arg.model}")
    print(f"  Model Args: {fused_arg.model_args}") # 打印完整的模型参数
    print(f"  Feeder Class: {fused_arg.feeder}")
    print(f"  Train Feeder Args: {fused_arg.train_feeder_args}") # 打印训练 feeder 参数
    print(f"  Test Feeder Args: {fused_arg.test_feeder_args}")   # 打印测试 feeder 参数
    print(f"  Train Batch Size: {fused_arg.batch_size}")
    print(f"  Test Batch Size: {fused_arg.test_batch_size}")
    if fused_arg.phase == 'test': print(f"  Weights: {fused_arg.weights}")
    print("-" * 30)

    # --- 执行训练或测试 ---
    processor = None
    final_score_path = None
    final_best_acc = 0.0 # 记录训练中最佳验证集精度

    try:
        # 实例化 Processor，传入融合后的配置
        processor = Processor(fused_arg)
        # 启动训练或测试流程
        final_score_path = processor.start() # Processor.start() 返回最终测试分数文件路径
        if processor and hasattr(processor, 'best_acc'):
            final_best_acc = processor.best_acc * 100
    except KeyboardInterrupt:
        print(f"\n融合训练/测试被手动中断。")
    except Exception as e:
        print(f"\n处理融合模态时发生错误: {e}")
        traceback.print_exc()
    finally:
        # 关闭 TensorBoard writer
        if processor and hasattr(processor, 'train_writer') and processor.train_writer:
            try: processor.train_writer.close()
            except Exception: pass
        if processor and hasattr(processor, 'val_writer') and processor.val_writer:
            try: processor.val_writer.close()
            except Exception: pass

    # --- 最终评估结果 (从保存的分数文件加载) ---
    print("\n=== 进入最终评估阶段 ===")
    if final_score_path and os.path.exists(final_score_path):
        print(f"训练/测试已完成，尝试从分数文件评估最终模型性能: {final_score_path}")

        # --- 加载真实标签 ---
        label_file_to_load = arg.label_file # 优先使用 --label-file
        # 传递 fused_arg.test_feeder_args 给 load_labels
        true_labels = load_labels(label_file_to_load, arg.feeder, fused_arg.test_feeder_args, arg.work_dir)

        if true_labels is not None:
            print("真实标签加载成功，开始评估分数文件...")
            try:
                with open(final_score_path, 'rb') as f:
                    score_dict = pickle.load(f)

                # 确保 score_dict 是字典 {index: score_array}
                if not isinstance(score_dict, dict):
                     print(f"错误: 分数文件 {final_score_path} 内容不是预期的字典格式。")
                else:
                    # 按样本索引排序，并转换为 Tensor
                    # 注意：需要确保 score_dict 的 key 与 true_labels 的顺序对应
                    # 如果 Processor 保存的 index 是从 0 开始的连续索引，可以直接排序
                    try:
                        # 假设 key 是整数索引
                        sorted_indices = sorted(score_dict.keys())
                        # 检查索引是否连续且从0开始
                        # if sorted_indices != list(range(len(sorted_indices))):
                        #    logger.warning("分数文件中的样本索引不连续或不从0开始，结果可能不准确！")

                        scores_list = [score_dict[idx] for idx in sorted_indices]
                        scores_tensor = torch.from_numpy(np.stack(scores_list, axis=0)).float() # (N, C)
                    except KeyError:
                        logger.error("分数文件中的 key 不是预期的整数索引，无法排序。")
                        raise # 重新抛出异常
                    except Exception as e_sort:
                        logger.error(f"处理分数文件字典时出错: {e_sort}")
                        raise

                    if scores_tensor.shape[0] != len(true_labels):
                        print(f"[评估错误] 分数文件中的样本数 ({scores_tensor.shape[0]}) 与标签数 ({len(true_labels)}) 不匹配！")
                    else:
                        print(f"分数与标签数量匹配 ({scores_tensor.shape[0]})，计算最终准确率...")
                        _, predict_label = torch.max(scores_tensor, 1)
                        final_test_acc = accuracy_score(true_labels.numpy(), predict_label.numpy())
                        print(f"\n最终模型在测试集上的准确率 (Top-1): {final_test_acc * 100:.2f}%")

                        # 打印更详细的报告
                        try:
                            # zero_division=0 避免类别样本数为0时报错
                            report = classification_report(true_labels.numpy(), predict_label.numpy(), zero_division=0)
                            cm = confusion_matrix(true_labels.numpy(), predict_label.numpy())
                            print("\nClassification Report:\n", report)
                            print("\nConfusion Matrix:\n", cm)
                            # 可以将报告和混淆矩阵保存到文件
                            report_path = os.path.join(fused_arg.work_dir, 'final_test_report.txt')
                            with open(report_path, 'w', encoding='utf-8') as f_report:
                                f_report.write(f"Final Test Accuracy (Top-1): {final_test_acc * 100:.2f}%\n\n")
                                f_report.write("Classification Report:\n")
                                f_report.write(report)
                                f_report.write("\n\nConfusion Matrix:\n")
                                f_report.write(np.array2string(cm, separator=', '))
                            print(f"详细评估报告已保存到: {report_path}")
                        except Exception as e_report:
                            print(f"生成或保存详细评估报告时出错: {e_report}")

            except FileNotFoundError:
                print(f"错误: 未找到分数文件 {final_score_path}")
            except Exception as e:
                print(f"错误: 加载或评估分数文件 {final_score_path} 失败: {e}")
                traceback.print_exc()
        else:
            print("无法加载真实标签，跳过最终测试准确率计算。")

        # 打印训练过程中的最佳验证准确率
        if arg.phase == 'train':
            print(f"\n融合模型训练过程中的最佳验证准确率: {final_best_acc:.2f}%")

    else:
        print("未能找到有效的分数文件，无法进行最终评估。")
        # 如果是训练阶段，仍然打印最佳验证精度
        if arg.phase == 'train' and final_best_acc > 0:
             print(f"\n融合模型训练过程中的最佳验证准确率: {final_best_acc:.2f}%")
        elif arg.phase == 'train':
             print("\n训练过程中未能记录有效的最佳验证准确率。")


    print("\n程序退出。")