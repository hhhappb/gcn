# -*- coding: utf-8 -*-
# 文件名: main.py (v15.4 - Robust YAML Priority and Dynamic Arg Handling)
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

# --- 导入模块 ---
try:
    from utils import init_seed, str2bool, DictAction, import_class, LabelSmoothingCrossEntropy
    from processor.processor import Processor
except ModuleNotFoundError as e:
    print(f"错误: 无法导入必要的模块 ({e})。请确保 utils.py 和 processor/processor.py 文件存在且路径正确。")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        try:
            from utils import init_seed, str2bool, DictAction, import_class, LabelSmoothingCrossEntropy
            from processor.processor import Processor
            print("成功从上级目录导入模块。")
        except ModuleNotFoundError:
            print("错误：即使添加了上级目录，仍然无法导入必要的模块。请检查文件结构或 PYTHONPATH。"); sys.exit(1)
    else:
         print("错误：无法导入必要的模块，且上级目录已在 sys.path 中。请检查文件结构或 PYTHONPATH。"); sys.exit(1)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
logger = logging.getLogger(__name__) # 使用 __main__ 作为 logger 名称

def get_parser():
    parser = argparse.ArgumentParser(description='骨骼动作识别训练器 (早期融合多模态)')
    parser.add_argument('--work-dir', default='./work_dir/default_run', help='主工作目录')
    parser.add_argument('--config', default=None, help='YAML 配置文件的路径')
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'model_size'], help='运行阶段') # 添加 model_size
    parser.add_argument('--modalities', type=str, default=['joint'], nargs='+', help='要融合的模态列表')
    parser.add_argument('--label-file', type=str, default=None, help='测试集真实标签文件路径 (主要用于最终评估)')
    parser.add_argument('--device', type=int, default=None, nargs='+', help='GPU 索引列表 (例如 0 1 或 -1 代表CPU)')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--model', default=None, help='模型类的导入路径 (例如 model.MyModel)')
    parser.add_argument('--model-args', action=DictAction, default={}, help='模型初始化参数 (字典形式)')
    parser.add_argument('--weights', default=None, help='预训练权重路径或继续训练的模型路径')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='加载权重时要忽略的层名关键字')
    parser.add_argument('--feeder', default=None, help='数据加载器类的导入路径 (例如 feeders.feeder_ucla.Feeder)')
    parser.add_argument('--train-feeder-args', action=DictAction, default={}, help='训练数据加载器参数 (字典形式)')
    parser.add_argument('--test-feeder-args', action=DictAction, default={}, help='测试数据加载器参数 (字典形式)')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小') # None 表示必须在config中指定
    parser.add_argument('--test-batch-size', type=int, default=None, help='测试批次大小') # None 表示必须在config中指定
    parser.add_argument('--optimizer', default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='优化器类型')
    parser.add_argument('--base-lr', type=float, default=0.001, help='基础学习率')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='学习率调度器使用的最小学习率 (例如 Cosine)')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='SGD 是否使用 Nesterov momentum')
    parser.add_argument('--grad-clip', type=str2bool, default=True, help='是否启用梯度裁剪')
    parser.add_argument('--grad-max', type=float, default=1.0, help='梯度裁剪的最大范数')
    parser.add_argument('--lr-scheduler', default='multistep', choices=['cosine', 'multistep'], help='学习率调度器类型')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+', help='MultiStepLR 的衰减轮次节点')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='MultiStepLR 的衰减率')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='学习率 Warmup 的轮数 (0 表示不使用)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warmup 的起始学习率')
    parser.add_argument('--warmup-prefix', type=str2bool, default=True, help='(timm.CosineLRScheduler) Warmup 是否计入总 step')
    parser.add_argument('--num-epoch', type=int, default=100, help='总训练轮数')
    parser.add_argument('--start-epoch', type=int, default=0, help='从指定轮数开始训练 (0-based)')
    parser.add_argument('--loss-type', type=str, default='CE', choices=['CE', 'SmoothCE'], help='损失函数类型')
    parser.add_argument('--early-stop-patience', type=int, default=0, help='早停耐心值 (0 表示不使用早停)')
    parser.add_argument('--log-interval', type=int, default=50, help='训练时打印日志的迭代间隔')
    parser.add_argument('--eval-interval', type=int, default=1, help='每隔多少轮进行一次评估')
    parser.add_argument('--save-interval', type=int, default=0, help='每隔多少轮保存一次模型 (0 表示只保存最佳和最终)')
    parser.add_argument('--save-epoch', type=int, default=0, help='从第几轮之后开始考虑周期性保存 (与 save_interval 配合)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='是否打印日志到控制台和文件')
    parser.add_argument('--save-score', type=str2bool, default=True, help='测试或评估时是否保存预测分数')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='评估时显示 Top-K 准确率')
    parser.add_argument('--base-channel', type=int, default=3, help='单个输入模态的基础通道数 (例如 xyz 是 3)')
    # 为了让早期融合的 main.py 不再需要 ensemble_weights
    parser.add_argument('--ensemble-weights', default=None, help=argparse.SUPPRESS)
    return parser

def load_labels(label_file_path, feeder_class_str=None, feeder_args=None, work_dir='.'):
    # ... (load_labels 函数保持不变，使用你之前提供的版本)
    print(f"尝试加载标签...")
    loaded_labels = None; actual_path_used = None; path_to_try = label_file_path
    source_msg = f"命令行/YAML (--label-file)"
    if not path_to_try or not os.path.exists(path_to_try):
        if feeder_args: path_to_try = feeder_args.get('val_pkl_path', feeder_args.get('label_path')); source_msg = f"feeder_args ('val_pkl_path' 或 'label_path')"
        else: path_to_try = None
    if path_to_try and not os.path.isabs(path_to_try) and not os.path.exists(path_to_try):
         base_dirs_to_try = [work_dir, feeder_args.get('root_dir', '.')]
         for base_dir in base_dirs_to_try:
             potential_path = os.path.join(base_dir, path_to_try)
             if os.path.exists(potential_path): path_to_try = potential_path; print(f"  推断标签文件路径为: {path_to_try}"); break
    if path_to_try and os.path.exists(path_to_try):
        actual_path_used = path_to_try; print(f"  尝试从以下路径加载标签 (来源: {source_msg}): {actual_path_used}")
        try:
            if actual_path_used.endswith('.txt'): labels = np.loadtxt(actual_path_used, dtype=int); loaded_labels = torch.from_numpy(labels).long(); print(f"  成功从 TXT 文件加载 {len(loaded_labels)} 个标签。")
            elif actual_path_used.endswith('.pkl'):
                 with open(actual_path_used, 'rb') as f: label_info = pickle.load(f)
                 print(f"  成功加载 PKL 文件，尝试解析标签... 数据类型: {type(label_info)}"); labels_list = []
                 if isinstance(label_info, list) and label_info and isinstance(label_info[0], dict) and 'label' in label_info[0]: print("    检测到 PKL 格式: list of dicts with 'label' key."); labels_list = [item['label'] - 1 for item in label_info if isinstance(item.get('label'), int)]
                 elif isinstance(label_info, dict):
                     if all(isinstance(v, int) for v in label_info.values()): print("    检测到 PKL 格式: dict of filename -> label (int)."); logger.warning("从 dict[str, int] 格式PKL加载标签，顺序可能不保！"); labels_list = [v - 1 for v in label_info.values()]
                     elif label_info and isinstance(list(label_info.values())[0], dict) and 'label' in list(label_info.values())[0]: print("    检测到 PKL 格式: dict of filename -> dict with 'label'."); logger.warning("从 dict[str, dict] 格式PKL加载标签，顺序可能不保！"); labels_list = [v['label'] - 1 for v in label_info.values() if isinstance(v.get('label'), int)]
                     else: print(f"    警告: 未知的 PKL 字典格式。")
                 else: print(f"    警告: 未知的 PKL 文件顶层格式 ({type(label_info)})。")
                 if labels_list:
                      num_classes = feeder_args.get('num_classes', 10) if feeder_args else 10; valid_labels = [lbl for lbl in labels_list if 0 <= lbl < num_classes]
                      if len(valid_labels) != len(labels_list): print(f"    警告: PKL含无效标签！过滤前{len(labels_list)}后{len(valid_labels)}。")
                      if valid_labels: loaded_labels = torch.tensor(valid_labels, dtype=torch.long); print(f"    成功从 PKL 解析并转换 {len(loaded_labels)} 个标签。")
                      else: print(f"    错误: PKL 解析后无有效标签。")
                 else: print(f"    错误: 无法从 PKL 文件内容提取标签列表。")
            else: print(f"警告: 不支持的标签文件格式: {actual_path_used}")
        except Exception as e: print(f"错误: 加载或处理标签文件 {actual_path_used} 失败: {e}"); traceback.print_exc()
    if loaded_labels is None and feeder_class_str and feeder_args:
        print("  所有文件加载方法失败，尝试实例化 Feeder 获取标签...")
        try:
            Feeder = import_class(feeder_class_str); temp_feeder_args = feeder_args.copy()
            temp_feeder_args.update({'split':'val', 'repeat':1, 'debug':False, 'data_path': temp_feeder_args.get('data_path','joint')})
            if 'root_dir' not in temp_feeder_args: raise ValueError("Feeder 需要 root_dir")
            feeder_instance = Feeder(**temp_feeder_args)
            if hasattr(feeder_instance, 'label') and isinstance(feeder_instance.label, list): print("    成功从 Feeder 实例获取标签。"); loaded_labels = torch.tensor(feeder_instance.label).long()
            else: print("    Feeder 实例没有 'label' 列表属性或类型不符。")
        except Exception as e: print(f"    实例化 Feeder 或获取标签失败: {e}")
    if loaded_labels is None: print("错误: 所有方法都无法加载有效的标签！评估将无法进行。")
    return loaded_labels

if __name__ == '__main__':
    parser = get_parser()

    # 1. 解析命令行参数，获取用户在命令行上显式指定的参数
    cmd_line_args_obj, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"警告: 未知命令行参数: {unknown_args}")

    # 2. 加载YAML配置文件（如果通过 --config 指定）
    yaml_config = {}
    config_path_from_cmd = getattr(cmd_line_args_obj, 'config', None)
    if config_path_from_cmd and os.path.exists(config_path_from_cmd):
        try:
            with open(config_path_from_cmd, 'r', encoding='utf-8') as f:
                yaml_config = yaml.load(f, Loader=Loader)
            if yaml_config is None: yaml_config = {}
            print(f"--- 已加载配置文件: {config_path_from_cmd} ---")
        except Exception as e:
            print(f"错误: 解析配置文件 {config_path_from_cmd} 失败: {e}"); sys.exit(1)
    elif cmd_line_args_obj.config: # 如果指定了 --config 但文件不存在
        print(f"警告: 找不到配置文件: {cmd_line_args_obj.config}。")

    # 3. 构建最终的 arg 对象，应用优先级：命令行 > YAML > argparse 默认值
    #    首先，创建一个包含所有 argparse 默认值的 Namespace 对象
    #    通过解析一个空列表来实现，这样所有定义在 parser 中的参数都会被赋予其 default 值
    final_arg = parser.parse_args([])

    #    然后，用 YAML 中的值更新/覆盖 argparse 的默认值
    if yaml_config:
        for key, value in yaml_config.items():
            if hasattr(final_arg, key):
                current_val = getattr(final_arg, key)
                # 对字典进行深度合并，YAML 中的键值对会覆盖或添加到 argparse 默认的字典中
                if isinstance(current_val, dict) and isinstance(value, dict):
                    # print(f"  从 YAML 合并字典 '{key}': {value} 到默认值: {current_val}")
                    current_val.update(value)
                else:
                    setattr(final_arg, key, value)
            else:
                print(f"警告: YAML中的键 '{key}' 在 argparse 定义中不存在，将被忽略。")
    
    #    最后，用命令行中实际指定的值覆盖 YAML/argparse默认值
    #    vars(cmd_line_args_obj) 包含了所有被 argparse 解析的命令行参数
    #    (即使没在命令行出现，但有默认值的也会在这里)
    for key, cmd_value in vars(cmd_line_args_obj).items():
        argparse_action = None
        for action in parser._actions: # 需要从 parser._actions 找到对应的 action
            if action.dest == key:
                argparse_action = action
                break
        
        # 只有当命令行参数的值不是其在 argparse 中定义的默认值时，才认为它是用户显式指定的覆盖
        # 或者，如果它是通过 --config 传入的，也需要保留
        if cmd_value is not None and (cmd_value != argparse_action.default if argparse_action else True) or key == 'config':
            current_val_before_cmd_override = getattr(final_arg, key, None)
            # 对字典进行深度合并
            if isinstance(current_val_before_cmd_override, dict) and isinstance(cmd_value, dict):
                # 只有当命令行提供的字典非空时才进行合并，避免空字典覆盖
                if cmd_value: 
                    # print(f"  通过命令行合并/覆盖字典 '{key}': {cmd_value}")
                    current_val_before_cmd_override.update(cmd_value)
            else:
                setattr(final_arg, key, cmd_value)

    # 确保记录的是实际使用的配置文件路径
    final_arg.config = config_path_from_cmd


    # --- 关键参数检查 ---
    required_general = ['work_dir', 'model', 'feeder', 'modalities', 'base_channel']
    if final_arg.phase == 'train': required_general.extend(['batch_size', 'test_batch_size'])
    required_train = ['optimizer', 'base_lr', 'num_epoch']
    
    missing = [k for k in required_general if getattr(final_arg, k, None) is None]
    if final_arg.phase == 'train':
        missing.extend([k for k in required_train if getattr(final_arg, k, None) is None])
        if not final_arg.model_args: missing.append('model_args (字典不能为空)')
        if not final_arg.train_feeder_args: missing.append('train_feeder_args (字典不能为空)')
        if not final_arg.test_feeder_args: missing.append('test_feeder_args (字典不能为空)')

    if missing: print(f"错误：缺少必要的配置参数或字典为空: {missing}。"); sys.exit(1)
    if not final_arg.modalities: print("错误: `modalities` 列表不能为空。"); sys.exit(1)
    if 'num_classes' not in final_arg.model_args or not isinstance(final_arg.model_args['num_classes'], int) or final_arg.model_args['num_classes'] <= 0:
        print(f"错误: model_args 中必须包含有效的 'num_classes' (正整数)。当前值: {final_arg.model_args.get('num_classes')}"); sys.exit(1)

    init_seed(final_arg.seed)

    # --- 准备最终传递给 Processor 的配置 (fused_arg) ---
    fused_arg = copy.deepcopy(final_arg) # 从已正确合并优先级后的 final_arg 开始

    modalities_to_fuse = sorted(list(set(fused_arg.modalities)))
    if not modalities_to_fuse: print("错误: 未指定有效的模态进行融合。"); sys.exit(1)
    print(f"--- 将融合以下模态进行训练/测试: {modalities_to_fuse} ---")

    # 1. 动态计算并强制设置 num_input_dim (对于早期融合是必须的)
    base_channel = fused_arg.base_channel
    calculated_num_input_dim = base_channel * len(modalities_to_fuse)
    if fused_arg.model_args.get('num_input_dim') != calculated_num_input_dim:
        print(f"信息: model_args.num_input_dim (原值: {fused_arg.model_args.get('num_input_dim')}) "
              f"将被根据 modalities ({modalities_to_fuse}) 和 base_channel ({base_channel}) "
              f"计算的值 ({calculated_num_input_dim}) 强制设置。")
    fused_arg.model_args['num_input_dim'] = calculated_num_input_dim
    print(f"基础通道数: {base_channel}, 融合模态数: {len(modalities_to_fuse)}")
    print(f"模型期望的总输入维度 (num_input_dim) 已设置为: {fused_arg.model_args['num_input_dim']}")

    # 2. 为 feeder_args 设置正确的 'modalities' 参数，但不覆盖文件路径 'data_path'
    #    同时，确保 feeder_args 中有 base_channel (如果 feeder 需要它)

    # modalities_to_fuse 是在之前定义的，例如 ['joint'] 或 ['joint', 'bone']
    # combined_data_path = ','.join(modalities_to_fuse) # 这个变量不再用于直接覆盖文件路径

    for fe_args_attr_name in ['train_feeder_args', 'test_feeder_args']:
        fe_args = getattr(fused_arg, fe_args_attr_name) # 获取 train_feeder_args 或 test_feeder_args 字典

        # 步骤 2.1: 保留 YAML 中为 data_path 设置的原始文件路径。
        # 我们不再用模态名称字符串覆盖它。
        # 因此，之前用于覆盖 data_path 的代码被移除或注释。
        # 之前的覆盖逻辑:
        # if fe_args.get('data_path') != combined_data_path: 
        #     print(f"信息: {fe_args_attr_name}.data_path (原值: '{fe_args.get('data_path')}') "
        #           f"将被根据 modalities 计算的值 ('{combined_data_path}') 强制设置。")
        # fe_args['data_path'] = combined_data_path  # <<--- 这行被移除了

        # 步骤 2.2: 确保 Feeder 知道要加载哪些模态。
        # fused_arg.modalities 是顶层命令行/YAML中指定的、当前实验要融合的模态列表。
        # 我们将这个列表转换成逗号分隔的字符串，并设置/覆盖 feeder_args 的 'modalities' 键。
        # 这样 Feeder 就可以根据这个 'modalities' 参数来决定加载和处理哪些数据流。
        modalities_str_for_feeder = ','.join(modalities_to_fuse) # 例如 "joint" 或 "joint,bone"
        
        # 只有当 feeder_args 中原有的 modalities 与程序决定的不同时才打印信息，避免冗余日志
        if fe_args.get('modalities') != modalities_str_for_feeder:
            print(f"信息: {fe_args_attr_name}.modalities (原值: '{fe_args.get('modalities')}') "
                  f"将设置为程序融合的模态 ('{modalities_str_for_feeder}')。")
        fe_args['modalities'] = modalities_str_for_feeder
        
        # 步骤 2.3: 确保 feeder_args 中也有 base_channel，如果 feeder 内部需要它。
        # base_channel 是在前面从 fused_arg.base_channel 获取的。
        if 'base_channel' not in fe_args: 
            fe_args['base_channel'] = base_channel
            # 可以选择性地打印这个信息，如果需要追踪 base_channel 的来源
            # print(f"信息: 为 {fe_args_attr_name} 添加 base_channel: {base_channel}")


    # 3. 同步 num_classes (以 model_args 为准)
    authoritative_num_classes = fused_arg.model_args['num_classes']
    for fe_args_attr_name in ['train_feeder_args', 'test_feeder_args']:
        fe_args = getattr(fused_arg, fe_args_attr_name)
        if fe_args.get('num_classes') != authoritative_num_classes:
            # print(f"信息: {fe_args_attr_name}.num_classes (原值: {fe_args.get('num_classes')}) "
            #       f"将与 model_args.num_classes ({authoritative_num_classes}) 同步。")
            pass # 不再打印这个，因为 feeder 内部通常也会用 model_args 的 num_classes
        fe_args['num_classes'] = authoritative_num_classes


    # 4. 同步 max_len/max_seq_len (以 YAML 中 model_args 或 train_feeder_args 的显式定义为准)
    #   优先级: YAML中 model_args.max_seq_len > YAML中 train_feeder_args.max_len > YAML中 train_feeder_args.window_size
    #   如果 YAML 中都没有，则使用 argparse 的默认值 (如果argparse有定义) 或代码内默认值
    
    yaml_model_max_seq_len = yaml_config.get('model_args', {}).get('max_seq_len')
    yaml_train_max_len = yaml_config.get('train_feeder_args', {}).get('max_len')
    yaml_train_window_size = yaml_config.get('train_feeder_args', {}).get('window_size')

    authoritative_max_len = None
    source_of_max_len = "argparse default or code default (64)"

    if yaml_model_max_seq_len is not None:
        authoritative_max_len = yaml_model_max_seq_len
        source_of_max_len = "YAML model_args.max_seq_len"
    elif yaml_train_max_len is not None:
        authoritative_max_len = yaml_train_max_len
        source_of_max_len = "YAML train_feeder_args.max_len"
    elif yaml_train_window_size is not None:
        authoritative_max_len = yaml_train_window_size
        source_of_max_len = "YAML train_feeder_args.window_size"
    else: # 如果 YAML 中完全没有定义，则使用最终 arg 对象中的值 (可能来自 argparse 默认)
        authoritative_max_len = fused_arg.model_args.get('max_seq_len', 
                                   fused_arg.train_feeder_args.get('max_len', 
                                   fused_arg.train_feeder_args.get('window_size', 64)))
        # source_of_max_len 已在上面初始化

    print(f"序列长度 (max_len/max_seq_len) 将使用: {authoritative_max_len} (来源: {source_of_max_len})")
    fused_arg.model_args['max_seq_len'] = authoritative_max_len
    fused_arg.train_feeder_args['max_len'] = authoritative_max_len
    fused_arg.test_feeder_args['max_len'] = authoritative_max_len
    if 'window_size' in fused_arg.train_feeder_args: fused_arg.train_feeder_args['window_size'] = authoritative_max_len
    if 'window_size' in fused_arg.test_feeder_args: fused_arg.test_feeder_args['window_size'] = authoritative_max_len


    # 5. 工作目录 (Processor 使用的实际工作目录)
    fused_modality_name = '_vs_'.join(modalities_to_fuse)
    # final_arg.work_dir 此处是顶层 work_dir (已合并命令行、YAML、argparse默认)
    final_processor_work_dir = os.path.join(final_arg.work_dir, fused_modality_name)
    if not os.path.exists(final_processor_work_dir):
        try: os.makedirs(final_processor_work_dir); print(f"创建融合 Processor 工作目录: {final_processor_work_dir}")
        except OSError as e: print(f"错误: 创建 Processor 工作目录 '{final_processor_work_dir}' 失败: {e}"); sys.exit(1)
    fused_arg.work_dir = final_processor_work_dir

    fused_arg.modalities = modalities_to_fuse # 只保留实际融合的列表
    if hasattr(fused_arg, 'ensemble_weights'): # ensemble_weights 不用于早期融合
        fused_arg.ensemble_weights = None

    # 打印最终融合配置
    print("\n--- 最终融合配置 (传递给 Processor) ---")
    print(f"  Work Dir: {fused_arg.work_dir}")
    print(f"  Phase: {fused_arg.phase}")
    print(f"  Fused Modalities: {fused_arg.modalities}")
    print(f"  Model Class: {fused_arg.model}")
    print(f"  Model Args: {fused_arg.model_args}")
    print(f"  Feeder Class: {fused_arg.feeder}")
    print(f"  Train Feeder Args: {fused_arg.train_feeder_args}")
    print(f"  Test Feeder Args: {fused_arg.test_feeder_args}")
    # ... (可以添加更多你想确认的 fused_arg 打印) ...
    print("-" * 30)

    # --- 执行训练或测试 ---
    processor = None; final_score_path = None; final_best_acc = 0.0
    try:
        processor = Processor(fused_arg)
        final_score_path = processor.start()
        if processor and hasattr(processor, 'best_acc') and processor.best_acc is not None: # 添加 None 检查
            final_best_acc = processor.best_acc * 100
    except KeyboardInterrupt: print(f"\n融合训练/测试被手动中断。")
    except Exception as e: print(f"\n处理融合模态时发生错误: {e}"); traceback.print_exc()
    finally:
        if processor and hasattr(processor, 'train_writer') and processor.train_writer:
            try: processor.train_writer.close()
            except Exception: pass
        if processor and hasattr(processor, 'val_writer') and processor.val_writer:
            try: processor.val_writer.close()
            except Exception: pass

    # --- 最终评估结果 ---
    print("\n=== 进入最终评估阶段 ===")
    if final_score_path and os.path.exists(final_score_path):
        print(f"训练/测试已完成，尝试从分数文件评估最终模型性能: {final_score_path}")
        # 使用 final_arg.work_dir 的上一级作为标签搜索基准
        true_labels = load_labels(final_arg.label_file, final_arg.feeder, fused_arg.test_feeder_args, os.path.dirname(fused_arg.work_dir))
        if true_labels is not None:
            print("真实标签加载成功，开始评估分数文件...")
            try:
                with open(final_score_path, 'rb') as f: score_dict = pickle.load(f)
                if not isinstance(score_dict, dict): print(f"错误: 分数文件 {final_score_path} 内容不是预期的字典格式。")
                else:
                    try:
                        # 确保分数文件中的 key 和 true_labels 的顺序能够对应
                        # Processor 保存的 index 应该是从 0 开始的样本索引
                        # score_keys = sorted([int(k) for k in score_dict.keys()]) # 确保是整数并排序
                        score_keys = sorted(score_dict.keys()) # 假设 key 本身就是可排序的（例如整数）
                        
                        # 检查分数数量是否与标签数量匹配
                        if len(score_keys) != len(true_labels):
                            print(f"[评估警告] 分数文件中的键数量 ({len(score_keys)}) 与标签数 ({len(true_labels)}) 不匹配！可能部分样本没有分数。")
                            # 尝试找到匹配的部分进行评估 (这种处理比较复杂，暂时简化)
                            # 这里假设如果数量不匹配，则不进行评估，或者你可以添加更复杂的匹配逻辑
                            raise ValueError("分数与标签数量不匹配")

                        scores_list = [score_dict[idx] for idx in score_keys]
                        scores_tensor = torch.from_numpy(np.stack(scores_list, axis=0)).float()
                    except Exception as e_sort_load: 
                        logger.error(f"处理或对齐分数文件字典时出错: {e_sort_load}"); 
                        traceback.print_exc(); # 打印更详细的堆栈
                        raise # 重新抛出以便上层知道评估失败
                    
                    # 再次确认维度（虽然上面已经检查过键的数量）
                    if scores_tensor.shape[0] != len(true_labels):
                         print(f"[评估错误] 最终分数张量样本数 ({scores_tensor.shape[0]}) 与标签数 ({len(true_labels)}) 不匹配！")
                    else:
                        print(f"分数与标签数量匹配 ({scores_tensor.shape[0]})，计算最终准确率...")
                        _, predict_label = torch.max(scores_tensor, 1)
                        final_test_acc_val = accuracy_score(true_labels.numpy(), predict_label.numpy()) # 重命名变量
                        print(f"\n最终模型在测试集上的准确率 (Top-1): {final_test_acc_val * 100:.2f}%")
                        try:
                            report = classification_report(true_labels.numpy(), predict_label.numpy(), zero_division=0, labels=np.arange(fused_arg.model_args['num_classes']), target_names=[f'C{i}' for i in range(fused_arg.model_args['num_classes'])])
                            cm = confusion_matrix(true_labels.numpy(), predict_label.numpy(), labels=np.arange(fused_arg.model_args['num_classes']))
                            print("\nClassification Report:\n", report); print("\nConfusion Matrix:\n", cm)
                            report_path = os.path.join(fused_arg.work_dir, 'final_test_report.txt')
                            with open(report_path, 'w', encoding='utf-8') as f_report:
                                f_report.write(f"Final Test Accuracy (Top-1): {final_test_acc_val * 100:.2f}%\n\nClassification Report:\n{report}\n\nConfusion Matrix:\n{np.array2string(cm, separator=', ')}")
                            print(f"详细评估报告已保存到: {report_path}")
                        except Exception as e_report_save: print(f"生成或保存详细评估报告时出错: {e_report_save}")
            except FileNotFoundError: print(f"错误: 未找到分数文件 {final_score_path}")
            except Exception as e_eval: print(f"错误: 加载或评估分数文件 {final_score_path} 失败: {e_eval}"); traceback.print_exc()
        else: print("无法加载真实标签，跳过最终测试准确率计算。")
        if final_arg.phase == 'train': print(f"\n融合模型训练过程中的最佳验证准确率 (来自Processor): {final_best_acc:.2f}%")
    else:
        print("未能找到有效的分数文件 (final_score_path 未定义或文件不存在)，无法进行最终评估。")
        if final_arg.phase == 'train' and final_best_acc > 0: print(f"\n融合模型训练过程中的最佳验证准确率 (来自Processor): {final_best_acc:.2f}%")
        elif final_arg.phase == 'train': print("\n训练过程中未能记录有效的最佳验证准确率。")
    print("\n程序退出。")