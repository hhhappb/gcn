# -*- coding: utf-8 -*-
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

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # 保留用于最终评估报告
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description='骨骼动作识别训练器')
    parser.add_argument('--work-dir', default='./work_dir/default_run', help='主工作目录')
    parser.add_argument('--config', default=None, help='YAML 配置文件的路径')
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'model_size'], help='运行阶段')
    # modalities 现在对于单流训练，应该只包含一个模态名称
    parser.add_argument('--modalities', type=str, default=['joint'], nargs='+', help='要处理的模态列表 (单流训练时只有一个)')
    # label_file 在后期融合流程中可能不是必须的，因为标签可以从分数文件中获取
    # parser.add_argument('--label-file', type=str, default=None, help='测试集真实标签文件路径 (主要用于最终评估)')
    parser.add_argument('--device', type=int, default=None, nargs='+', help='GPU 索引列表 (例如 0 1 或 -1 代表CPU)')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--model', default=None, help='模型类的导入路径')
    parser.add_argument('--model-args', action=DictAction, default={}, help='模型初始化参数')
    parser.add_argument('--weights', default=None, help='预训练权重路径或继续训练的模型路径')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='加载权重时要忽略的层名关键字')
    parser.add_argument('--feeder', default=None, help='数据加载器类的导入路径')
    parser.add_argument('--train-feeder-args', action=DictAction, default={}, help='训练数据加载器参数')
    parser.add_argument('--test-feeder-args', action=DictAction, default={}, help='测试数据加载器参数')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小')
    parser.add_argument('--test-batch-size', type=int, default=None, help='测试批次大小')
    parser.add_argument('--optimizer', default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='优化器类型')
    parser.add_argument('--base-lr', type=float, default=0.001, help='基础学习率')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='学习率调度器使用的最小学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='SGD 是否使用 Nesterov momentum')
    parser.add_argument('--grad-clip', type=str2bool, default=True, help='是否启用梯度裁剪')
    parser.add_argument('--grad-max', type=float, default=1.0, help='梯度裁剪的最大范数')
    parser.add_argument('--lr-scheduler', default='multistep', choices=['cosine', 'multistep'], help='学习率调度器类型')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+', help='MultiStepLR 的衰减轮次节点')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='MultiStepLR 的衰减率')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='学习率 Warmup 的轮数')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warmup 的起始学习率')
    parser.add_argument('--warmup-prefix', type=str2bool, default=True, help='(timm.CosineLRScheduler) Warmup 是否计入总 step')
    parser.add_argument('--num-epoch', type=int, default=100, help='总训练轮数')
    parser.add_argument('--start-epoch', type=int, default=0, help='从指定轮数开始训练')
    parser.add_argument('--loss-type', type=str, default='CE', choices=['CE', 'SmoothCE', 'FocalSmoothCE'], help='损失函数类型') # 添加 FocalSmoothCE
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑系数 (用于SmoothCE或FocalSmoothCE)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数 (用于FocalSmoothCE)')
    parser.add_argument('--early-stop-patience', type=int, default=0, help='早停耐心值 (0 表示不使用)')
    parser.add_argument('--log-interval', type=int, default=50, help='训练时打印日志的迭代间隔')
    parser.add_argument('--eval-interval', type=int, default=1, help='每隔多少轮进行一次评估')
    parser.add_argument('--save-interval', type=int, default=0, help='每隔多少轮保存一次模型 (0 表示只保存最佳和最终)')
    parser.add_argument('--save-epoch', type=int, default=0, help='从第几轮之后开始考虑周期性保存')
    parser.add_argument('--print-log', type=str2bool, default=True, help='是否打印日志到控制台和文件')
    parser.add_argument('--save-score', type=str2bool, default=True, help='测试或评估时是否保存预测分数')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='评估时显示 Top-K 准确率')
    parser.add_argument('--base-channel', type=int, default=3, help='单个输入模态的基础通道数 (例如 xyz 是 3)')
    return parser

# load_labels 函数在这里不再那么重要，因为标签会从 Processor 保存的 pkl 文件中读取，
# 或者由 ensemble 脚本自行处理。如果 Processor 内部评估仍需要，可以保留。
# 为了简化，暂时注释掉，如果 Processor 内部评估需要，可以恢复或调整。
# def load_labels(...):
#    ...

if __name__ == '__main__':
    parser = get_parser()
    cmd_line_args_obj, unknown_args = parser.parse_known_args()
    if unknown_args: print(f"警告: 未知命令行参数: {unknown_args}")

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
    elif cmd_line_args_obj.config:
        print(f"警告: 找不到配置文件: {cmd_line_args_obj.config}。")

    final_arg = parser.parse_args([]) # 获取 argparse 默认值
    if yaml_config: # 用 YAML 更新默认值
        for key, value in yaml_config.items():
            if hasattr(final_arg, key):
                if isinstance(getattr(final_arg, key), dict) and isinstance(value, dict):
                    getattr(final_arg, key).update(value)
                else:
                    setattr(final_arg, key, value)
            # else: print(f"警告: YAML中的键 '{key}' 在 argparse 定义中不存在，将被忽略。") # 可以减少这个日志

    for key, cmd_value in vars(cmd_line_args_obj).items(): # 用命令行指定的值覆盖
        argparse_action = next((action for action in parser._actions if action.dest == key), None)
        if cmd_value is not None and (cmd_value != argparse_action.default if argparse_action else True) or key == 'config':
            if isinstance(getattr(final_arg, key, None), dict) and isinstance(cmd_value, dict) and cmd_value:
                getattr(final_arg, key).update(cmd_value)
            else:
                setattr(final_arg, key, cmd_value)
    final_arg.config = config_path_from_cmd

    # --- 关键参数检查 ---
    # 对于单流训练，modalities 列表应该只有一个元素
    if len(final_arg.modalities) != 1:
        print(f"错误: 为实现后期融合的单流训练，'modalities' 参数应只包含一个模态名称，但得到: {final_arg.modalities}")
        sys.exit(1)
    
    required_general = ['work_dir', 'model', 'feeder', 'base_channel']
    if final_arg.phase == 'train': required_general.extend(['batch_size', 'test_batch_size'])
    missing = [k for k in required_general if getattr(final_arg, k, None) is None]
    if missing: print(f"错误：缺少必要的配置参数: {missing}。"); sys.exit(1)
    if 'num_classes' not in final_arg.model_args or final_arg.model_args['num_classes'] <= 0:
        print(f"错误: model_args 中必须包含有效的 'num_classes'。"); sys.exit(1)

    init_seed(final_arg.seed)
    fused_arg = copy.deepcopy(final_arg)

    # --- 针对单流训练的参数调整 ---
    # modalities_to_process 现在只包含一个模态
    modalities_to_process = sorted(list(set(fused_arg.modalities))) 
    print(f"--- 将处理单一模态进行训练/测试: {modalities_to_process[0]} ---")

    # 1. 动态计算并强制设置 num_input_dim
    #    对于单流，len(modalities_to_process) 为 1，所以 num_input_dim 就是 base_channel
    base_channel = fused_arg.base_channel
    calculated_num_input_dim = base_channel * len(modalities_to_process) # 应该是 base_channel * 1
    if fused_arg.model_args.get('num_input_dim') != calculated_num_input_dim:
        print(f"信息: model_args.num_input_dim (原值: {fused_arg.model_args.get('num_input_dim')}) "
              f"将被根据单模态和 base_channel ({base_channel}) 计算的值 ({calculated_num_input_dim}) 强制设置。")
    fused_arg.model_args['num_input_dim'] = calculated_num_input_dim # 确保这里是3 (或base_channel)

    # 2. 为 feeder_args 设置正确的 'data_path' (现在是单模态名称)
    single_modality_name_for_feeder = modalities_to_process[0]
    for fe_args_attr_name in ['train_feeder_args', 'test_feeder_args']:
        fe_args = getattr(fused_arg, fe_args_attr_name)
        
        # data_path 直接设为单模态名
        if fe_args.get('data_path') != single_modality_name_for_feeder:
            print(f"信息: {fe_args_attr_name}.data_path (原值: '{fe_args.get('data_path')}') "
                  f"将设置为当前流的模态 '{single_modality_name_for_feeder}'.")
        fe_args['data_path'] = single_modality_name_for_feeder
        
        # 'modalities' 键也设为单模态名，以防 Feeder 内部或其他地方仍有检查
        fe_args['modalities'] = single_modality_name_for_feeder
        
        if 'base_channel' not in fe_args: 
            fe_args['base_channel'] = base_channel
        fe_args['num_classes'] = fused_arg.model_args['num_classes'] # 同步num_classes

    # 3. 同步 max_len/max_seq_len (逻辑保持，但确保与单流Feeder的期望一致)
    yaml_model_max_seq_len = yaml_config.get('model_args', {}).get('max_seq_len')
    yaml_train_max_len = yaml_config.get('train_feeder_args', {}).get('max_len')
    yaml_train_window_size = yaml_config.get('train_feeder_args', {}).get('window_size')
    authoritative_max_len = None
    source_of_max_len = "argparse default or code default (e.g., 52 for UCLA)"
    if yaml_model_max_seq_len is not None: authoritative_max_len, source_of_max_len = yaml_model_max_seq_len, "YAML model_args.max_seq_len"
    elif yaml_train_max_len is not None: authoritative_max_len, source_of_max_len = yaml_train_max_len, "YAML train_feeder_args.max_len"
    elif yaml_train_window_size is not None: authoritative_max_len, source_of_max_len = yaml_train_window_size, "YAML train_feeder_args.window_size"
    else: authoritative_max_len = fused_arg.model_args.get('max_seq_len', fused_arg.train_feeder_args.get('max_len', fused_arg.train_feeder_args.get('window_size', 52))) # 默认52
    
    print(f"序列长度 (max_len/max_seq_len) 将使用: {authoritative_max_len} (来源: {source_of_max_len})")
    fused_arg.model_args['max_seq_len'] = authoritative_max_len
    fused_arg.train_feeder_args['max_len'] = authoritative_max_len
    fused_arg.test_feeder_args['max_len'] = authoritative_max_len
    if 'window_size' in fused_arg.train_feeder_args: fused_arg.train_feeder_args['window_size'] = authoritative_max_len
    if 'window_size' in fused_arg.test_feeder_args: fused_arg.test_feeder_args['window_size'] = authoritative_max_len

    # 4. 工作目录 (Processor 使用的实际工作目录)
    # final_arg.work_dir 是 YAML 中为该单模态流指定的 work_dir
    # 不需要再拼接模态名，因为 YAML 中每个流的 work_dir 已经不同了
    final_processor_work_dir = final_arg.work_dir 
    if not os.path.exists(final_processor_work_dir):
        try: os.makedirs(final_processor_work_dir); print(f"创建 Processor 工作目录: {final_processor_work_dir}")
        except OSError as e: print(f"错误: 创建 Processor 工作目录 '{final_processor_work_dir}' 失败: {e}"); sys.exit(1)
    fused_arg.work_dir = final_processor_work_dir

    fused_arg.modalities = modalities_to_process # 确保是单模态列表，如 ['joint']

    print("\n--- 最终单流配置 (传递给 Processor) ---")
    print(f"  Work Dir: {fused_arg.work_dir}")
    print(f"  Phase: {fused_arg.phase}")
    print(f"  Processing Modality: {fused_arg.modalities[0]}") # 只打印第一个（也是唯一一个）
    print(f"  Model Class: {fused_arg.model}")
    print(f"  Model Args: {fused_arg.model_args}")
    print(f"  Feeder Class: {fused_arg.feeder}")
    print(f"  Train Feeder Args: {fused_arg.train_feeder_args}")
    print(f"  Test Feeder Args: {fused_arg.test_feeder_args}")
    print("-" * 30)

    processor = None; final_score_path = None
    try:
        processor = Processor(fused_arg)
        final_score_path = processor.start() # Processor.start() 会返回评估分数文件的路径
    except KeyboardInterrupt: print(f"\n单流训练/测试被手动中断。")
    except Exception as e: print(f"\n处理单流模态时发生错误: {e}"); traceback.print_exc()
    finally:
        if processor and hasattr(processor, 'train_writer') and processor.train_writer:
            try: processor.train_writer.close()
            except Exception: pass
        if processor and hasattr(processor, 'val_writer') and processor.val_writer:
            try: processor.val_writer.close()
            except Exception: pass
 
    # --- 最终评估报告 (基于 Processor 保存的分数文件) ---
    # 这部分逻辑与你之前类似，但现在是针对单个流的评估结果
    if final_score_path and os.path.exists(final_score_path):
        print(f"最终评估：尝试从 Processor 保存的文件中加载分数和标签进行评估: {final_score_path}")
        try:
            with open(final_score_path, 'rb') as f:
                eval_data = pickle.load(f) 

            if not (isinstance(eval_data, dict) and 'scores' in eval_data and 'labels' in eval_data):
                print(f"错误: 分数文件 {final_score_path} 内容格式不符合预期。"); sys.exit(1)
            
            scores_from_file = eval_data['scores']      
            true_labels_from_file = eval_data['labels']

            if scores_from_file.shape[0] == 0 or scores_from_file.shape[0] != true_labels_from_file.shape[0]:
                print(f"错误: 分数文件 {final_score_path} 中的分数或标签数量不匹配或为0。"); sys.exit(1)
            
            print(f"从文件加载的分数与标签数量匹配 ({scores_from_file.shape[0]})，计算单流最终准确率...")
            
            scores_tensor = torch.from_numpy(scores_from_file).float()
            true_labels_tensor_for_eval = torch.from_numpy(true_labels_from_file).long() 
            _, predict_label = torch.max(scores_tensor, 1)
            final_test_acc_val = accuracy_score(true_labels_tensor_for_eval.numpy(), predict_label.numpy())
            
            print(f"\n当前流最终测试集准确率 (Top-1，基于已保存的评估结果): {final_test_acc_val * 100:.2f}%")
            
            try:
                num_classes_for_report = fused_arg.model_args['num_classes'] 
                target_names_report = [f'C{i}' for i in range(num_classes_for_report)]
                labels_for_cm = np.arange(num_classes_for_report)
                report = classification_report(true_labels_tensor_for_eval.numpy(), predict_label.numpy(), labels=labels_for_cm, target_names=target_names_report, zero_division=0)
                cm = confusion_matrix(true_labels_tensor_for_eval.numpy(), predict_label.numpy(), labels=labels_for_cm)
                print("\n当前流最终分类报告:\n", report)
                print("\n当前流最终混淆矩阵:\n", cm)
                
                report_path = os.path.join(fused_arg.work_dir, 'final_evaluation_report_STREAM.txt') 
                with open(report_path, 'w', encoding='utf-8') as f_report:
                    epoch_info_str = f" (Epoch {processor.best_acc_epoch})" if hasattr(processor, 'best_acc_epoch') and processor.best_acc_epoch > 0 else ""
                    if fused_arg.phase == 'test' and fused_arg.weights: epoch_info_str = f" (权重: {os.path.basename(fused_arg.weights)})"
                    f_report.write(f"当前流最终测试集准确率 (Top-1){epoch_info_str}: {final_test_acc_val * 100:.2f}%\n\n")
                    f_report.write("分类报告:\n"); f_report.write(report)
                    f_report.write("\n\n混淆矩阵:\n"); f_report.write(np.array2string(cm, separator=', '))
                print(f"详细的当前流最终评估报告已保存到: {report_path}")
            except Exception as e_report_save:
                print(f"生成或保存详细的当前流最终评估报告时出错: {e_report_save}")
        except Exception as e_eval_main:
            print(f"错误: 在main.py中加载或评估分数文件 {final_score_path} 失败: {e_eval_main}"); traceback.print_exc()
    else:
        if final_arg.phase == 'train' and hasattr(processor, 'best_acc') and processor.best_acc > 0 : 
             print(f"Processor 内部记录的最佳验证准确率: {processor.best_acc * 100:.2f}% (Epoch {processor.best_acc_epoch})")
        elif final_arg.phase == 'test':
             print(f"错误：测试阶段未能找到有效的评估分数文件 ({final_score_path})。")
    
    print(f"\n--- 单流 ({modalities_to_process[0]}) 处理完成 ---")
    print("程序退出。")