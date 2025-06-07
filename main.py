# -*- coding: utf-8 -*-
# 文件名: main.py (v15.5 - 修正 work_dir 路径重复问题)
from __future__ import print_function
import argparse
import os
import sys
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
import traceback
from collections import OrderedDict

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
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description='骨骼动作识别训练器 (早期融合多模态)') # 描述可以根据你的实际项目修改
    parser.add_argument('--work-dir', default='./work_dir/default_run', help='主工作目录')
    parser.add_argument('--config', default=None, help='YAML 配置文件的路径') # 例如: config/ucla/my_experiment.yaml
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'model_size'], help='运行阶段 (train, test, model_size)')
    parser.add_argument('--modalities', type=str, default=['joint'], nargs='+', help='要处理的模态列表 (例如: joint bone)')
    parser.add_argument('--label-file', type=str, default=None, help='测试集真实标签文件路径 (主要用于最终评估)') # 例如: data/ucla/val_label.pkl
    parser.add_argument('--device', type=int, default=None, nargs='+', help='GPU 索引列表 (例如: 0 1, 或 -1 代表CPU)')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--model', default=None, help='模型类的导入路径 (例如: model.MyModel)')
    parser.add_argument('--model-args', action=DictAction, default={}, help='模型初始化参数 (字典形式, 例如: "{num_classes:10,dropout:0.5}")')
    parser.add_argument('--weights', default=None, help='预训练权重路径或继续训练的模型路径')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='加载权重时要忽略的层名关键字')
    parser.add_argument('--feeder', default=None, help='数据加载器类的导入路径 (例如: feeders.feeder_ucla.Feeder)')
    parser.add_argument('--train-feeder-args', action=DictAction, default={}, help='训练数据加载器参数 (字典形式)')
    parser.add_argument('--test-feeder-args', action=DictAction, default={}, help='测试数据加载器参数 (字典形式)')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader 工作进程数') # 根据你的CPU核心数调整
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小 (在config中指定)')
    parser.add_argument('--test-batch-size', type=int, default=None, help='测试批次大小 (在config中指定)')
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'AdamW'], help='优化器类型 (SGD 或 AdamW)')
    parser.add_argument('--base-lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='优化器的权重衰减系数')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量因子')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='SGD优化器是否使用Nesterov动量')
    parser.add_argument('--lr_scheduler', default='manual_multistep', help='学习率调度器类型 (当前主要依赖手动调整)') # 例如: manual_multistep, cosine (如果将来支持)
    parser.add_argument('--step', type=int, default=[60, 75], nargs='+', help='手动MultiStepLR的衰减轮次节点 (1-based epoch)')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='手动MultiStepLR的学习率衰减率')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='学习率预热的轮数')
    parser.add_argument('--warmup_lr', type=float, default=1.0e-5, help='预热阶段的起始学习率')
    parser.add_argument('--warmup-prefix', type=str2bool, default=True, help='(timm.CosineLRScheduler专用) Warmup是否计入总step') # 如果不用timm的scheduler可以移除
    parser.add_argument('--num-epoch', type=int, default=80, help='总训练轮数') # 你之前的配置是80或65
    parser.add_argument('--start-epoch', type=int, default=0, help='从指定轮数开始训练 (0-based)')
    parser.add_argument('--loss-type', type=str, default='SmoothCE', choices=['CE', 'SmoothCE', 'FocalSmoothCE'], help='损失函数类型') # 确保FocalSmoothCE也支持
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑系数 (用于SmoothCE或FocalSmoothCE)') # 新增
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数 (用于FocalSmoothCE)') # 新增
    parser.add_argument('--early-stop-patience', type=int, default=0, help='早停的耐心值 (0 表示不使用)')
    parser.add_argument('--log-interval', type=int, default=50, help='训练时打印日志的迭代间隔 (设为0则只在epoch末尾打印)')
    parser.add_argument('--eval-interval', type=int, default=1, help='每隔多少轮进行一次评估')
    parser.add_argument('--save-interval', type=int, default=0, help='每隔多少轮保存一次检查点模型 (0 表示不按间隔保存)')
    parser.add_argument('--save-epoch', type=int, default=0, help='从第几轮之后开始考虑周期性保存 (与 save_interval 配合)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='是否打印日志到控制台和文件')
    parser.add_argument('--save-score', type=str2bool, default=True, help='测试或评估时是否保存预测分数')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='评估时显示 Top-K 准确率')
    parser.add_argument('--base-channel', type=int, default=3, help='单个输入模态的基础通道数')
    parser.add_argument('--grad_clip', type=str2bool, default=True,help='是否启用梯度裁剪')
    parser.add_argument('--grad_max',type=float,default=10.0,help='梯度裁剪的最大范数 (如果 grad_clip 为 True)')
    # --- 新增参数，用于前置GCN (如果你的解耦模型需要) ---
    parser.add_argument('--use-pre-gcn', type=str2bool, default=False, help='是否在SpatialAttention前使用GCN层')
    parser.add_argument('--pre-gcn-layers', type=int, default=1, help='前置GCN的层数')
    parser.add_argument('--pre-gcn-hidden-dim', type=int, default=128, help='前置GCN的隐藏层/输出维度')
    parser.add_argument('--pre-gcn-dropout', type=float, default=0.1, help='前置GCN的dropout率')
    # --- 解耦模型的主要时间模型参数 ---
    parser.add_argument('--embedding-dim', type=int, default=128, help='输入嵌入和空间处理的特征维度')
    parser.add_argument('--temporal-model-type', default='gru', choices=['gru', 'transformer'], help='主要时间模型的类型')
    parser.add_argument('--temporal-hidden-dim', type=int, default=128, help='主要时间模型的隐藏/特征维度')
    parser.add_argument('--num-temporal-main-layers', type=int, default=2, help='主要时间模型的层数')
    parser.add_argument('--bidirectional-time-gru', type=str2bool, default=True, help='如果主时间模型是GRU，是否双向')
    parser.add_argument('--temporal-main-dropout', type=float, default=0.15, help='主要时间模型的dropout率')
    parser.add_argument('--temporal-main-n-heads', type=int, default=8, help='(Transformer用)主时间模型的头数')
    parser.add_argument('--temporal-main-ffn-dim', type=int, default=512, help='(Transformer用)主时间模型的FFN维度')
    parser.add_argument('--use-time-pos-enc', type=str2bool, default=True, help='是否为时间序列模型添加时间位置编码')
    # --- 空间注意力参数 (sa_*) ---
    parser.add_argument('--sa-local-use-conv-proj', type=str2bool, default=False, help='空间注意力局部路径是否用卷积投影')
    parser.add_argument('--sa-global-use-conv-proj', type=str2bool, default=False, help='空间注意力全局路径是否用卷积投影')
    # ... Provided_TA 的其他参数，如 provided_ta_freq_num ...
    parser.add_argument('--use-multiscale-temporal-after-main', type=str2bool, default=True, help='是否使用后续的MultiScaleTemporalModeling模块')
    # ... MultiScaleTemporalModeling 的参数通过 multiscale_temporal_args (DictAction) 传入 ...
    parser.add_argument('--multiscale-temporal-args', action=DictAction, default={'short_term_kernels':[1,3,5], 'long_term_kernels':[9], 'long_term_dilations':[5], 'conv_out_channels_ratio':0.3, 'fusion_hidden_dim_ratio':0.3, 'dropout_rate':0.2}, help='MultiScaleTemporalModeling的参数')
    parser.add_argument('--use-temporal-attn-after-main-specific', type=str2bool, default=True, help='是否使用后续的TemporalTransformerBlock模块')
    parser.add_argument('--num-temporal-layers-after-main', type=int, default=2, help='后续TemporalTransformerBlock的层数')
    return parser

def generate_final_report(final_arg, processor, score_file_path):
    """根据保存的分数文件生成最终评估报告"""
    if not score_file_path or not os.path.exists(score_file_path):
        print(f"提示: 未找到有效的评估分数文件 ({score_file_path})，无法生成最终报告。")
        if hasattr(processor, 'best_acc') and processor.best_acc > 0:
             print(f"Processor 内部记录的最佳验证准确率: {processor.best_acc * 100:.2f}% "
                   f"(Epoch {processor.best_acc_epoch if hasattr(processor, 'best_acc_epoch') else 'N/A'})")
        return

    print(f"\n--- Main.py: 基于最终保存的评估分数进行报告 ---")
    print(f"评估分数文件: {score_file_path}")
    try:
        with open(score_file_path, 'rb') as f: eval_data = pickle.load(f)
        
        if not (isinstance(eval_data, dict) and 'scores' in eval_data and 'labels' in eval_data):
            print(f"错误: 分数文件 {score_file_path} 内容格式不符合预期。"); return
        
        scores_from_file = np.array(eval_data['scores'])
        true_labels_from_file = np.array(eval_data['labels'])

        if scores_from_file.shape[0] == 0 or scores_from_file.shape[0] != true_labels_from_file.shape[0]:
            print(f"错误: 分数文件 {score_file_path} 中的分数或标签数量不匹配或为0。"); return

        scores_tensor = torch.from_numpy(scores_from_file).float()
        true_labels_tensor_for_eval = torch.from_numpy(true_labels_from_file).long()
        _, predict_label = torch.max(scores_tensor, 1)
        final_acc_val = accuracy_score(true_labels_tensor_for_eval.numpy(), predict_label.numpy())
        print(f"  准确率 (Top-1，基于已保存的评估结果): {final_acc_val * 100:.2f}%")
        
        try:
            num_classes = final_arg.model_args.get('num_classes', len(np.unique(true_labels_from_file)))
            target_names = [f'C{i}' for i in range(num_classes)]
            labels_for_cm_report = np.arange(num_classes)
            
            valid_indices = (true_labels_tensor_for_eval.numpy() < num_classes) & \
                            (predict_label.numpy() < num_classes) & \
                            (true_labels_tensor_for_eval.numpy() >=0) & \
                            (predict_label.numpy() >= 0)

            report = classification_report(
                true_labels_tensor_for_eval.numpy()[valid_indices], 
                predict_label.numpy()[valid_indices], 
                labels=labels_for_cm_report, target_names=target_names, zero_division=0
            )
            cm = confusion_matrix(
                true_labels_tensor_for_eval.numpy()[valid_indices], 
                predict_label.numpy()[valid_indices], labels=labels_for_cm_report
            )
            print("\n  最终分类报告 (基于已保存的评估结果):\n", report)
            print("\n  最终混淆矩阵 (基于已保存的评估结果):\n", cm)
            
            report_suffix = "train_best_model_eval" if final_arg.phase == 'train' else "test_eval"
            report_filename = f'final_evaluation_report_{report_suffix}.txt'
            report_path = os.path.join(final_arg.work_dir, report_filename)

            with open(report_path, 'w', encoding='utf-8') as f_report:
                epoch_info = ""
                if final_arg.phase == 'train' and hasattr(processor, 'best_acc_epoch') and processor.best_acc_epoch > 0:
                    epoch_info = f" (基于Epoch {processor.best_acc_epoch} 的最佳模型)"
                elif hasattr(final_arg, 'weights') and final_arg.weights:
                    epoch_info = f" (基于权重 {os.path.basename(final_arg.weights)})"
                
                f_report.write(f"最终评估准确率 (Top-1){epoch_info}: {final_acc_val * 100:.2f}%\n\n")
                f_report.write("分类报告:\n"); f_report.write(report)
                f_report.write("\n\n混淆矩阵:\n"); f_report.write(np.array2string(cm, separator=', '))
            print(f"  详细的最终评估报告已保存到: {report_path}")
        except Exception as e_report:
            print(f"警告: 生成或保存详细最终评估报告时出错: {e_report}")
    except FileNotFoundError:
        print(f"错误: 未找到最终评估分数文件 {score_file_path}")
    except Exception as e_load:
        print(f"错误: 在main.py中加载或评估分数文件 {score_file_path} 失败: {e_load}")
        traceback.print_exc()


if __name__ == '__main__':
    parser = get_parser()

    # 1. 解析命令行参数，这将包含用户在命令行中明确指定的值，
    #    以及那些命令行未指定但在parser中定义了默认值的参数。
    cmd_line_args_obj = parser.parse_args()

    # 2. 加载YAML配置文件 (如果通过 --config 在命令行中指定了路径)
    yaml_config = {}
    config_path_from_cmd = cmd_line_args_obj.config # 直接从解析结果中获取config路径

    if config_path_from_cmd:
        if os.path.exists(config_path_from_cmd):
            try:
                with open(config_path_from_cmd, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.load(f, Loader=Loader)
                if yaml_config is None: yaml_config = {}
                print(f"--- 已成功加载配置文件: {config_path_from_cmd} ---")
            except Exception as e:
                print(f"错误: 解析配置文件 {config_path_from_cmd} 失败: {e}"); sys.exit(1)
        else:
            print(f"警告: 找不到配置文件: {config_path_from_cmd}。将仅使用命令行参数和argparse默认值。")
    else:
        print("信息: 未通过 --config 提供配置文件。将使用命令行参数和argparse默认值。")

    # 3. 构建最终的配置对象 `final_arg`
    # 优先级顺序: 命令行指定的值 > YAML文件中的值 > argparse定义的默认值

    # 3a. 获取 argparse 定义的所有参数的默认值作为基础
    # 我们需要一个只包含原始默认值的对象，以便后续比较
    default_arg_ns = argparse.Namespace()
    for action in parser._actions:
        if action.dest != "help" and action.default != argparse.SUPPRESS:
                setattr(default_arg_ns, action.dest, action.default)

    # 3b. 以 argparse 默认值为基础，创建 final_arg
    final_arg = copy.deepcopy(default_arg_ns)

    # 3c. 使用 YAML 文件中的值来更新/覆盖 final_arg 中的 argparse 默认值
    if yaml_config:
        for key, yaml_value in yaml_config.items():
            if hasattr(final_arg, key): # 只更新在parser中定义的参数
                current_val_in_final_arg = getattr(final_arg, key)
                if isinstance(current_val_in_final_arg, dict) and isinstance(yaml_value, dict):
                    # print(f"DEBUG: YAML merging dict for '{key}'")
                    current_val_in_final_arg.update(yaml_value)
                else:
                    # print(f"DEBUG: YAML overriding '{key}' to '{yaml_value}'")
                    setattr(final_arg, key, yaml_value)
            # else:
                # print(f"警告: YAML中的键 '{key}' 未在 argparse 中定义，将被忽略。")

    # 3d. 使用命令行中实际提供的值来最终覆盖 (具有最高优先级)
    # vars(cmd_line_args_obj) 包含了所有通过命令行解析出来的参数
    # (如果命令行没提供，但parser有默认值，这里会是那个默认值)
    for key, cmd_value in vars(cmd_line_args_obj).items():
        # 只有当命令行的值与该参数在argparse中的原始默认值不同时，
        # 或者该参数是'config'本身，才用命令行的值覆盖。
        # 这确保了如果命令行没有显式指定某个参数（使其保持为argparse默认值），
        # 而YAML中指定了该参数，那么YAML的值会被保留。
        # 如果命令行显式指定了，则命令行优先。
        original_parser_default = getattr(default_arg_ns, key, None) # 获取原始默认
        
        if cmd_value != original_parser_default or key == 'config':
            current_val_in_final_arg = getattr(final_arg, key, None) # 获取当前值 (可能来自YAML或默认)
            if isinstance(current_val_in_final_arg, dict) and isinstance(cmd_value, dict) and cmd_value:
                # print(f"DEBUG: CMD merging dict for '{key}'")
                current_val_in_final_arg.update(cmd_value)
            else:
                # print(f"DEBUG: CMD overriding '{key}' to '{cmd_value}'")
                setattr(final_arg, key, cmd_value)
    
    # 确保 final_arg.config 反映的是命令行传入的config路径（如果有的话）
    final_arg.config = config_path_from_cmd


    # --- 初始化随机种子 ---
    init_seed(final_arg.seed) # 确保 init_seed 函数已正确导入

    # --- 根据顶层参数调整和校验 model_args 和 feeder_args ---
    if not isinstance(final_arg.modalities, list): final_arg.modalities = [final_arg.modalities]
    if not final_arg.modalities: print("错误: `modalities` 参数不能为空列表。"); sys.exit(1)
    
    calculated_num_input_dim = final_arg.base_channel * len(final_arg.modalities)
    if not isinstance(final_arg.model_args, dict): final_arg.model_args = {} # 确保是字典
    
    if final_arg.model_args.get('num_input_dim') != calculated_num_input_dim:
        print(f"信息: model_args.num_input_dim (原值: {final_arg.model_args.get('num_input_dim')}) "
              f"将被根据顶层 modalities ({final_arg.modalities}) 和 base_channel ({final_arg.base_channel}) "
              f"计算的值 ({calculated_num_input_dim}) 设置或覆盖。")
    final_arg.model_args['num_input_dim'] = calculated_num_input_dim

    modalities_str_for_feeder = ','.join(final_arg.modalities)
    for fe_args_attr_name in ['train_feeder_args', 'test_feeder_args']:
        if not isinstance(getattr(final_arg, fe_args_attr_name), dict):
            setattr(final_arg, fe_args_attr_name, {})
        fe_args = getattr(final_arg, fe_args_attr_name)
        fe_args['modalities'] = modalities_str_for_feeder
        fe_args['data_path'] = modalities_str_for_feeder
        if 'base_channel' not in fe_args: fe_args['base_channel'] = final_arg.base_channel
        if 'num_classes' not in fe_args and 'num_classes' in final_arg.model_args:
            fe_args['num_classes'] = final_arg.model_args['num_classes']
        
        authoritative_len = final_arg.model_args.get('max_seq_len')
        if authoritative_len is None:
            authoritative_len = fe_args.get('max_len', fe_args.get('window_size'))
            if authoritative_len is None:
                authoritative_len = 64 # 硬编码的最终默认值
                print(f"警告: 未能在配置中明确找到序列长度，将为 {fe_args_attr_name} 使用默认值 {authoritative_len}")
        
        fe_args['max_len'] = authoritative_len
        if 'window_size' in fe_args: fe_args['window_size'] = authoritative_len
        if 'max_seq_len' not in final_arg.model_args or final_arg.model_args.get('max_seq_len') != authoritative_len :
            if authoritative_len is not None: # 只有当确定了长度才更新
                final_arg.model_args['max_seq_len'] = authoritative_len
                print(f"信息: model_args.max_seq_len 已同步为 {authoritative_len}")

    # 确保 batch_size 和 test_batch_size 是有效值
    if final_arg.batch_size is None or not isinstance(final_arg.batch_size, int) or final_arg.batch_size <= 0:
        print(f"错误: 训练 'batch_size' ({final_arg.batch_size}) 无效。请在命令行或配置文件中指定一个正整数。"); sys.exit(1)
    if final_arg.test_batch_size is None or not isinstance(final_arg.test_batch_size, int) or final_arg.test_batch_size <= 0:
        print(f"错误: 'test_batch_size' ({final_arg.test_batch_size}) 无效。请在命令行或配置文件中指定一个正整数。"); sys.exit(1)

    # --- 打印最终生效的配置 ---
    print("\n" + "="*15 + " 最终运行配置 " + "="*15)
    try:
        config_to_print_dict = OrderedDict() # 使用 OrderedDict 保持一点顺序
        # 定义希望优先打印的参数顺序和选择
        keys_to_print_ordered = [
            'config', 'work_dir', 'phase', 'device', 'seed', 'modalities', 'base_channel',
            'model', # model_args 会在下面单独处理
            'feeder', # feeder_args 会在下面单独处理
            'optimizer', 'base_lr', 'weight_decay', 'momentum', 'nesterov',
            'lr_scheduler', 'step', 'lr_decay_rate', 'warm_up_epoch', 'warmup_lr', 'min_lr',
            'num_epoch', 'start_epoch', 'batch_size', 'test_batch_size', 'num_worker',
            'loss_type', 'label_smoothing', 'focal_gamma',
            'grad_clip', 'grad_max', 'early_stop_patience',
            'eval_interval', 'log_interval', 'save_interval', 'save_epoch',
            'print_log', 'save_score', 'show_topk'
        ]
        for key in keys_to_print_ordered:
            if hasattr(final_arg, key):
                config_to_print_dict[key] = getattr(final_arg, key)
        
        # 打印有序的核心参数
        if config_to_print_dict:
             print(yaml.dump(dict(config_to_print_dict), default_flow_style=False, sort_keys=False, indent=2, Dumper=Dumper, allow_unicode=True))

        # 单独、更完整地打印字典类型的参数
        dict_args_to_print_separately = ['model_args', 'train_feeder_args', 'test_feeder_args']
        # 从 config_to_print_dict 中移除这些，因为它们将被单独打印
        for key in dict_args_to_print_separately:
            config_to_print_dict.pop(key, None) 

        for dict_key in dict_args_to_print_separately:
            if hasattr(final_arg, dict_key) and isinstance(getattr(final_arg, dict_key), dict):
                print(f"{dict_key}:") # 打印字典名
                # 使用 indent=4 使其层级更清晰
                print(yaml.dump(getattr(final_arg, dict_key), default_flow_style=False, sort_keys=False, indent=4, Dumper=Dumper, allow_unicode=True).strip())

    except Exception as e_print:
        print(f"警告: 打印最终配置时出错: {e_print}")
        print("将尝试直接打印 vars(final_arg):")
        try: print(vars(final_arg))
        except: print("直接打印 vars(final_arg) 也失败。")
    print("="*50 + "\n")

    # --- 工作目录创建 ---
    if not os.path.exists(final_arg.work_dir):
        try:
            os.makedirs(final_arg.work_dir)
            print(f"已创建主工作目录: {final_arg.work_dir}")
        except OSError as e:
            print(f"错误: 创建主工作目录 '{final_arg.work_dir}' 失败: {e}"); sys.exit(1)
    
    # --- 实例化并启动 Processor ---
    processor = None
    final_score_path = None
    try:
        processor = Processor(final_arg) # Processor内部不再重复打印完整参数
        final_score_path = processor.start()
    except KeyboardInterrupt:
        print(f"\n训练/测试被用户手动中断。")
    except Exception as e:
        print(f"\n处理过程中发生严重错误: {e}")
        traceback.print_exc()
    finally:
        if processor and hasattr(processor, 'train_writer') and processor.train_writer:
            try: processor.train_writer.close()
            except Exception: pass
        if processor and hasattr(processor, 'val_writer') and processor.val_writer:
            try: processor.val_writer.close()
            except Exception: pass
 
    # --- 训练结束后的最终评估报告 ---
    # (generate_final_report 函数的定义需要在 if __name__ == '__main__': 块之外)
    if final_arg.phase == 'train': # 只在训练阶段结束后尝试生成主报告
        generate_final_report(final_arg, processor, final_score_path)
    elif final_arg.phase == 'test': # 测试阶段也生成报告
        generate_final_report(final_arg, processor, final_score_path)


    print("\n程序正常结束。")