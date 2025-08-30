# -*- coding: utf-8 -*-
# 文件名: main.py (8.3 - 清理无用参数)
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
from utils import init_seed, str2bool, DictAction, import_class, LabelSmoothingCrossEntropy
from processor.processor import Processor

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description='骨骼动作识别训练器')
    # === 基础配置参数 ===
    parser.add_argument('--work-dir', default='./work_dir/default_run', help='主工作目录')
    parser.add_argument('--config', default=None, help='YAML 配置文件的路径') # 例如: config/ucla/my_experiment.yaml
    parser.add_argument('--phase', default='train',choices=['train', 'test', 'model_size'], help='运行阶段 (train, test, model_size)')
    parser.add_argument('--device', type=int, default=None, nargs='+', help='GPU 索引列表 (例如: 0 1, 或 -1 代表CPU)')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    # === 数据相关参数 ===
    parser.add_argument('--feeder', default=None, help='数据加载器类的导入路径')
    parser.add_argument('--train-feeder-args', action=DictAction, default={}, help='训练数据加载器参数')
    parser.add_argument('--test-feeder-args', action=DictAction, default={}, help='测试数据加载器参数')
    parser.add_argument('--num-worker', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--batch-size', type=int, default=None, help='训练批次大小')
    parser.add_argument('--test-batch-size', type=int, default=None, help='测试批次大小')
    # === 模型相关参数 ===
    parser.add_argument('--model', default=None, help='模型类的导入路径')
    parser.add_argument('--model-args', action=DictAction, default={}, help='模型初始化参数')
    parser.add_argument('--weights', default=None, help='预训练权重路径')
    # === 训练相关参数 ===
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'AdamW'], help='优化器类型')
    parser.add_argument('--base-lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='权重衰减系数')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量因子')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='SGD是否使用Nesterov动量')
    parser.add_argument('--num-epoch', type=int, default=80, help='总训练轮数')
    parser.add_argument('--loss-type', type=str, default='LSCE', choices=['CE', 'LSCE'], help='损失函数类型')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑系数')
    # === 学习率调度参数 ===
    parser.add_argument('--step', type=int, default=[60, 75], nargs='+', help='学习率衰减轮次节点')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='学习率预热轮数')
    
    # === 训练控制参数 ===
    parser.add_argument('--eval-interval', type=int, default=1, help='每隔多少轮进行一次评估')
    parser.add_argument('--save-interval', type=int, default=0, help='每隔多少轮保存一次检查点模型 (0 表示不按间隔保存)')
    parser.add_argument('--save-epoch', type=int, default=0, help='从第几轮之后开始考虑周期性保存 (与 save_interval 配合)')
    parser.add_argument('--lr_scheduler', default='manual_multistep', help='学习率调度器类型 (当前主要依赖手动调整)')
    parser.add_argument('--log-interval', type=int, default=50, help='训练时打印日志的迭代间隔 (设为0则只在epoch末尾打印)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='是否打印日志到控制台和文件')
    parser.add_argument('--save-score', type=str2bool, default=True, help='测试或评估时是否保存预测分数')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='评估时显示 Top-K 准确率')

    return parser

if __name__ == '__main__':
    parser = get_parser()

    # 1. 解析命令行参数
    cmd_line_args_obj = parser.parse_args()

    # 2. 加载YAML配置文件
    yaml_config = {}
    yaml_set_params = set()  # 记录YAML中实际设置的参数
    config_path_from_cmd = cmd_line_args_obj.config

    if config_path_from_cmd:
        if os.path.exists(config_path_from_cmd):
            try:
                with open(config_path_from_cmd, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.load(f, Loader=Loader)
                if yaml_config is None: 
                    yaml_config = {}
                else:
                    # 记录YAML中设置的所有参数（包括嵌套的）
                    yaml_set_params.update(yaml_config.keys())
                    for key, value in yaml_config.items():
                        if isinstance(value, dict):
                            for sub_key in value.keys():
                                yaml_set_params.add(f"{key}.{sub_key}")
                # 只保留一行简洁提示
                print(f"已加载配置文件: {config_path_from_cmd}")
            except Exception as e:
                print(f"错误: 解析配置文件 {config_path_from_cmd} 失败: {e}")
                sys.exit(1)
        else:
            print(f"错误: 找不到配置文件: {config_path_from_cmd}")
            sys.exit(1)
    else:
        print("信息: 未通过 --config 提供配置文件。将使用命令行参数和默认值。")

    # 3. 构建最终配置 - 按优先级：默认值 < YAML < 命令行
    
    # 3a. 获取 argparse 默认值作为基础
    default_arg_ns = argparse.Namespace()
    for action in parser._actions:
        if action.dest != "help" and action.default != argparse.SUPPRESS:
            setattr(default_arg_ns, action.dest, action.default)

    # 3b. 以默认值创建最终配置
    final_arg = copy.deepcopy(default_arg_ns)

    # 3c. 用YAML覆盖默认值
    if yaml_config:
        for key, yaml_value in yaml_config.items():
            if hasattr(final_arg, key):
                current_val = getattr(final_arg, key)
                if isinstance(current_val, dict) and isinstance(yaml_value, dict):
                    current_val.clear()
                    current_val.update(yaml_value)
                else:
                    setattr(final_arg, key, yaml_value)
            else:
                print(f"警告: YAML中的参数 '{key}' 在argparse中未定义，已忽略")

    # 3d. 最后用命令行覆盖（优先级最高，除了 config 本身）
    for key, cmd_value in vars(cmd_line_args_obj).items():
        if key == 'config':
            continue
        original_default = getattr(default_arg_ns, key, None)
        if cmd_value != original_default:  # 仅当用户显式设置
            current_val = getattr(final_arg, key, None)
            if isinstance(current_val, dict) and isinstance(cmd_value, dict) and cmd_value:
                current_val.update(cmd_value)
            else:
                setattr(final_arg, key, cmd_value)

    # 设置config路径
    final_arg.config = config_path_from_cmd
    
    # 将YAML设置的参数列表保存到final_arg中，供后续保存配置时使用
    final_arg._yaml_set_params = yaml_set_params

    # --- 初始化随机种子 ---
    init_seed(final_arg.seed)

    # --- 验证必要参数 ---
    if not final_arg.feeder:
        print("错误: 必须指定 'feeder' 参数。")
        sys.exit(1)
    
    if not final_arg.model:
        print("错误: 必须指定 'model' 参数。")
        sys.exit(1)

    # 验证batch_size
    if final_arg.batch_size is None or not isinstance(final_arg.batch_size, int) or final_arg.batch_size <= 0:
        print(f"错误: 训练 'batch_size' ({final_arg.batch_size}) 无效。请在YAML或命令行中指定一个正整数。")
        sys.exit(1)
    if final_arg.test_batch_size is None or not isinstance(final_arg.test_batch_size, int) or final_arg.test_batch_size <= 0:
        print(f"错误: 'test_batch_size' ({final_arg.test_batch_size}) 无效。请在YAML或命令行中指定一个正整数。")
        sys.exit(1)

    # --- 打印最终生效的配置（只显示YAML中设置的参数） ---
    print("\n" + "="*15 + " 运行配置 (YAML+命令行最终生效) " + "="*15)
    try:
        # 按照YAML文件中的原始顺序构建配置
        yaml_ordered_config = OrderedDict()
        
        # 首先添加YAML中设置的顶层参数，保持原始顺序
        if yaml_config:
            for key, value in yaml_config.items():
                if hasattr(final_arg, key):
                    yaml_ordered_config[key] = getattr(final_arg, key)
        
        # 然后添加其他在yaml_set_params中但不在yaml_config中的参数
        for param in yaml_set_params:
            if '.' not in param and param not in yaml_ordered_config and hasattr(final_arg, param):
                yaml_ordered_config[param] = getattr(final_arg, param)
        
        if yaml_ordered_config:
            print(yaml.dump(dict(yaml_ordered_config), default_flow_style=False, 
                          sort_keys=False, indent=2, Dumper=Dumper, allow_unicode=True))
    except Exception as e_print:
        print(f"警告: 打印最终配置时出错: {e_print}")
    print("="*70 + "\n")

    # --- 工作目录创建 ---
    if not os.path.exists(final_arg.work_dir):
        try:
            os.makedirs(final_arg.work_dir)
            print(f"已创建主工作目录: {final_arg.work_dir}")
        except OSError as e:
            print(f"错误: 创建主工作目录 '{final_arg.work_dir}' 失败: {e}")
            sys.exit(1)
    
    # --- 实例化并启动 Processor ---
    processor = None
    final_score_path = None
    try:
        processor = Processor(final_arg)
        final_score_path = processor.start()
    except KeyboardInterrupt:
        print("\n进程被用户中断。")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        if processor and hasattr(processor, 'train_writer') and processor.train_writer:
            try: processor.train_writer.close()
            except Exception: pass
        if processor and hasattr(processor, 'val_writer') and processor.val_writer:
            try: processor.val_writer.close()
            except Exception: pass

    print("\n程序正常结束。")