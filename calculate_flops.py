# 文件名: calculate_flops.py
# 描述: 严格根据YAML配置文件，计算SDT Transformer v2模型的FLOPs和参数量。
# 新模型特点：局部GCN + 全局Transformer + 动态融合，固定输入通道数为3 (x,y,z坐标)

import torch
import torch.nn as nn
import numpy as np
from thop import profile, clever_format
import logging
import yaml
import argparse
import sys

# 尝试从你的项目导入模型，如果失败则给出提示
try:
    from model.sgt_net import Model
except ImportError:
    print("错误：无法从 'model.sgt_net' 导入 Model。")
    print("请确保你在项目的根目录下运行此脚本，并且模型文件路径正确。")
    sys.exit(1)

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path):
    """从YAML文件加载配置"""
    logger.info(f"正在从 {path} 加载配置文件...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"错误：配置文件未找到于路径 '{path}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"错误：解析YAML文件时出错: {e}")
        sys.exit(1)


def create_sample_input(model_cfg, sequence_length, batch_size=1):
    """
    根据模型配置和指定的序列长度，创建用于FLOPs计算的样本输入。
    """
    num_nodes = model_cfg['num_nodes']
    num_person = model_cfg.get('num_person', 1)  # 如果YAML没写，默认为1人

    # 新模型固定输入通道数为3 (x,y,z坐标)
    num_input_dim = 3

    # 创建标准格式的输入: (B, C, T, V, M)
    # B: batch_size, C: channels(固定为3), T: time(sequence_length), V: nodes, M: persons
    logger.info(f"创建样本输入，维度: (B={batch_size}, C={num_input_dim}, T={sequence_length}, V={num_nodes}, M={num_person})")
    sample_input = torch.randn(batch_size, num_input_dim, sequence_length, num_nodes, num_person)
    return sample_input


def calculate_model_flops(model_cfg, sequence_length, device='cpu'):
    """
    为给定的模型配置和序列长度，计算一次FLOPs和参数量。
    """
    logger.info("正在实例化模型...")
    model = Model(model_cfg)
    model = model.to(device)
    model.eval()

    sample_input = create_sample_input(model_cfg, sequence_length)
    sample_input = sample_input.to(device)

    logger.info("正在使用 'thop' 计算FLOPs和参数量...")
    flops, params = profile(model, inputs=(sample_input,), verbose=False)
    
    # 使用thop的格式化工具
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    
    return flops, params, flops_formatted, params_formatted


def main():
    """
    主函数：解析命令行，加载配置，执行计算并生成报告。
    """
    parser = argparse.ArgumentParser(description="为SDT-Transformer v2模型计算FLOPs和参数量")
    parser.add_argument('--config', type=str, required=True,
                        help="指向模型配置YAML文件的路径，例如：'config/ntu/xsub/bone.yaml'")
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    full_config = load_config(args.config)

    # --- 2. 提取关键参数 ---
    # 提取模型蓝图
    if 'model_args' not in full_config:
        logger.error(f"错误：配置文件 {args.config} 中必须包含 'model_args' 部分。")
        sys.exit(1)
    model_cfg = full_config['model_args']

    # 验证新模型必需的参数
    required_params = ['num_nodes', 'num_class', 'dataset_name']
    missing_params = [param for param in required_params if param not in model_cfg]
    if missing_params:
        logger.error(f"错误：配置文件缺少新模型必需的参数: {missing_params}")
        logger.error("新模型需要: num_nodes, num_class, dataset_name")
        sys.exit(1)

    # 智能提取序列长度
    train_args = full_config.get('train_feeder_args', {})
    sequence_length = train_args.get('window_size')
    
    if sequence_length is None:
        logger.error(f"错误：在配置文件的 'train_feeder_args' 部分未找到 'window_size'。")
        logger.error("无法确定计算FLOPs所用的序列长度。")
        sys.exit(1)
    
    logger.info(f"成功从配置中读取到 'window_size' = {sequence_length}，将作为计算基准。")
    
    # 显示新模型的特点
    logger.info(f"数据集: {model_cfg.get('dataset_name', '未知')}")
    logger.info(f"关节点数: {model_cfg['num_nodes']}")
    logger.info(f"类别数: {model_cfg['num_class']}")
    logger.info("模型特点: 局部GCN + 全局Transformer + 动态融合")

    # --- 3. 执行计算 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"将使用设备: {device.upper()}")

    flops, params, flops_fmt, params_fmt = calculate_model_flops(
        model_cfg,
        sequence_length=sequence_length,
        device=device
    )
    
    # 计算每帧的FLOPs
    flops_per_frame = flops / sequence_length
    flops_per_frame_fmt = clever_format([flops_per_frame], "%.3f")[0]

    # --- 4. 生成最终报告 ---
    # 尝试从 work_dir 中提取一个有意义的名字用于报告标题
    try:
        report_title = full_config.get('work_dir', args.config).split('/')[-3].upper()
    except IndexError:
        report_title = "模型"

    print("\n" + "="*60)
    print(f"       {report_title} 复杂度最终报告 (基于 {args.config.split('/')[-1]})")
    print("="*60)
    print(f"{'配置来源':<25}: {args.config}")
    print(f"{'计算基准 (序列长度)':<25}: T = {sequence_length} (来自 'window_size')")
    print(f"{'数据集':<25}: {model_cfg.get('dataset_name', '未知')}")
    print(f"{'关节点数':<25}: {model_cfg['num_nodes']}")
    print(f"{'类别数':<25}: {model_cfg['num_class']}")
    print("-" * 60)
    print(f"{'模型总参数量 (Params)':<25}: {params_fmt}")
    print(f"{'总计算量 (FLOPs)':<25}: {flops_fmt}")
    print(f"{'每帧计算量 (FLOPs/Frame)':<25}: {flops_per_frame_fmt}")
    print("="*60)
    print("\n*注：FLOPs (Floating Point Operations) 指的是浮点运算次数。")
    print(" GFLOPs = 10^9 FLOPs. 这个值可以用来衡量模型的计算复杂度。")


if __name__ == "__main__":
    main()