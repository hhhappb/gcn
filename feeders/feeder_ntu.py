# -*- coding: utf-8 -*-
# 文件名: feeders/feeder_ntu.py
# 描述: 适用于 SDT_GRUs_Gesture 模型的 NTU RGB+D 数据集加载器。
#       基于 TD-GCN 的 npz 数据格式进行适配。

import os
import numpy as np
import pickle
import random
import math
import logging
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # 用于显示数据加载进度
from utils import collate_fn_filter_none
logger_utils_test = logging.getLogger(__name__) # 用 feeder 的 logger
logger = logging.getLogger(__name__) # 获取当前模块的 logger

# --- NTU RGB+D 骨骼对定义 (25个关节点，1-based) ---
# 这些骨骼对用于从关节点数据计算骨骼向量。
# 格式通常是 (父节点, 子节点) 或者 (端点1, 端点2)，具体取决于如何定义骨骼向量。
# TD-GCN 使用的 ntu_pairs 如下，其骨骼计算方式为 bone[v1] = joint[v1] - joint[v2]
# 这意味着骨骼向量存储在骨骼对的第一个关节点索引处。
NTU_PAIRS_TDGCN_STYLE = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (24,25) # (21,21)可能是为了让中心点骨骼为0，(24,25)是手尖
]


# --- 数据处理辅助函数 ---
def pad_and_mask_sequence(sequence_data, target_len, num_nodes, num_final_channels, pad_value=0.0):
    """
    将序列数据填充或截断到目标长度，并生成对应的掩码。
    :param sequence_data: 输入序列，形状 (当前帧数, 节点数, 处理后的通道数)
    :param target_len: 目标序列长度 (max_len)
    :param num_nodes: 节点数
    :param num_final_channels: 最终拼接后的通道数
    :param pad_value: 用于填充的值
    :return: (填充/截断后的序列, 布尔掩码)
    """
    current_len = sequence_data.shape[0]
    
    if current_len == 0: # 处理空序列的情况
        padded_sequence = np.full((target_len, num_nodes, num_final_channels), pad_value, dtype=np.float32)
        mask = np.zeros(target_len, dtype=bool)
        return padded_sequence, mask

    if current_len < target_len:
        pad_width = target_len - current_len
        padding_shape = (pad_width, num_nodes, num_final_channels)
        padding_array = np.full(padding_shape, pad_value, dtype=sequence_data.dtype)
        padded_sequence = np.concatenate((sequence_data, padding_array), axis=0)
        mask = np.concatenate((np.ones(current_len, dtype=bool), np.zeros(pad_width, dtype=bool)))
    elif current_len > target_len:
        padded_sequence = sequence_data[:target_len, :, :]
        mask = np.ones(target_len, dtype=bool)
    else:
        padded_sequence = sequence_data
        mask = np.ones(target_len, dtype=bool)
        
    return padded_sequence, mask

def calculate_bone_data_from_joints(joint_data_tvc, bone_pairs, num_nodes):
    """
    根据关节数据计算骨骼数据 (TD-GCN风格：bone[v1] = joint[v1] - joint[v2])。
    :param joint_data_tvc: 关节数据，形状 (帧数, 节点数, 3)
    :param bone_pairs: 骨骼对列表，元素为 (v1, v2)，1-based 索引
    :param num_nodes: 节点数
    :return: 骨骼数据，形状 (帧数, 节点数, 3)
    """
    T, V, C = joint_data_tvc.shape
    if C != 3:
        raise ValueError(f"计算骨骼数据时期望关节数据有3个通道 (xyz)，但得到 {C} 个。")
    if V != num_nodes:
        raise ValueError(f"关节数据中的节点数 {V} 与期望的节点数 {num_nodes} 不符。")

    bone_data_tvc = np.zeros_like(joint_data_tvc)
    for v1_one_based, v2_one_based in bone_pairs:
        v1_zero_based = v1_one_based - 1
        v2_zero_based = v2_one_based - 1
        if 0 <= v1_zero_based < num_nodes and 0 <= v2_zero_based < num_nodes:
            bone_data_tvc[:, v1_zero_based, :] = joint_data_tvc[:, v1_zero_based, :] - joint_data_tvc[:, v2_zero_based, :]
        # 对于像 (21,21) 这样的对，骨骼向量自然为0，无需特殊处理
    return bone_data_tvc

def calculate_motion_data(data_tvc):
    """
    计算帧间运动数据 (相邻帧的差值)。
    :param data_tvc: 输入数据 (关节或骨骼)，形状 (帧数, 节点数, 通道数)
    :return: 运动数据，形状 (帧数, 节点数, 通道数)
    """
    motion_data_tvc = np.zeros_like(data_tvc)
    T = data_tvc.shape[0]
    if T > 1:
        motion_data_tvc[:-1, :, :] = data_tvc[1:, :, :] - data_tvc[:-1, :, :]
        motion_data_tvc[-1, :, :] = motion_data_tvc[-2, :, :] # 最后一帧的运动用前一帧的运动填充
    # 如果 T=1 或 T=0，motion_data_tvc 将保持为全零，这是合理的
    return motion_data_tvc


class Feeder_NTU(Dataset):
    """
    NTU RGB+D 数据集加载器。
    适配 SDT_GRUs_Gesture 模型输入。
    数据源: .npz 文件 (通常包含 'x_train'/'x_test', 'y_train'/'y_test')。
    """
    def __init__(self,
                 data_path,                 # .npz 数据文件路径
                 label_path,                # .npz 标签文件路径 (与TD-GCN feeder不同，这里明确分离)
                 split='train',             # 'train' 或 'test'
                 max_len=150,               # 目标序列长度 (填充或截断到此长度)
                 modalities="joint",        # 需要加载的模态，逗号分隔字符串，如 "joint,bone"
                 num_nodes=25,              # NTU数据集通常是25个关节点
                 base_channel=3,            # 每个关节点的基础通道数 (xyz 为 3)
                 num_classes=60,            # 类别数 (NTU-60 为 60, NTU-120 为 120)
                 random_choose=False,       # 是否在训练时随机选择时间窗口
                 random_rot=False,          # 是否在训练时应用随机旋转增强
                 center_joint_idx=0,        # 用于中心化的关节点索引 (0-based)。NTU中，关节点1(索引0)为底部脊柱点。
                 debug=False,               # Debug模式，只加载少量样本
                 **kwargs):                 # 捕获其他可能的参数
        super().__init__()

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件未找到: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件未找到: {label_path}")

        self.data_path = data_path
        self.label_path = label_path
        self.split = split.lower()
        self.target_seq_len = max_len
        
        self.modalities = [m.strip().lower() for m in modalities.split(',') if m.strip()]
        valid_modalities = ['joint', 'bone', 'joint_motion', 'bone_motion']
        for m in self.modalities:
            if m not in valid_modalities:
                raise ValueError(f"不支持的数据模态: '{m}'. 支持的模态: {valid_modalities}")
        if not self.modalities:
            raise ValueError("必须至少指定一种数据模态 (参数 'modalities')")

        self.num_nodes = num_nodes
        self.base_channel = base_channel # xyz
        self.num_classes = num_classes
        # 最终输入给模型的通道数 = 基础通道数 * 模态数量
        self.num_input_dim = self.base_channel * len(self.modalities)

        self.random_choose = random_choose if self.split == 'train' else False
        self.random_rot = random_rot if self.split == 'train' else False
        self.center_joint_idx = center_joint_idx 
        self.debug = debug
        
        self.bone_pairs = NTU_PAIRS_TDGCN_STYLE # 使用 TD-GCN 风格的骨骼对

        self.load_data()

        logger.info(f"Feeder_NTU ({self.split}集) 初始化完成:")
        logger.info(f"  数据文件: {self.data_path}")
        logger.info(f"  标签文件: {self.label_path}")
        logger.info(f"  目标序列长度: {self.target_seq_len}")
        logger.info(f"  加载模态: {self.modalities} (总输入通道数: {self.num_input_dim})")
        logger.info(f"  样本数量: {len(self.labels)}")
        if self.debug:
            logger.warning(f"  DEBUG模式开启，只使用前 {len(self.skeletons_data)} 个样本。")

    def load_data(self):
        logger.info(f"开始从 .npz 文件加载数据和标签 ({self.split}集)...")
        
        # 加载骨骼数据
        data_npz = np.load(self.data_path)
        data_key = 'x_train' if self.split == 'train' else 'x_test'
        if data_key not in data_npz:
            raise KeyError(f"数据 .npz 文件中未找到键 '{data_key}': {self.data_path}")
        
        # 原始数据形状: (样本数 N, 原始总帧数 T_orig, M_max * V * C)
        # M_max: 文件中记录的最大人数 (通常为2)
        # V: 关节点数 (num_nodes)
        # C: 坐标数 (base_channel)
        raw_data_flat = data_npz[data_key] 
        N_samples, T_orig_from_file, Features_flat = raw_data_flat.shape

        expected_features_per_person = self.num_nodes * self.base_channel
        if Features_flat % expected_features_per_person != 0:
            raise ValueError(f"NPZ中的特征维度 ({Features_flat}) 不能被 (节点数*基础通道数={expected_features_per_person}) 整除。请检查数据格式。")
        M_in_file = Features_flat // expected_features_per_person
        
        logger.info(f"  从数据文件推断出原始帧长 T_orig={T_orig_from_file}, 最大人数 M_in_file={M_in_file}.")

        # Reshape: (N, T_orig, M, V, C) -> Transpose: (N, M, T_orig, V, C)
        try:
            self.skeletons_data = raw_data_flat.reshape(N_samples, T_orig_from_file, M_in_file, self.num_nodes, self.base_channel)
            self.skeletons_data = self.skeletons_data.transpose(0, 2, 1, 3, 4) # N, M, T, V, C
        except ValueError as e:
            logger.error(f"Reshape 原始数据失败。原始形状: {raw_data_flat.shape}, 目标 M={M_in_file}, V={self.num_nodes}, C={self.base_channel}. 错误: {e}")
            raise

        # 加载标签数据
        label_npz = np.load(self.label_path)
        label_key = 'y_train' if self.split == 'train' else 'y_test'
        if label_key not in label_npz:
            # 尝试其他可能的标签键，例如 'arr_0' (如果npz只有一个数组) 或 'labels'
            if 'arr_0' in label_npz: label_key = 'arr_0'
            elif 'labels' in label_npz: label_key = 'labels'
            else: raise KeyError(f"标签 .npz 文件中未找到期望的键 (尝试过 'y_train'/'y_test', 'arr_0', 'labels'): {self.label_path}")

        raw_labels = label_npz[label_key]
        if raw_labels.ndim == 2 and raw_labels.shape[1] > 1: # 可能是 one-hot 编码
            self.labels = np.argmax(raw_labels, axis=1).astype(int)
            logger.info(f"  标签从one-hot编码转换 (形状: {raw_labels.shape} -> {self.labels.shape})。")
        elif raw_labels.ndim == 1: # 已经是类别索引
            self.labels = raw_labels.astype(int)
            logger.info(f"  直接加载类别索引标签 (形状: {self.labels.shape})。")
        else:
            raise ValueError(f"不支持的标签形状: {raw_labels.shape} from {self.label_path}")

        if len(self.skeletons_data) != len(self.labels):
            raise ValueError(f"加载的数据样本数 ({len(self.skeletons_data)}) 与标签数 ({len(self.labels)}) 不匹配。")

        if self.debug:
            num_debug_samples = min(100, len(self.labels)) # 确保不超过实际样本数
            self.skeletons_data = self.skeletons_data[:num_debug_samples]
            self.labels = self.labels[:num_debug_samples]
            logger.info(f"  DEBUG模式: 数据截断为前 {num_debug_samples} 个样本。")
        
        logger.info("数据和标签加载完成。")


    def _temporal_sampling(self, joint_data_tvc, target_len):
        """
        对时间序列进行采样或选择窗口。
        :param joint_data_tvc: 输入的单个骨骼数据，形状 (原始帧数 T_orig, 节点数 V, 基础通道数 C)
        :param target_len: 目标帧数 (max_len)
        :return: 采样/选择后的数据，形状 (输出帧数 T_out, V, C)
        """
        T_orig = joint_data_tvc.shape[0]
        
        if T_orig == 0: # 如果原始数据就没有帧
            return np.zeros((0, self.num_nodes, self.base_channel), dtype=np.float32)

        if T_orig <= target_len: # 帧数不足或刚好，无需采样，后续会填充
            return joint_data_tvc
        else: # 帧数超过目标长度，需要采样
            if self.random_choose: # 训练时随机选择一个窗口
                start_idx = random.randint(0, T_orig - target_len)
                return joint_data_tvc[start_idx : start_idx + target_len, :, :]
            else: # 测试时，从中间取一个窗口 (或者可以实现均匀采样)
                start_idx = (T_orig - target_len) // 2
                return joint_data_tvc[start_idx : start_idx + target_len, :, :]

    def _random_rotation_augmentation(self, joint_data_tvc):
        """
        对骨骼数据应用随机旋转。
        :param joint_data_tvc: 形状 (帧数 T, 节点数 V, 3)
        :return: 旋转后的数据，形状 (T, V, 3)
        """
        if joint_data_tvc.shape[2] != 3: # 仅对xyz坐标应用
            return joint_data_tvc

        angle_range = 10 # 旋转角度范围 (+/- 10度)
        theta_x = np.random.uniform(-angle_range, angle_range) * (math.pi / 180.0)
        theta_y = np.random.uniform(-angle_range, angle_range) * (math.pi / 180.0)
        theta_z = np.random.uniform(-angle_range, angle_range) * (math.pi / 180.0)

        # Rotation matrix for X-axis
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(theta_x), -math.sin(theta_x)],
                       [0, math.sin(theta_x), math.cos(theta_x)]])
        # Rotation matrix for Y-axis
        Ry = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
                       [0, 1, 0],
                       [-math.sin(theta_y), 0, math.cos(theta_y)]])
        # Rotation matrix for Z-axis
        Rz = np.array([[math.cos(theta_z), -math.sin(theta_z), 0],
                       [math.sin(theta_z), math.cos(theta_z), 0],
                       [0, 0, 1]])
        
        R_combined = np.dot(Rz, np.dot(Ry, Rx))
        
        # (T, V, C) -> (T*V, C) then apply rotation, then reshape back
        original_shape = joint_data_tvc.shape
        joint_data_flat = joint_data_tvc.reshape(-1, 3)
        rotated_data_flat = np.dot(joint_data_flat, R_combined.T) # Transpose R for (N,3) x (3,3)
        
        return rotated_data_flat.reshape(original_shape).astype(joint_data_tvc.dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 获取0-based标签
        label_0based = self.labels[index]
        if not (0 <= label_0based < self.num_classes):
            logger.error(f"样本 {index}: 标签 {label_0based} 超出有效范围 [0, {self.num_classes-1}]。跳过此样本。")
            return None # 让 collate_fn 过滤掉这个样本

        # 获取该样本的所有骨骼数据: (M_in_file, T_orig, V, C)
        skeletons_for_sample = self.skeletons_data[index]

        # --- 骨骼选择与预处理 ---
        # 优先选择第一个骨骼 (M=0)。
        # TODO: 可以加入更复杂的选择逻辑，比如选择能量最大的骨骼，或者处理双人互动
        selected_joint_data_tvc = None
        if skeletons_for_sample.shape[0] > 0: # 确保至少有一个人
            # 取第一个人的数据 (T_orig, V, C)
            selected_joint_data_tvc = skeletons_for_sample[0, :, :, :].copy() 
            
            # 检查所选骨骼是否几乎为空 (所有关节点在所有帧中的坐标和接近于0)
            if np.sum(np.abs(selected_joint_data_tvc)) < 1e-6:
                 # logger.warning(f"样本 {index}: 选择的第一个骨骼数据接近全零。尝试选择第二个骨骼（如果存在）。")
                 if skeletons_for_sample.shape[0] > 1 and np.sum(np.abs(skeletons_for_sample[1,:,:,:])) > 1e-6 :
                     selected_joint_data_tvc = skeletons_for_sample[1, :, :, :].copy()
                     # logger.info(f"样本 {index}: 已切换到第二个骨骼。")
                 else:
                     logger.warning(f"样本 {index}: 第一个骨骼为空，且无有效备选骨骼。将使用全零数据。")
                     selected_joint_data_tvc = np.zeros((selected_joint_data_tvc.shape[0] if selected_joint_data_tvc.shape[0]>0 else 1, 
                                                        self.num_nodes, self.base_channel), dtype=np.float32)
        else: # 文件中记录的最大人数为0，或者此样本没有任何人的数据
            logger.warning(f"样本 {index}: 无有效骨骼数据。将使用全零数据。")
            # 获取一个典型的原始帧长用于创建dummy数据，如果所有样本帧长都一样，可以用self.skeletons_data.shape[2]
            # 否则，使用一个预设值或者target_seq_len
            t_for_dummy = self.skeletons_data.shape[2] if self.skeletons_data.ndim == 5 and self.skeletons_data.shape[2] > 0 else self.target_seq_len
            selected_joint_data_tvc = np.zeros((t_for_dummy, self.num_nodes, self.base_channel), dtype=np.float32)

        try:
            # 1. 中心化 (可选)
            if self.center_joint_idx is not None and 0 <= self.center_joint_idx < self.num_nodes:
                if selected_joint_data_tvc.shape[0] > 0: # 确保有帧数据
                    center_coord = selected_joint_data_tvc[0, self.center_joint_idx, :].copy()
                    selected_joint_data_tvc = selected_joint_data_tvc - center_coord
            
            # 2. 时间采样
            # 输入 (T_orig, V, C=3), 输出 (T_sampled, V, C=3)
            joint_data_sampled_tvc = self._temporal_sampling(selected_joint_data_tvc, self.target_seq_len)
            
            # 如果采样后帧数为0 (例如原始帧数也为0，且target_len>0), pad_and_mask_sequence 会处理
            if joint_data_sampled_tvc.shape[0] == 0 and self.target_seq_len > 0:
                # logger.debug(f"样本 {index}: 时间采样后帧数为0，后续将由padding处理。")
                # pad_and_mask_sequence 会创建一个全零的序列
                pass


            # 3. 随机旋转 (仅训练时)
            if self.random_rot: # 已在 __init__ 中根据 split 设置
                joint_data_sampled_tvc = self._random_rotation_augmentation(joint_data_sampled_tvc)

            # --- 生成并拼接模态 ---
            modal_data_list_tvc = [] # 存储 (T_sampled, V, C=3) 的模态数据
            
            # 确保基础的 joint_data_sampled_tvc 是 (T,V,3)
            if joint_data_sampled_tvc.ndim != 3 or joint_data_sampled_tvc.shape[1] != self.num_nodes or joint_data_sampled_tvc.shape[2] != self.base_channel:
                logger.error(f"样本 {index}: 采样/增强后的关节数据维度不正确: {joint_data_sampled_tvc.shape}，期望 (T, {self.num_nodes}, {self.base_channel})。跳过。")
                return None

            bone_data_xyz_cache = None # 用于缓存骨骼数据，避免重复计算

            for modality_name in self.modalities:
                if modality_name == 'joint':
                    modal_data_list_tvc.append(joint_data_sampled_tvc.copy())
                elif modality_name == 'bone':
                    bone_data_xyz_cache = calculate_bone_data_from_joints(joint_data_sampled_tvc, self.bone_pairs, self.num_nodes)
                    modal_data_list_tvc.append(bone_data_xyz_cache)
                elif modality_name == 'joint_motion':
                    modal_data_list_tvc.append(calculate_motion_data(joint_data_sampled_tvc))
                elif modality_name == 'bone_motion':
                    if bone_data_xyz_cache is None:
                        bone_data_xyz_cache = calculate_bone_data_from_joints(joint_data_sampled_tvc, self.bone_pairs, self.num_nodes)
                    modal_data_list_tvc.append(calculate_motion_data(bone_data_xyz_cache))
            
            if not modal_data_list_tvc:
                logger.error(f"样本 {index}: 未能生成任何模态数据。跳过。")
                return None
            
            # 拼接所有模态: (T_sampled, V, C=3*num_modalities)
            data_concatenated_tvc = np.concatenate(modal_data_list_tvc, axis=-1)

            # 验证拼接后的通道数
            if data_concatenated_tvc.shape[-1] != self.num_input_dim:
                logger.error(f"样本 {index}: 拼接后的通道数 ({data_concatenated_tvc.shape[-1]}) 与预期的总输入维度 ({self.num_input_dim}) 不符。跳过。")
                return None

            # --- 序列填充与掩码生成 ---
            # 输入 (T_sampled, V, num_input_dim), 输出 (target_seq_len, V, num_input_dim) 和 (target_seq_len,)
            data_padded_tvc, mask_np = pad_and_mask_sequence(data_concatenated_tvc, 
                                                             self.target_seq_len, 
                                                             self.num_nodes, 
                                                             self.num_input_dim)

            # --- 转换为 PyTorch 张量 ---
            data_tensor = torch.from_numpy(data_padded_tvc).float() # (max_len, num_nodes, num_input_dim)
            label_tensor = torch.tensor(label_0based, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool() # (max_len,)

            return data_tensor, label_tensor, mask_tensor, index

        except Exception as e:
            logger.error(f"处理样本 {index} (标签: {label_0based}) 时发生未知错误: {e}", exc_info=True)
            return None # 确保任何未捕获的异常都返回 None

# --- 用于独立测试 Feeder_NTU 的代码块 ---
if __name__ == '__main__':
    # 配置日志记录器，以便在测试时看到 Feeder 的日志输出
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # --- 用户需要提供实际的 .npz 文件路径来进行测试 ---
    # 请替换下面的路径为你本地 NTU RGB+D .npz 文件的路径
    # 例如，对于 NTU-60 Cross-View:
    # DUMMY_DATA_PATH_TRAIN = 'path/to/your/ntu_cv_x_train.npz' 
    # DUMMY_LABEL_PATH_TRAIN = 'path/to/your/ntu_cv_y_train.npz'
    # DUMMY_DATA_PATH_TEST = 'path/to/your/ntu_cv_x_test.npz'
    # DUMMY_LABEL_PATH_TEST = 'path/to/your/ntu_cv_y_test.npz'
    
    # 为了能直接运行示例，我们创建一个小型的虚拟 .npz 文件
    dummy_npz_file = 'dummy_ntu_data_for_feeder_test.npz'
    if not os.path.exists(dummy_npz_file):
        logger.info(f"创建虚拟 .npz 文件: {dummy_npz_file} 用于测试...")
        _N, _T, _M, _V, _C = 20, 70, 2, 25, 3 # 样本数，原始帧数，最大人数，关节点数，坐标数
        _x_train = np.random.rand(_N, _T, _M * _V * _C).astype(np.float32)
        _y_train_onehot = np.zeros((_N, 60), dtype=np.float32)
        _y_train_idx = np.random.randint(0, 60, size=_N)
        _y_train_onehot[np.arange(_N), _y_train_idx] = 1.0

        _x_test = np.random.rand(_N // 2, _T, _M * _V * _C).astype(np.float32)
        _y_test_onehot = np.zeros((_N // 2, 60), dtype=np.float32)
        _y_test_idx = np.random.randint(0, 60, size=(_N // 2))
        _y_test_onehot[np.arange(_N // 2), _y_test_idx] = 1.0
        np.savez(dummy_npz_file, x_train=_x_train, y_train=_y_train_onehot, x_test=_x_test, y_test=_y_test_onehot)
    
    DUMMY_DATA_PATH = dummy_npz_file
    DUMMY_LABEL_PATH = dummy_npz_file # 在这个虚拟例子中，标签和数据在同一个文件


    logger.info("\n--- 测试 Feeder_NTU (使用虚拟数据) ---")
    
    # 测试训练集加载
    try:
        train_feeder_params = {
            'data_path': DUMMY_DATA_PATH,
            'label_path': DUMMY_LABEL_PATH, 
            'split': 'train',
            'max_len': 100,
            'modalities': "joint,bone,joint_motion", # 测试多种模态
            'num_classes': 60,
            'num_nodes': 25,
            'base_channel': 3,
            'random_choose': True,
            'random_rot': True,
            'center_joint_idx': 0, # 以底部脊柱为中心
            'debug': False # 设为 False 以测试完整虚拟数据
        }
        feeder_train = Feeder_NTU(**train_feeder_params)
        logger.info(f"虚拟训练集 Feeder 长度: {len(feeder_train)}")

        if len(feeder_train) > 0:
            data, label, mask, idx = feeder_train[0] # 获取第一个样本
            if data is not None:
                logger.info(f"虚拟训练集样本 {idx}: 数据形状 {data.shape}, 标签 {label.item()}, Mask形状 {mask.shape}, Mask有效帧数 {mask.sum().item()}")
                # 期望形状: (max_len, num_nodes, num_modalities * base_channel)
                expected_channels = len(train_feeder_params['modalities'].split(',')) * train_feeder_params['base_channel']
                assert data.shape == (train_feeder_params['max_len'], train_feeder_params['num_nodes'], expected_channels)
                assert mask.shape == (train_feeder_params['max_len'],)
                logger.info("第一个训练样本形状和掩码校验通过。")
            else:
                logger.warning("获取的第一个训练样本为 None。")

            # 使用 DataLoader 进行测试 (需要 utils.collate_fn_filter_none)
            if 'collate_fn_filter_none' in globals() or hasattr(sys.modules.get('utils'), 'collate_fn_filter_none'):
                collate_fn_to_use = collate_fn_filter_none if 'collate_fn_filter_none' in globals() else sys.modules.get('utils').collate_fn_filter_none
                train_loader = DataLoader(feeder_train, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_to_use)
                first_batch = next(iter(train_loader), None)
                if first_batch:
                    xb, lb, mb, idxb = first_batch
                    logger.info(f"虚拟训练集 DataLoader 第一个批次: X形状 {xb.shape}, L形状 {lb.shape}, M形状 {mb.shape}")
                else:
                    logger.warning("虚拟训练集 DataLoader 返回的第一个批次为空 (可能所有样本都被过滤)。")
            else:
                logger.warning("无法找到 collate_fn_filter_none，跳过 DataLoader 测试。")
        else:
            logger.warning("虚拟训练集为空。")

    except Exception as e:
        logger.error(f"测试 Feeder_NTU (训练集) 时出错: {e}", exc_info=True)

    # 你可以添加类似的测试块用于 'test' split