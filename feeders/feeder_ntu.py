# -*- coding: utf-8 -*-
# 文件名: feeders/feeder_ntu.py (v2.0 - 修正中心化和时间采样逻辑)
# 描述: 适用于 SDT_GRUs_Gesture 模型的 NTU RGB+D 数据集加载器。
#       适配由提供的预处理脚本生成的 .npz 数据格式。

import os
import numpy as np
import pickle
import random
import math
import logging
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 尝试导入项目根目录下的 utils ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils import collate_fn_filter_none
except ImportError as e_import:
    logging.warning(f"无法从项目根目录的 utils.py 导入 collate_fn_filter_none (错误: {e_import})。"
                              "如果直接运行此文件进行测试，DataLoader 可能无法正确工作。")
    def collate_fn_filter_none(batch): # pragma: no cover
        original_len = len(batch)
        valid_batch = []
        for item_idx, item in enumerate(batch):
            if item is None: continue
            if isinstance(item, (list, tuple)):
                if any(sub_item is None for sub_item in item): continue
            valid_batch.append(item)
        batch = valid_batch
        filtered_len = len(batch)
        if original_len > filtered_len:
            logging.warning(f"Collate (feeder_ntu fallback): 过滤掉 {original_len - filtered_len} 个无效样本。")
        if not batch:
            logging.warning("Collate (feeder_ntu fallback): 整个批次无效，返回 None。")
            return None
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

logger = logging.getLogger(__name__)

NTU_PAIRS_TDGCN_STYLE = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (24, 25), (23, 8) # 确保所有索引有效
]

def pad_and_mask_sequence(sequence_data, target_len, num_nodes, num_final_channels, pad_value=0.0):
    if sequence_data is None or sequence_data.size == 0:
        padded_sequence = np.full((target_len, num_nodes, num_final_channels), pad_value, dtype=np.float32)
        mask = np.zeros(target_len, dtype=bool)
        return padded_sequence, mask
    
    current_len = sequence_data.shape[0]
    if current_len == 0: 
        padded_sequence = np.full((target_len, num_nodes, num_final_channels), pad_value, dtype=np.float32)
        mask = np.zeros(target_len, dtype=bool)
        return padded_sequence, mask

    if sequence_data.ndim != 3 or sequence_data.shape[1] != num_nodes or sequence_data.shape[2] != num_final_channels:
         logger.error(f"Pad input seq shape {sequence_data.shape} mismatch expected (T, {num_nodes}, {num_final_channels}). Returning zeros.")
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
    T, V, C = joint_data_tvc.shape
    if C != 3: # 假设 bone 总是从 3 通道的 joint 计算
        raise ValueError(f"计算骨骼数据时期望关节数据有3个通道 (xyz)，但得到 {C} 个。")
    if V != num_nodes:
        raise ValueError(f"关节数据中的节点数 {V} 与期望的节点数 {num_nodes} 不符。")

    bone_data_tvc = np.zeros_like(joint_data_tvc) # 输出也是 (T,V,3)
    for v1_one_based, v2_one_based in bone_pairs:
        v1_zero_based = v1_one_based - 1
        v2_zero_based = v2_one_based - 1
        if 0 <= v1_zero_based < num_nodes and 0 <= v2_zero_based < num_nodes:
            bone_data_tvc[:, v1_zero_based, :] = joint_data_tvc[:, v1_zero_based, :] - joint_data_tvc[:, v2_zero_based, :]
    return bone_data_tvc

def calculate_motion_data(data_tvc): # 输入可以是 joint (T,V,3) 或 bone (T,V,3)
    motion_data_tvc = np.zeros_like(data_tvc)
    T = data_tvc.shape[0]
    if T > 1:
        motion_data_tvc[:-1, :, :] = data_tvc[1:, :, :] - data_tvc[:-1, :, :]
        motion_data_tvc[-1, :, :] = motion_data_tvc[-2, :, :] 
    return motion_data_tvc


class Feeder_NTU(Dataset):
    def __init__(self,
                 root_dir,
                 data_path,
                 label_path,
                 split='train',
                 max_len=150,
                 modalities="joint",
                 num_nodes=25,
                 base_channel=3,
                 num_classes=60,
                 random_choose=False, # 用于时间采样策略
                 random_rot=False,
                 rotation_angle_limit=10, # 新增：旋转角度限制
                 center_joint_idx=None, # <<<--- 默认不进行中心化，假设预处理已完成
                 debug=False,
                 label_source='from_label_path',
                 **kwargs): # 吸收其他未明确定义的参数
        super().__init__()

        self.root_dir = root_dir
        self.actual_data_path = os.path.join(self.root_dir, data_path)
        self.actual_label_path = os.path.join(self.root_dir, label_path)
        
        self.label_source = label_source.lower()
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
        self.base_channel = base_channel # 这是指原始 joint 数据的通道数，通常为3
        self.num_classes = num_classes
        # self.num_input_dim 是最终拼接后喂给模型的特征维度
        # 每个模态（joint, bone, motion）都基于 base_channel (通常是3) 来计算
        self.num_input_dim = self.base_channel * len(self.modalities)


        self.random_choose_sampling = random_choose if self.split == 'train' else False # 重命名以区分
        self.random_rot = random_rot if self.split == 'train' else False
        self.rotation_angle_limit = rotation_angle_limit # 保存旋转角度限制

        # 中心化处理逻辑
        if center_joint_idx is not None and not (0 <= center_joint_idx < self.num_nodes):
            logger.warning(f"提供的 center_joint_idx ({center_joint_idx}) 无效，将禁用 Feeder 内部中心化。")
            self.center_joint_idx = None
        else:
            self.center_joint_idx = center_joint_idx
        
        if self.center_joint_idx is not None:
            logger.info(f"Feeder 将执行基于关节点索引 {self.center_joint_idx} 的中心化。")
            logger.warning("如果您的数据在预处理阶段已经中心化，请在配置文件中将 center_joint_idx 设置为 null 或移除该参数，以禁用此处的重复中心化。")
        else:
            logger.info("Feeder 将不执行中心化操作 (center_joint_idx is None)。")

        self.debug = debug
        self.bone_pairs = NTU_PAIRS_TDGCN_STYLE

        self.load_data()

        logger.info(f"Feeder_NTU ({self.split}集) 初始化完成 (v2.0):")
        logger.info(f"  实际数据文件路径: {self.actual_data_path}")
        logger.info(f"  标签来源: '{self.label_source}' (指向: {self.actual_label_path if self.label_source == 'from_label_path' else self.actual_data_path})")
        logger.info(f"  目标序列长度: {self.target_seq_len}")
        logger.info(f"  加载模态: {self.modalities} (基础通道数: {self.base_channel}, 总输入通道数: {self.num_input_dim})")
        logger.info(f"  时间采样策略 (训练时): {'随机连续片段' if self.random_choose_sampling else '中间连续片段'}")
        logger.info(f"  时间采样策略 (测试/验证时): 均匀采样")
        logger.info(f"  样本数量: {len(self.labels)}")
        if self.debug:
            logger.warning(f"  DEBUG模式开启，只使用前 {len(self.skeletons_data)} 个样本。")


    def load_data(self):
        logger.info(f"开始从 .npz 文件加载数据和标签 ({self.split}集)...")
        
        if not os.path.exists(self.actual_data_path):
            raise FileNotFoundError(f"数据文件未找到: {self.actual_data_path}")

        data_npz = np.load(self.actual_data_path)
        data_key = 'x_train' if self.split == 'train' else 'x_test'
        if data_key not in data_npz:
            raise KeyError(f"数据 .npz 文件 ('{self.actual_data_path}') 中未找到键 '{data_key}'")
        
        raw_data_flat = data_npz[data_key] 
        N_samples, T_orig_from_file, Features_flat = raw_data_flat.shape
        logger.info(f"  原始数据形状 (N, T_orig, Features_flat): ({N_samples}, {T_orig_from_file}, {Features_flat})")


        # 你的预处理脚本 (seq_transformation.py align_frames) 会将数据处理成 (N, T_max, 150)
        # 其中 150 代表 M_max=2, V=25, C=3 (2*25*3 = 150)
        # 所以 M_in_file 应该是 2
        M_in_file = 2 
        expected_features = M_in_file * self.num_nodes * self.base_channel
        if Features_flat != expected_features:
            raise ValueError(f"NPZ中的特征维度 ({Features_flat}) 与预期的 M_in_file={M_in_file} * num_nodes={self.num_nodes} * base_channel={self.base_channel} = {expected_features} 不符。请检查预处理流程或Feeder参数。")
        
        logger.info(f"  根据预处理流程，假定 M_in_file (最大人数) = {M_in_file}.")
        logger.info(f"  原始帧长 T_orig_from_file = {T_orig_from_file}.")


        try:
            # (N, T, 150) -> (N, T, M=2, V=25, C=3)
            self.skeletons_data = raw_data_flat.reshape(N_samples, T_orig_from_file, M_in_file, self.num_nodes, self.base_channel)
            # (N, T, M, V, C) -> (N, M, T, V, C)
            self.skeletons_data = self.skeletons_data.transpose(0, 2, 1, 3, 4)
            logger.debug(f"  Reshaped skeletons_data shape (N, M, T, V, C): {self.skeletons_data.shape}")
        except ValueError as e:
            logger.error(f"Reshape 原始数据失败。原始形状: {raw_data_flat.shape}, 目标 M={M_in_file}, V={self.num_nodes}, C={self.base_channel}. 错误: {e}")
            raise

        # 加载标签数据 (逻辑与之前一致)
        if self.label_source == 'from_data_npz':
            label_npz_source = data_npz
            source_path_for_log = self.actual_data_path
        else: 
            if not os.path.exists(self.actual_label_path):
                raise FileNotFoundError(f"标签文件未找到: {self.actual_label_path}")
            label_npz_source = np.load(self.actual_label_path)
            source_path_for_log = self.actual_label_path
            
        label_key = 'y_train' if self.split == 'train' else 'y_test'
        if label_key not in label_npz_source:
            # 尝试备用键
            if 'arr_1' in label_npz_source: label_key = 'arr_1' # 兼容 np.savez(a,b) 时第二个数组默认名
            elif 'labels' in label_npz_source: label_key = 'labels'
            else: raise KeyError(f"标签源 ('{source_path_for_log}') 中未找到期望的标签键 (尝试过 '{label_key}', 'arr_1', 'labels')")

        raw_labels = label_npz_source[label_key]
        if raw_labels.ndim == 2 and raw_labels.shape[1] > 1: 
            self.labels = np.argmax(raw_labels, axis=1).astype(int)
            logger.info(f"  标签从one-hot编码转换 (源: {source_path_for_log}, 键: {label_key}, 形状: {raw_labels.shape} -> {self.labels.shape})。")
        elif raw_labels.ndim == 1: 
            self.labels = raw_labels.astype(int)
            logger.info(f"  直接加载类别索引标签 (源: {source_path_for_log}, 键: {label_key}, 形状: {self.labels.shape})。")
        else:
            raise ValueError(f"不支持的标签形状: {raw_labels.shape} from {source_path_for_log}")

        if len(self.skeletons_data) != len(self.labels):
            raise ValueError(f"加载的数据样本数 ({len(self.skeletons_data)}) 与标签数 ({len(self.labels)}) 不匹配。")

        if self.debug:
            num_debug_samples = min(100, len(self.labels)) 
            self.skeletons_data = self.skeletons_data[:num_debug_samples]
            self.labels = self.labels[:num_debug_samples]
            logger.info(f"  DEBUG模式: 数据截断为前 {num_debug_samples} 个样本。")
        
        logger.info("数据和标签加载完成。")

    def _temporal_sampling(self, joint_data_tvc, target_len):
        T_orig = joint_data_tvc.shape[0]
        # 获取输入数据的实际通道数，以正确创建空序列
        current_channels = joint_data_tvc.shape[2] if joint_data_tvc.ndim == 3 and joint_data_tvc.size > 0 else self.base_channel

        if T_orig == 0: 
            logger.debug(f"Temporal sampling: Input T_orig is 0, returning empty array of shape (0, {self.num_nodes}, {current_channels})")
            return np.zeros((0, self.num_nodes, current_channels), dtype=np.float32)
        
        if T_orig <= target_len: 
            logger.debug(f"Temporal sampling: T_orig ({T_orig}) <= target_len ({target_len}). No sampling needed. Returning original.")
            return joint_data_tvc
        else: # T_orig > target_len
            if self.random_choose_sampling: # 通常用于训练: 随机选择连续片段
                start_idx = random.randint(0, T_orig - target_len)
                sampled_data = joint_data_tvc[start_idx : start_idx + target_len, :, :]
                logger.debug(f"Temporal sampling (train, random_choose): T_orig={T_orig}, target_len={target_len}. Sampled segment [{start_idx}-{start_idx+target_len-1}]. Output shape: {sampled_data.shape}")
                return sampled_data
            else: # 通常用于验证/测试: 均匀采样帧
                indices = np.linspace(0, T_orig - 1, target_len).round().astype(np.int_)
                indices = np.clip(indices, 0, T_orig - 1) # 确保索引在界内
                # 通常情况下，linspace后round得到的索引数量就是target_len
                # 如果需要严格去重，可以 unique，但可能会导致帧数少于 target_len
                # unique_indices = np.unique(indices)
                # sampled_data = joint_data_tvc[unique_indices, :, :]
                sampled_data = joint_data_tvc[indices, :, :]
                logger.debug(f"Temporal sampling (test/val, uniform): T_orig={T_orig}, target_len={target_len}. Sampled {len(indices)} indices. Output shape: {sampled_data.shape}")
                return sampled_data


    def _random_rotation_augmentation(self, joint_data_tvc, angle_limit_degrees=10):
        if joint_data_tvc.shape[2] != 3: # 确保只对XYZ数据进行旋转
            logger.warning(f"随机旋转期望输入通道为3(XYZ)，但得到{joint_data_tvc.shape[2]}。跳过旋转。")
            return joint_data_tvc
        
        theta_x = np.random.uniform(-angle_limit_degrees, angle_limit_degrees) * (math.pi / 180.0)
        theta_y = np.random.uniform(-angle_limit_degrees, angle_limit_degrees) * (math.pi / 180.0)
        theta_z = np.random.uniform(-angle_limit_degrees, angle_limit_degrees) * (math.pi / 180.0)
        Rx = np.array([[1,0,0],[0,math.cos(theta_x),-math.sin(theta_x)],[0,math.sin(theta_x),math.cos(theta_x)]])
        Ry = np.array([[math.cos(theta_y),0,math.sin(theta_y)],[0,1,0],[-math.sin(theta_y),0,math.cos(theta_y)]])
        Rz = np.array([[math.cos(theta_z),-math.sin(theta_z),0],[math.sin(theta_z),math.cos(theta_z),0],[0,0,1]])
        R_combined = np.dot(Rz, np.dot(Ry, Rx))
        
        original_shape = joint_data_tvc.shape
        joint_data_flat = joint_data_tvc.reshape(-1, 3) # (T*V, 3)
        rotated_data_flat = np.dot(joint_data_flat, R_combined.T) 
        return rotated_data_flat.reshape(original_shape).astype(joint_data_tvc.dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        try:
            label_0based = self.labels[index]
            if not (0 <= label_0based < self.num_classes):
                logger.error(f"样本 {index}: 标签 {label_0based} 超出有效范围 [0, {self.num_classes-1}]。跳过此样本。")
                return None 

            skeletons_for_sample = self.skeletons_data[index] # Shape (M, T_orig, V, C_base)
            
            # 选择主要演员的数据 (T_orig, V, C_base)
            selected_joint_data_tvc = None
            if skeletons_for_sample.shape[0] > 0: # M > 0
                # 尝试第一个人 (M=0)
                person1_data = skeletons_for_sample[0, :, :, :].copy()
                if np.sum(np.abs(person1_data)) > 1e-6 : # 第一个人数据有效
                    selected_joint_data_tvc = person1_data
                elif skeletons_for_sample.shape[0] > 1: # 如果第一个人无效，且存在第二个人 (M=1)
                    person2_data = skeletons_for_sample[1, :, :, :].copy()
                    if np.sum(np.abs(person2_data)) > 1e-6: # 第二个人数据有效
                        selected_joint_data_tvc = person2_data
                        logger.debug(f"样本 {index}: 第一个人数据无效，使用第二个人数据。")
                    else: # 两个人都无效或接近全零
                        selected_joint_data_tvc = person1_data # 使用第一个人的（可能全零）数据
                        logger.warning(f"样本 {index}: 主要演员数据(第一或第二人)均接近全零。")
                else: # 只有一个人，且数据无效
                    selected_joint_data_tvc = person1_data
                    logger.warning(f"样本 {index}: 单人场景但数据接近全零。")
            
            if selected_joint_data_tvc is None: # 如果由于某种原因没有选出数据（例如 M=0）
                logger.warning(f"样本 {index}: 未能选择任何有效的演员骨骼数据 (M={skeletons_for_sample.shape[0]})。将使用全零数据。")
                # T_orig_from_file 来自 load_data, 是 .npz 文件中的原始 T
                # 如果 skeletons_data 本身 T 维度为0 (不太可能，但防御一下)
                t_for_dummy = self.skeletons_data.shape[2] if self.skeletons_data.ndim == 5 and self.skeletons_data.shape[2] > 0 else self.target_seq_len
                selected_joint_data_tvc = np.zeros((t_for_dummy, self.num_nodes, self.base_channel), dtype=np.float32)


            # 1. 中心化 (如果配置了)
            # 注意：预处理脚本 seq_transformation.py 中的 seq_translation 已经做了一次中心化。
            # 如果 Feeder 再次中心化，可能是二次中心化。
            if self.center_joint_idx is not None: # 只有当配置了有效的 center_joint_idx 时执行
                if selected_joint_data_tvc.shape[0] > 0: # 确保有帧数据
                    center_coord = selected_joint_data_tvc[0, self.center_joint_idx, :].copy()
                    selected_joint_data_tvc = selected_joint_data_tvc - center_coord
                    logger.debug(f"样本 {index}: 执行了基于关节 {self.center_joint_idx} 的中心化。")
                else:
                    logger.debug(f"样本 {index}: 跳过中心化，因为选择的演员数据帧数为0。")
            
            # 2. 时间采样
            joint_data_sampled_tvc = self._temporal_sampling(selected_joint_data_tvc, self.target_seq_len)
            # _temporal_sampling 会处理 T_orig=0 的情况，但我们还是检查一下输出
            if joint_data_sampled_tvc.shape[0] == 0 and self.target_seq_len > 0 :
                logger.warning(f"样本 {index}: 时间采样后数据帧数为0，但目标长度为 {self.target_seq_len}。将用零填充。")
                # pad_and_mask_sequence 会处理这种情况，创建一个全零的序列

            # 3. 随机旋转增强 (仅训练时)
            if self.random_rot: 
                if joint_data_sampled_tvc.shape[0] > 0: # 只对有帧的数据旋转
                    joint_data_sampled_tvc = self._random_rotation_augmentation(joint_data_sampled_tvc, angle_limit_degrees=self.rotation_angle_limit)
            
            # --- 模态计算和拼接 ---
            # 此时 joint_data_sampled_tvc 是 (T_sampled, V, C_base)
            # C_base 应该是 self.base_channel (通常是3)
            if joint_data_sampled_tvc.ndim != 3 or \
               joint_data_sampled_tvc.shape[1] != self.num_nodes or \
               joint_data_sampled_tvc.shape[2] != self.base_channel:
                # 除非 T_sampled = 0，这时shape[2]可能是0 (如果从空数组开始)
                if not (joint_data_sampled_tvc.shape[0] == 0 and joint_data_sampled_tvc.shape[2] == self.base_channel): #允许 (0,V,C)
                    logger.error(f"样本 {index}: 采样/增强后的关节数据维度不正确: {joint_data_sampled_tvc.shape} "
                                 f"(期望 T, {self.num_nodes}, {self.base_channel})。跳过。")
                    return None

            modal_data_list_tvc = [] 
            bone_data_xyz_cache = None # 用于缓存 bone 数据，避免重复计算

            for modality_name in self.modalities:
                current_modal_data = None
                if joint_data_sampled_tvc.shape[0] == 0: # 如果没有有效帧
                    current_modal_data = np.zeros((0, self.num_nodes, self.base_channel), dtype=np.float32)
                elif modality_name == 'joint':
                    current_modal_data = joint_data_sampled_tvc.copy()
                elif modality_name == 'bone':
                    # bone 总是从原始3通道joint计算
                    bone_data_xyz_cache = calculate_bone_data_from_joints(joint_data_sampled_tvc, self.bone_pairs, self.num_nodes)
                    current_modal_data = bone_data_xyz_cache
                elif modality_name == 'joint_motion':
                    current_modal_data = calculate_motion_data(joint_data_sampled_tvc)
                elif modality_name == 'bone_motion':
                    if bone_data_xyz_cache is None: # 如果之前没算过 bone
                        bone_data_xyz_cache = calculate_bone_data_from_joints(joint_data_sampled_tvc, self.bone_pairs, self.num_nodes)
                    current_modal_data = calculate_motion_data(bone_data_xyz_cache)
                
                if current_modal_data is not None:
                    modal_data_list_tvc.append(current_modal_data)
                else: # 理论上不应该发生，因为上面处理了空帧
                    logger.error(f"样本 {index}: 模态 '{modality_name}' 计算失败。跳过。")
                    return None
            
            if not modal_data_list_tvc:
                logger.error(f"样本 {index}: 未能生成任何模态数据。跳过。")
                return None
            
            # 拼接 (T_sampled, V, C_base * num_modalities)
            data_concatenated_tvc = np.concatenate(modal_data_list_tvc, axis=-1)
            if data_concatenated_tvc.shape[-1] != self.num_input_dim:
                logger.error(f"样本 {index}: 拼接后的通道数 ({data_concatenated_tvc.shape[-1]}) "
                             f"与预期的总输入维度 ({self.num_input_dim}) 不符。模态: {self.modalities}。跳过。")
                return None

            # 4. 填充到 target_seq_len 并生成掩码
            data_padded_tvc, mask_np = pad_and_mask_sequence(data_concatenated_tvc, 
                                                             self.target_seq_len, 
                                                             self.num_nodes, 
                                                             self.num_input_dim)

            data_tensor = torch.from_numpy(data_padded_tvc).float()
            label_tensor = torch.tensor(label_0based, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool() # (target_seq_len,)

            # +++ 调试：保存一个样本 +++
            # if self.split == 'train' and index < 1 and self.debug is False : # 只保存第一个训练样本一次
            #     debug_save_dir = "./feeder_debug_output"
            #     os.makedirs(debug_save_dir, exist_ok=True)
            #     np.save(os.path.join(debug_save_dir, f"sample_{index}_data_final.npy"), data_padded_tvc)
            #     np.save(os.path.join(debug_save_dir, f"sample_{index}_mask.npy"), mask_np)
            #     with open(os.path.join(debug_save_dir, f"sample_{index}_label.txt"), "w") as f_label:
            #         f_label.write(str(label_0based))
            #     logger.info(f"DEBUG: Saved sample {index} data to {debug_save_dir}")
            # +++ 结束调试 +++

            return data_tensor, label_tensor, mask_tensor, index # 返回原始索引用于追踪

        except Exception as e:
            logger.error(f"处理样本 {index} (标签: {self.labels[index] if index < len(self.labels) else 'N/A'}) 时发生未知错误: {e}", exc_info=True)
            return None # 确保任何意外都返回 None

# --- 用于独立测试 Feeder_NTU 的代码块 (与之前版本类似) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, # <<<--- 设置为 DEBUG 以查看详细日志
                        format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # 创建虚拟 NPZ 文件
    dummy_npz_file = 'dummy_ntu_preprocessed.npz' # 模拟预处理后的文件名
    if not os.path.exists(dummy_npz_file):
        logger.info(f"创建虚拟 .npz 文件: {dummy_npz_file} (模拟预处理后的数据)...")
        _N_train, _N_test = 40, 20 # 样本数
        _T_max = 300 # 预处理后的最大帧长
        _M_max = 2   # 最大人数
        _V_nodes = 25 # 节点数
        _C_joint = 3  # XYZ
        
        # 训练数据 (N, T_max, M_max * V_nodes * C_joint)
        _x_train_flat = np.random.rand(_N_train, _T_max, _M_max * _V_nodes * _C_joint).astype(np.float32)
        # 模拟单人场景：将第二个人的数据置零
        for i in range(_N_train // 2):
            _x_train_flat[i, :, _V_nodes*_C_joint:] = 0 
        # 模拟一个完全无效的样本（例如，在选择演员后数据全零）
        if _N_train > 0: _x_train_flat[0, :, :] = 0 # 第一个样本全零

        _y_train_onehot = np.zeros((_N_train, 60), dtype=np.float32)
        _y_train_idx = np.random.randint(0, 60, size=_N_train)
        if _N_train > 0: _y_train_onehot[np.arange(_N_train), _y_train_idx] = 1.0
        
        _x_test_flat = np.random.rand(_N_test, _T_max, _M_max * _V_nodes * _C_joint).astype(np.float32)
        _y_test_onehot = np.zeros((_N_test, 60), dtype=np.float32)
        _y_test_idx = np.random.randint(0, 60, size=_N_test)
        if _N_test > 0: _y_test_onehot[np.arange(_N_test), _y_test_idx] = 1.0
        
        np.savez(dummy_npz_file, x_train=_x_train_flat, y_train=_y_train_onehot, x_test=_x_test_flat, y_test=_y_test_onehot)
        logger.info(f"虚拟数据文件 {dummy_npz_file} 已创建。")

    DUMMY_ROOT_DIR = '.' 
    DUMMY_DATA_FILENAME = dummy_npz_file 
    
    logger.info("\n--- 测试 Feeder_NTU (使用模拟的预处理后数据) ---")
    try:
        train_feeder_params = {
            'root_dir': DUMMY_ROOT_DIR,      
            'data_path': DUMMY_DATA_FILENAME, 
            'label_path': DUMMY_DATA_FILENAME, # 标签也在同一个文件
            'label_source': 'from_data_npz',  
            'split': 'train',
            'max_len': 64, # <<<--- 测试你的配置文件中的 max_len
            'modalities': "joint,bone", 
            'num_classes': 60,
            'num_nodes': 25,
            'base_channel': 3,
            'random_choose': True, # 训练时用随机片段
            'random_rot': True, 
            'rotation_angle_limit': 15,
            'center_joint_idx': None, # <<<--- 假设预处理已完成中心化，所以这里禁用
            'debug': False
        }
        logger.info(f"训练 Feeder 参数: {train_feeder_params}")
        feeder_train = Feeder_NTU(**train_feeder_params)
        logger.info(f"虚拟训练集 Feeder 长度: {len(feeder_train)}")

        if len(feeder_train) > 0:
            # 测试几个样本
            for i in range(min(5, len(feeder_train))):
                item_result = feeder_train[i] 
                if item_result is not None:
                    data, label, mask, idx = item_result
                    logger.info(f"  训练样本 {idx} (orig_idx {i}): 数据形状 {data.shape}, 标签 {label.item()}, Mask形状 {mask.shape}, Mask有效帧数 {mask.sum().item()}")
                    
                    num_modalities_req = len(train_feeder_params['modalities'].split(','))
                    expected_channels_final = num_modalities_req * train_feeder_params['base_channel']
                    
                    assert data.shape == (train_feeder_params['max_len'], 
                                          train_feeder_params['num_nodes'], 
                                          expected_channels_final), f"Data shape error sample {i}"
                    assert mask.shape == (train_feeder_params['max_len'],), f"Mask shape error sample {i}"
                else:
                    logger.warning(f"  获取的训练样本 {i} 为 None。")
            logger.info("训练样本形状和掩码校验通过 (部分样本)。")

            # 测试 DataLoader
            if 'collate_fn_filter_none' in globals() and callable(globals()['collate_fn_filter_none']):
                collate_fn_to_use_test = globals()['collate_fn_filter_none']
                train_loader = DataLoader(feeder_train, batch_size=4, shuffle=True, num_workers=0, 
                                          collate_fn=collate_fn_to_use_test)
                
                num_batches_tested = 0
                for batch_idx, first_batch in enumerate(train_loader):
                    if batch_idx >= 2 : break # 测试几批
                    if first_batch:
                        xb, lb, mb, idxb = first_batch
                        logger.info(f"  训练 DataLoader 第 {batch_idx+1} 个批次: "
                                    f"X形状 {xb.shape}, L形状 {lb.shape}, M形状 {mb.shape}")
                        num_batches_tested +=1
                    else:
                        logger.warning(f"  训练 DataLoader 第 {batch_idx+1} 个批次为空。")
                if num_batches_tested > 0: logger.info("训练 DataLoader 批次测试通过。")
                else: logger.warning("训练 DataLoader 未能成功生成任何批次。")
            else:
                logger.warning("无法找到 collate_fn_filter_none，跳过 DataLoader 测试。")
        else:
            logger.warning("虚拟训练集为空。")

        logger.info("\n--- 测试 Feeder_NTU (测试集) ---")
        test_feeder_params = train_feeder_params.copy()
        test_feeder_params['split'] = 'test'
        test_feeder_params['random_choose'] = False # 测试时用均匀采样
        test_feeder_params['random_rot'] = False
        # center_joint_idx 保持为 None
        
        logger.info(f"测试 Feeder 参数: {test_feeder_params}")
        feeder_test = Feeder_NTU(**test_feeder_params)
        logger.info(f"虚拟测试集 Feeder 长度: {len(feeder_test)}")
        if len(feeder_test) > 0:
            item_result = feeder_test[0]
            if item_result is not None:
                 data, label, mask, idx = item_result
                 logger.info(f"  测试样本 {idx}: 数据形状 {data.shape}, 标签 {label.item()}, Mask有效帧数 {mask.sum().item()}")
            else:
                 logger.warning("  获取的第一个测试样本为 None。")
    except Exception as e:
        logger.error(f"测试 Feeder_NTU 时出错: {e}", exc_info=True)