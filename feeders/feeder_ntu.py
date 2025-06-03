# -*- coding: utf-8 -*-
# 文件名: feeders/feeder_ntu.py (v2.4.1 - FixedHelperFunctionsAndBonePairs)
# 描述: 集成类Hyperformer的处理逻辑到Feeder中，输出 (T,V,C) 格式。
#       修复了辅助函数未定义的问题，并更新了骨骼对。

import os
import numpy as np
# import pickle # 如果不从单独的pkl加载标签，可能不需要
import random
import math
import logging
import sys
import torch
from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm # 如果load_data中没有tqdm，可以移除

# 导入 tools.py 中的函数
from . import tools

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

# --- 修改后的 NTU 骨骼连接对 ---
ntu_pairs = ( # 使用你提供的骨骼对定义
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)
# 将其转换为模块级别的变量，以便在类外部的辅助函数中使用
NTU_BONE_PAIRS_FROM_USER = ntu_pairs

def pad_and_mask_sequence(sequence_data_tvc, target_len, num_nodes, num_final_channels, pad_value=0.0):
    # ... (这个函数保持不变) ...
    if sequence_data_tvc is None or sequence_data_tvc.size == 0:
        padded_sequence = np.full((target_len, num_nodes, num_final_channels), pad_value, dtype=np.float32)
        mask = np.zeros(target_len, dtype=bool)
        return padded_sequence, mask
    current_len = sequence_data_tvc.shape[0]
    if current_len == 0:
        padded_sequence = np.full((target_len, num_nodes, num_final_channels), pad_value, dtype=np.float32)
        mask = np.zeros(target_len, dtype=bool)
        return padded_sequence, mask
    if sequence_data_tvc.ndim != 3 or sequence_data_tvc.shape[1] != num_nodes or sequence_data_tvc.shape[2] != num_final_channels:
         logger.error(f"Pad input seq shape {sequence_data_tvc.shape} mismatch expected (T, {num_nodes}, {num_final_channels}). Returning zeros.")
         padded_sequence = np.full((target_len, num_nodes, num_final_channels), pad_value, dtype=np.float32)
         mask = np.zeros(target_len, dtype=bool)
         return padded_sequence, mask
    if current_len < target_len:
        pad_width = target_len - current_len
        padding_shape = (pad_width, num_nodes, num_final_channels)
        padding_array = np.full(padding_shape, pad_value, dtype=sequence_data_tvc.dtype)
        padded_sequence = np.concatenate((sequence_data_tvc, padding_array), axis=0)
        mask = np.concatenate((np.ones(current_len, dtype=bool), np.zeros(pad_width, dtype=bool)))
    elif current_len > target_len:
        padded_sequence = sequence_data_tvc[:target_len, :, :]
        mask = np.ones(target_len, dtype=bool)
    else:
        padded_sequence = sequence_data_tvc
        mask = np.ones(target_len, dtype=bool)
    return padded_sequence, mask

# +++ 新增/修复的辅助函数定义 +++
def calculate_bone_data_with_spine_preservation(joint_data_tvc, bone_pairs, num_nodes, spine_center_idx=20):
    """计算骨骼数据，并保留指定spine center的原始关节轨迹。"""
    T, V, C = joint_data_tvc.shape
    if C != 3: raise ValueError(f"计算骨骼数据时期望关节数据有3个通道 (xyz)，但得到 {C} 个。")
    if V != num_nodes: raise ValueError(f"关节数据中的节点数 {V} 与期望的节点数 {num_nodes} 不符。")

    bone_data_tvc = np.zeros_like(joint_data_tvc)
    for v1_one_based, v2_one_based in bone_pairs: # bone_pairs 是1-based
        v1_zero_based = v1_one_based - 1
        v2_zero_based = v2_one_based - 1
        if 0 <= v1_zero_based < num_nodes and 0 <= v2_zero_based < num_nodes:
            bone_data_tvc[:, v1_zero_based, :] = joint_data_tvc[:, v1_zero_based, :] - joint_data_tvc[:, v2_zero_based, :]
        # else: # 可以选择性地记录无效的骨骼对
            # logger.debug(f"骨骼对 ({v1_one_based}, {v2_one_based}) 中的索引无效或被跳过。")

    # 特殊处理：保留spine center的原始关节轨迹
    if 0 <= spine_center_idx < num_nodes:
        bone_data_tvc[:, spine_center_idx, :] = joint_data_tvc[:, spine_center_idx, :].copy()
    else:
        logger.warning(f"提供的spine_center_idx ({spine_center_idx}) 无效，无法在骨骼数据中保留其轨迹。")
    return bone_data_tvc

def calculate_motion_with_zero_last_frame(data_tvc):
    """计算运动数据（帧间差分），最后一帧的运动设为0。"""
    motion_data_tvc = np.zeros_like(data_tvc)
    T = data_tvc.shape[0]
    if T > 1:
        motion_data_tvc[:-1, :, :] = data_tvc[1:, :, :] - data_tvc[:-1, :, :]
        motion_data_tvc[-1, :, :] = 0 # 最后一帧运动为0
    # 如果 T <= 1, motion_data_tvc 保持为全零，这是合理的
    return motion_data_tvc

def center_joint_data_relative_to_spine(joint_data_tvc, spine_center_idx=20):
    """将所有关节相对于当前帧的spine center进行表示，但保留spine center本身的绝对轨迹。"""
    T, V, C = joint_data_tvc.shape
    if not (0 <= spine_center_idx < V):
        logger.warning(f"提供的spine_center_idx ({spine_center_idx}) 无效，跳过特定关节中心化。")
        return joint_data_tvc
    if T == 0: # 如果没有帧数据，直接返回
        return joint_data_tvc

    centered_data_tvc = np.array(joint_data_tvc, copy=True)
    spine_center_trajectory = centered_data_tvc[:, spine_center_idx:spine_center_idx+1, :].copy() # (T, 1, C)
    centered_data_tvc = centered_data_tvc - spine_center_trajectory # 广播减法
    centered_data_tvc[:, spine_center_idx, :] = spine_center_trajectory.squeeze(axis=1) # 将绝对轨迹放回
    return centered_data_tvc
# +++ 辅助函数定义结束 +++


class Feeder_NTU(Dataset):
    def __init__(self,
                 root_dir, # <--- 新增 root_dir 参数
                 data_path,      # 现在是相对于 root_dir 的文件名
                 label_path=None,# 现在是相对于 root_dir 的文件名，或者与 data_path 相同
                 p_interval=None,
                 split='train', random_choose=False,
                 random_rot=False, window_size=-1,
                 normalization=False, debug=False,
                 bone=False, motion=False,
                 num_nodes=25, base_channel=3, num_classes=60,
                 use_relative_joint_centering: bool = True,
                 spine_center_joint_idx: int = 20,
                 **kwargs):
        super().__init__()

        self.debug = debug
        # --- 修改路径处理逻辑 ---
        if root_dir is None:
            # 如果 root_dir 未提供，则假定 data_path 和 label_path 是完整路径
            self.data_path = data_path
            self.label_path = label_path if label_path is not None else data_path # 如果label_path未提供，则默认为data_path
        else:
            self.data_path = os.path.join(root_dir, data_path)
            if label_path is None or label_path == data_path: # 如果label_path未提供或与data_path相同
                self.label_path = self.data_path
            else: # 如果label_path是不同的文件名
                self.label_path = os.path.join(root_dir, label_path)
        # --- 路径处理逻辑结束 ---

        self.split = split
        self.p_interval = p_interval if p_interval is not None else ([1.0] if split != 'train' else [0.5, 1.0])
        self.window_size = window_size if window_size > 0 else 300
        self.random_choose_for_crop = random_choose
        self.random_rot = random_rot if self.split == 'train' else False
        self.normalization = normalization
        self.bone = bone
        self.motion = motion
        self.num_nodes = num_nodes
        self.base_channel = base_channel
        self.num_classes = num_classes
        self.num_input_dim = self.base_channel
        self.use_relative_joint_centering = use_relative_joint_centering
        self.spine_center_joint_idx = spine_center_joint_idx
        self.bone_pairs = NTU_BONE_PAIRS_FROM_USER

        self.load_data()
        if normalization:
            self.get_mean_map()

        logger.info(f"Feeder_NTU ({self.split}集) 初始化完成 (v2.4.2-PathFixInInit):")
        logger.info(f"  实际数据文件路径: {self.data_path}") # 显示拼接后的路径
        logger.info(f"  实际标签文件路径: {self.label_path}") # 显示拼接后的路径
        logger.info(f"  输出骨骼模态: {self.bone}")
        logger.info(f"  输出运动模态: {self.motion}")
        # ... (其他日志保持不变) ...
        logger.info(f"  目标窗口大小 (window_size): {self.window_size}")
        logger.info(f"  裁剪比例 p_interval: {self.p_interval}")
        logger.info(f"  是否使用相对于Spine Center的关节中心化: {self.use_relative_joint_centering}")
        logger.info(f"  是否进行基于数据集的标准化: {self.normalization}")
        logger.info(f"  样本数量: {len(self.label)}")


    def load_data(self):
        logger.info(f"开始从 .npz 文件加载数据和标签 ({self.split}集) - Integrated Style...")
        # 使用 self.data_path 和 self.label_path (它们已经是完整路径了)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")
        npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            # 标签加载逻辑：如果label_path与data_path相同，则从npz_data取；否则从单独的label_npz取
            if self.label_path == self.data_path:
                label_source_npz = npz_data
            else:
                if not os.path.exists(self.label_path):
                    raise FileNotFoundError(f"标签文件未找到: {self.label_path}")
                label_source_npz = np.load(self.label_path)
            self.label = np.where(label_source_npz['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test' or self.split == 'val':
            self.data = npz_data['x_test']
            if self.label_path == self.data_path:
                label_source_npz = npz_data
            else:
                if not os.path.exists(self.label_path):
                    raise FileNotFoundError(f"标签文件未找到: {self.label_path}")
                label_source_npz = np.load(self.label_path)
            self.label = np.where(label_source_npz['y_test'] > 0)[1]
            self.sample_name = [f'{self.split}_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test/val')

        N, T_orig_from_file, Features_flat = self.data.shape
        M_in_file = 2
        expected_features = M_in_file * self.num_nodes * self.base_channel
        if Features_flat != expected_features:
            raise ValueError(f"NPZ中的特征维度 ({Features_flat}) 与预期的 M={M_in_file}*V={self.num_nodes}*C={self.base_channel}={expected_features} 不符。")
        self.data = self.data.reshape((N, T_orig_from_file, M_in_file, self.num_nodes, self.base_channel)).transpose(0, 4, 1, 3, 2)
        logger.info(f"  内部存储数据形状 self.data (N, C, T, V, M): {self.data.shape}")
        if self.debug:
            num_debug_samples = min(100, len(self.label))
            self.data = self.data[:num_debug_samples]; self.label = self.label[:num_debug_samples]; self.sample_name = self.sample_name[:num_debug_samples]
        logger.info("数据和标签加载完成 (Integrated Style)。")

    def get_mean_map(self):
        # ... (get_mean_map 方法保持不变) ...
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        logger.info("数据集的均值和标准差图已计算。")

    def __len__(self):
        return len(self.label)

    def __iter__(self): # pragma: no cover
        return self

    def __getitem__(self, index):
        try:
            label_0based = self.label[index]
            if not (0 <= label_0based < self.num_classes):
                logger.error(f"样本 {index}: 标签 {label_0based} 超出有效范围。跳过。")
                return None

            data_ctvm = self.data[index].copy()
            C_raw, T_orig, V_raw, M_raw = data_ctvm.shape

            selected_actor_data_ctv = None
            if M_raw > 0:
                selected_actor_data_ctv = data_ctvm[:, :, :, 0]
                if np.sum(np.abs(selected_actor_data_ctv)) < 1e-6 and M_raw > 1:
                    person2_data_ctv = data_ctvm[:, :, :, 1]
                    if np.sum(np.abs(person2_data_ctv)) > 1e-6:
                        selected_actor_data_ctv = person2_data_ctv
            if selected_actor_data_ctv is None:
                 logger.warning(f"样本 {index}: 没有有效的演员数据。返回全零。")
                 data_padded_tvc = np.zeros((self.window_size, self.num_nodes, self.base_channel), dtype=np.float32)
                 mask_np = np.zeros(self.window_size, dtype=bool)
                 return torch.from_numpy(data_padded_tvc).float(), torch.tensor(label_0based, dtype=torch.long), torch.from_numpy(mask_np).bool(), index

            valid_frame_num = np.sum(selected_actor_data_ctv.sum(0).sum(-1) != 0)
            if valid_frame_num == 0:
                logger.warning(f"样本 {index}: 选择的演员没有有效帧。返回全零。")
                data_padded_tvc = np.zeros((self.window_size, self.num_nodes, self.base_channel), dtype=np.float32)
                mask_np = np.zeros(self.window_size, dtype=bool)
                return torch.from_numpy(data_padded_tvc).float(), torch.tensor(label_0based, dtype=torch.long), torch.from_numpy(mask_np).bool(), index

            data_for_crop_ctvm = selected_actor_data_ctv[:, :valid_frame_num, :, np.newaxis]
            current_p_interval = self.p_interval
            if self.split == 'train' and self.random_choose_for_crop and isinstance(self.p_interval, list) and len(self.p_interval) == 2:
                pass
            elif isinstance(self.p_interval, list) and len(self.p_interval) == 1:
                current_p_interval = self.p_interval
            elif isinstance(self.p_interval, (int, float)):
                 current_p_interval = [self.p_interval]
            else:
                if isinstance(self.p_interval, list) and len(self.p_interval) == 2:
                    current_p_interval = [self.p_interval[1]]

            data_after_tools_ctvm = tools.valid_crop_resize(
                data_for_crop_ctvm, data_for_crop_ctvm.shape[1],
                current_p_interval, self.window_size)

            if self.random_rot:
                data_after_tools_ctvm = tools.random_rot(data_after_tools_ctvm)

            data_tvc_for_modalities = data_after_tools_ctvm.squeeze(axis=-1).transpose(1, 2, 0)

            if self.normalization and hasattr(self, 'mean_map') and hasattr(self, 'std_map'):
                data_to_norm_ctv = data_tvc_for_modalities.transpose(2,0,1)
                mean_for_sub = self.mean_map.squeeze(-1)
                std_for_div = self.std_map.squeeze(-1)
                std_for_div[std_for_div == 0] = 1e-6
                data_normed_ctv = (data_to_norm_ctv - mean_for_sub) / std_for_div
                data_tvc_for_modalities = data_normed_ctv.transpose(1,2,0)

            current_data_for_modality = data_tvc_for_modalities.copy()
            final_output_data = None # 初始化

            if self.bone:
                final_output_data = calculate_bone_data_with_spine_preservation(
                    current_data_for_modality, self.bone_pairs,
                    self.num_nodes, spine_center_idx=self.spine_center_joint_idx)
            elif self.use_relative_joint_centering:
                final_output_data = center_joint_data_relative_to_spine(
                    current_data_for_modality, spine_center_idx=self.spine_center_joint_idx)
            else:
                final_output_data = current_data_for_modality

            if self.motion:
                final_output_data = calculate_motion_with_zero_last_frame(final_output_data)

            data_padded_tvc, mask_np = pad_and_mask_sequence(final_output_data,
                                                             self.window_size,
                                                             self.num_nodes,
                                                             self.base_channel)

            data_tensor = torch.from_numpy(data_padded_tvc).float()
            label_tensor = torch.tensor(label_0based, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool()

            return data_tensor, label_tensor, mask_tensor, index

        except Exception as e:
            logger.error(f"处理样本 {index} (标签: {self.label[index] if index < len(self.label) else 'N/A'}) 时发生未知错误: {e}", exc_info=True)
            return None
