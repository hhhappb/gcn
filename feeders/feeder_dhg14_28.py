v# 文件名: feeders/feeder_dhg14_28.py (v2.0 - 完全修正LOSO加载逻辑)

import json
import os
import sys
import numpy as np
import random
import math
import pickle
import logging
import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import re
import traceback

# --- 日志记录器设置 ---
logger = logging.getLogger(__name__)
# 如果直接运行此文件进行测试，则需要配置 basicConfig
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, # <--- 可以设为 DEBUG 看更详细信息
                        format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')

# --- 辅助函数：pad_sequence_with_mask (保持不变) ---
def pad_sequence_with_mask(seq, max_len, num_nodes, num_channels, pad_value=0.0):
    if seq is None or seq.size == 0:
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask
    seq_len = seq.shape[0]
    if seq_len == 0:
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask
    if seq.ndim != 3 or seq.shape[1] != num_nodes:
         logger.warning(f"Pad input seq shape {seq.shape} mismatch expected node count {num_nodes}. Returning zeros.")
         padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
         mask = np.zeros(max_len, dtype=bool)
         return padded_seq, mask
    current_channels = seq.shape[2]
    if current_channels != num_channels:
        logger.warning(f"Pad input seq channel {current_channels} mismatch expected total channels {num_channels}. Returning zeros.")
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask
    if seq_len < max_len:
        pad_len = max_len - seq_len
        padding = np.full((pad_len, num_nodes, num_channels), pad_value, dtype=seq.dtype)
        padded_seq = np.concatenate([seq, padding], axis=0)
        mask = np.concatenate([np.ones(seq_len, dtype=bool), np.zeros(pad_len, dtype=bool)], axis=0)
    elif seq_len > max_len:
        padded_seq = seq[:max_len, :, :]
        mask = np.ones(max_len, dtype=bool)
    else:
        padded_seq = seq
        mask = np.ones(max_len, dtype=bool)
    return padded_seq, mask


class Feeder(Dataset):
    """
    用于 DHG14-28 数据集的 Feeder 类 (v2.0 - 完全修正 LOSO 加载逻辑)。
    从每个 subject 的文件夹读取对应的 samples.json 列表，
    并根据列表内容加载该 subject 下 train/val 子目录中的数据文件。
    """
    def __init__(self, root_dir, subject_idx, split,
                 label_type='label_28',
                 max_len=150,
                 data_path='joint',
                 base_channel=3,
                 repeat=1,
                 random_choose=False,
                 apply_rand_view_transform=True,
                 apply_random_shear=False, shear_amplitude=0.5,
                 apply_random_flip=False,
                 apply_coord_drop=False, coord_drop_prob=0.1, coord_drop_axis=None,
                 apply_joint_drop=False, joint_drop_prob=0.1,
                 center_joint_idx=0,
                 debug=False,
                 **kwargs):

        super().__init__()

        # --- 参数校验和基础设置 ---
        if not os.path.isdir(root_dir): raise ValueError(f"无效的数据集根目录: {root_dir}")
        if split not in ['train', 'val', 'test']: raise ValueError(f"无效的 split: {split}")
        if not (1 <= subject_idx <= 20): raise ValueError(f"无效的 subject_idx: {subject_idx}, 应在 1 到 20 之间")

        self.root_dir = root_dir # 指向 DHG14-28_sample_json 目录
        self.subject_idx_eval = subject_idx # 验证/测试时使用的 subject ID
        self.split = split # 'train', 'val', or 'test'
        self.label_type = label_type
        self.max_len = max_len
        self.repeat = repeat if self.split == 'train' else 1
        self.random_choose = random_choose if self.split == 'train' else False
        if center_joint_idx is not None and not (0 <= center_joint_idx < 22):
            logger.warning(f"提供的 center_joint_idx ({center_joint_idx}) 无效，将禁用中心化。")
            self.center_joint_idx = None
        else:
            self.center_joint_idx = center_joint_idx
        self.debug = debug

        # --- DHG 特定参数 ---
        self.num_nodes = 22
        self.base_channel = base_channel
        self.bone_pairs = [(0, 1), (2, 0), (3, 2), (4, 3), (5, 4), (6, 1), (7, 6), (8, 7), (9, 8), (10, 1), (11, 10),
                           (12, 11), (13, 12), (14, 1), (15, 14), (16, 15), (17, 16), (18, 1), (19, 18), (20, 19), (21, 20)]
        self.left_parts = []
        self.right_parts = []

        # --- 解析多模态 ---
        if isinstance(data_path, str): self.modalities = [m.strip().lower() for m in data_path.split(',') if m.strip()]
        elif isinstance(data_path, list): self.modalities = [m.strip().lower() for m in data_path if isinstance(m, str) and m.strip()]
        else: raise ValueError("data_path 必须是逗号分隔的字符串或字符串列表")
        valid_modalities = ['joint', 'bone', 'joint_motion', 'bone_motion']
        for m in self.modalities:
            if m not in valid_modalities: raise ValueError(f"不支持的数据模态: '{m}'. 支持: {valid_modalities}")
        if not self.modalities: raise ValueError("必须至少指定一种数据模态 (data_path)")
        self.num_input_dim = self.base_channel * len(self.modalities)

        # --- 保存增强参数 ---
        self.apply_rand_view_transform = apply_rand_view_transform if self.split == 'train' else False
        self.apply_random_shear = apply_random_shear if self.split == 'train' else False
        self.shear_amplitude = shear_amplitude
        self.apply_random_flip = apply_random_flip if self.split == 'train' else False
        self.apply_coord_drop = apply_coord_drop if self.split == 'train' else False
        self.coord_drop_prob = coord_drop_prob
        self.coord_drop_axis = coord_drop_axis
        self.apply_joint_drop = apply_joint_drop if self.split == 'train' else False
        self.joint_drop_prob = joint_drop_prob

        # --- 打印初始化信息 ---
        split_desc = f"TRAINING (leaving Subject {self.subject_idx_eval} out)" if self.split == 'train' else f"VALIDATION/TESTING on Subject {self.subject_idx_eval}"
        logger.info(f"初始化 Feeder for DHG14-28 ({split_desc}): root={self.root_dir}")
        logger.info(f"标签类型: {self.label_type}, 目标序列长度: {max_len}")
        logger.info(f"加载模态: {self.modalities}, 总输入维度: {self.num_input_dim}")
        if self.center_joint_idx is not None: logger.info(f"中心化关节索引: {self.center_joint_idx}")
        else: logger.info("中心化已禁用。")
        if self.split == 'train':
            logger.info(f"训练时增强: RandViewTransform={self.apply_rand_view_transform}, RandShear={self.apply_random_shear}, RandFlip={self.apply_random_flip}, CoordDrop={self.apply_coord_drop}, JointDrop={self.apply_joint_drop}")
        if self.apply_random_flip:
             logger.warning("!!! RandomFlip 已启用，但 DHG 的左右映射 (left/right_parts) 未定义，可能效果不佳或无效 !!!")

        # --- 加载数据 ---
        self.sample_info = [] # 存储 {'id': sample_id_base, 'path': json_path}
        self.label = []       # 存储 0-based 标签
        self.data = []        # 列表存储加载的 numpy 数据

        self._load_data_loso_corrected() # <<<--- 调用修正后的加载函数 v2.0

        # --- 加载后检查 ---
        if not self.sample_info: raise RuntimeError(f"未能加载任何有效的样本信息 for split '{split}' (Subject {self.subject_idx_eval} {'out' if split=='train' else 'as test'})")
        if len(self.sample_info) != len(self.label) or len(self.sample_info) != len(self.data):
             raise RuntimeError(f"样本信息 ({len(self.sample_info)}), 标签 ({len(self.label)}), 数据 ({len(self.data)}) 数量不匹配！ Split: {split}, Subject: {self.subject_idx_eval}")

        # --- 推断类别数 ---
        self.num_classes = kwargs.get('num_classes', 0)
        if self.num_classes == 0 and self.label:
            valid_labels = [l for l in self.label if l != -1]
            if valid_labels: self.num_classes = max(valid_labels) + 1
        if self.num_classes == 0:
             self.num_classes = 14 if '14' in self.label_type else 28
             logger.warning(f"无法从标签列表推断类别数，根据 label_type 猜测为 {self.num_classes}")
        logger.info(f"使用的类别数: {self.num_classes}")

        # --- Debug 模式截断 ---
        if self.debug:
             limit = min(100, len(self.sample_info))
             logger.warning(f"!!! DEBUG 模式开启，只使用前 {limit} 个样本 !!!")
             self.sample_info=self.sample_info[:limit]
             self.label=self.label[:limit]
             self.data=self.data[:limit]

        logger.info(f"成功加载 {len(self.sample_info)} 个有效样本用于 '{split}' split。")


    def _parse_subject_from_filename(self, filename):
        """ 从 gXXfXXsXXeXX 格式的文件名中解析出 subject ID (整数) """
        match = re.search(r's(\d+)', filename)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _load_data_loso_corrected(self):
        """根据 LOSO 结构加载数据列表和数据 (v2.0 - 完全修正逻辑)"""
        temp_sample_info = []
        temp_label_list = []
        temp_data_list = []

        # 确定要加载哪些 subject 的数据
        if self.split == 'train':
            subjects_to_load = [s for s in range(1, 21) if s != self.subject_idx_eval]
            list_suffix = "train_samples.json"
            data_subdir = "train"
            logger.info(f"加载训练数据，使用 Subjects: {subjects_to_load}")
        else: # 'val' or 'test'
            subjects_to_load = [self.subject_idx_eval]
            list_suffix = "val_samples.json"
            data_subdir = "val"
            logger.info(f"加载验证/测试数据，使用 Subject: {subjects_to_load}")

        total_loaded_count = 0
        total_skipped_label = 0
        total_skipped_data = 0
        total_invalid_label_value = 0

        # 遍历需要加载的 subject
        for current_subject_id in subjects_to_load:
            subject_dir = os.path.join(self.root_dir, str(current_subject_id))
            list_file_path = os.path.join(subject_dir, f"{current_subject_id}{list_suffix}")
            json_data_dir = os.path.join(subject_dir, data_subdir)

            # 检查列表文件和数据目录是否存在
            if not os.path.exists(list_file_path):
                logger.warning(f"List file not found for Subject {current_subject_id}: {list_file_path}, skipping this subject.")
                continue
            if not os.path.isdir(json_data_dir):
                logger.warning(f"Data directory not found for Subject {current_subject_id}: {json_data_dir}, skipping this subject.")
                continue

            logger.debug(f"Processing Subject {current_subject_id}, List: {list_file_path}, Data Dir: {json_data_dir}")

            # 加载当前 subject 的样本列表
            try:
                with open(list_file_path, 'r', encoding='utf-8') as f:
                    sample_list_raw = json.load(f)
            except Exception as e:
                logger.warning(f"Cannot load/parse list file {list_file_path}: {e}, skipping this subject.")
                continue

            # 遍历当前 subject 的样本列表
            subject_loaded_count = 0
            subject_skipped_label = 0
            subject_skipped_data = 0
            subject_invalid_label = 0

            for info in tqdm(sample_list_raw, desc=f"S{current_subject_id} {data_subdir}", leave=False):
                sample_id_base = info.get('file_name')
                if not sample_id_base:
                    subject_skipped_data += 1 # 列表条目本身有问题
                    continue

                # 构造实际数据文件路径
                json_path = os.path.join(json_data_dir, sample_id_base + '.json')

                # 检查文件是否存在
                if not os.path.exists(json_path):
                    # logger.debug(f"JSON file not found for {sample_id_base} at: {json_path}")
                    subject_skipped_data += 1
                    continue

                # 处理标签
                final_label = -1
                try:
                    label_val = int(info[self.label_type])
                    expected_max_label = 14 if '14' in self.label_type else 28
                    if not (1 <= label_val <= expected_max_label):
                         raise ValueError("Label value out of range")
                    final_label = label_val - 1
                except (KeyError, ValueError, TypeError) as e:
                     subject_skipped_label += 1
                     if final_label != -1: subject_invalid_label += 1
                     # logger.debug(f"Label invalid/missing for {sample_id_base}")
                     continue # 跳过标签无效的样本

                # 读取数据 JSON
                data_numpy = self._read_dhg_json(json_path, sample_id_base)

                if data_numpy is not None:
                    # 使用一个唯一ID，结合 subject 和 文件名，避免潜在冲突（虽然此场景下可能非必须）
                    # unique_id = f"subj{current_subject_id}_{sample_id_base}"
                    temp_sample_info.append({'id': sample_id_base, 'path': json_path}) # 存储基础 ID
                    temp_label_list.append(final_label)
                    temp_data_list.append(data_numpy)
                    subject_loaded_count += 1
                else:
                    subject_skipped_data += 1
                    # logger.debug(f"Failed to read/process JSON: {json_path}")

            logger.debug(f"Subject {current_subject_id} Summary: Loaded={subject_loaded_count}, SkippedLabel={subject_skipped_label} (InvalidVal={subject_invalid_label}), SkippedData={subject_skipped_data}")
            total_loaded_count += subject_loaded_count
            total_skipped_label += subject_skipped_label
            total_skipped_data += subject_skipped_data
            total_invalid_label_value += subject_invalid_label

        logger.info(f"Overall Data Loading Summary: Loaded={total_loaded_count}, SkippedLabel={total_skipped_label} (InvalidVal={total_invalid_label_value}), SkippedData={total_skipped_data}")

        # 更新实例变量
        self.sample_info = temp_sample_info
        self.label = temp_label_list
        self.data = temp_data_list


    def _read_dhg_json(self, filepath, sample_id_for_debug):
        """读取单个 DHG JSON 文件并提取骨骼数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f: json_data = json.load(f)
            skeletons_data = json_data.get('skeletons')
            if not skeletons_data or not isinstance(skeletons_data, list):
                if isinstance(json_data, list): skeletons_data = json_data
                else: return None
            if not skeletons_data: return None

            sequence = []
            zero_frame = np.zeros((self.num_nodes, self.base_channel), dtype=np.float32)
            for frame_idx, frame_joints_coords in enumerate(skeletons_data):
                if not isinstance(frame_joints_coords, list) or len(frame_joints_coords) == 0:
                    sequence.append(zero_frame); continue

                current_frame_joints_list = frame_joints_coords
                original_node_count = len(current_frame_joints_list)
                if original_node_count != self.num_nodes:
                    if original_node_count < self.num_nodes:
                        padding = [[0.0] * self.base_channel] * (self.num_nodes - original_node_count)
                        current_frame_joints_list.extend(padding)
                    else:
                        current_frame_joints_list = current_frame_joints_list[:self.num_nodes]

                frame_data = []
                for joint_idx, joint_coords in enumerate(current_frame_joints_list):
                    if isinstance(joint_coords, list) and len(joint_coords) >= self.base_channel and all(isinstance(c, (int, float)) and np.isfinite(c) for c in joint_coords[:self.base_channel]):
                        frame_data.append(joint_coords[:self.base_channel])
                    else: frame_data.append([0.0] * self.base_channel)
                sequence.append(np.array(frame_data, dtype=np.float32))

            if not sequence: return None
            data_numpy = np.stack(sequence, axis=0)
            raw_has_values = any(np.any(f) for f in skeletons_data if isinstance(f,list) and f for joint in f if isinstance(joint, list))
            if not np.any(data_numpy) and raw_has_values:
                 logger.warning(f"Processed sequence for {filepath} is all zeros, but original data had values. Returning None.")
                 return None
            return data_numpy
        except FileNotFoundError: return None
        except json.JSONDecodeError: logger.warning(f"JSON decode error: {filepath}"); return None
        except Exception as e: logger.warning(f"Error processing file {filepath}: {e}"); return None

    # --- 数据增强函数 ---
    # (保持不变)
    def rand_view_transform(self, X, agx=None, agy=None, s=None):
        if X.shape[-1] != 3: return X
        if agx is None: agx = random.randint(-60, 60);
        if agy is None: agy = random.randint(-60, 60);
        if s is None: s = random.uniform(0.7, 1.3);
        agx_rad = math.radians(agx); agy_rad = math.radians(agy);
        Rx = np.array([[1,0,0], [0,math.cos(agx_rad),math.sin(agx_rad)], [0,-math.sin(agx_rad),math.cos(agx_rad)]], dtype=X.dtype);
        Ry = np.array([[math.cos(agy_rad),0,-math.sin(agy_rad)], [0,1,0], [math.sin(agy_rad),0,math.cos(agy_rad)]], dtype=X.dtype);
        Ss = np.array([[s,0,0],[0,s,0],[0,0,s]], dtype=X.dtype);
        orig_shape = X.shape;
        X_transformed = np.dot(X.reshape(-1, 3), Ry @ Rx @ Ss);
        return X_transformed.reshape(orig_shape)
    def random_shear(self, X, amplitude=0.5):
        T, N, C = X.shape;
        if C != 3: return X;
        shear_vals = np.random.uniform(-amplitude, amplitude, size=(3, 3));
        shear_matrix = np.eye(3, dtype=X.dtype);
        axis_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
        pair_idx = random.choice(range(len(axis_pairs)));
        shear_matrix[axis_pairs[pair_idx]] = shear_vals[axis_pairs[pair_idx]];
        X_transformed = np.dot(X.reshape(-1, 3), shear_matrix);
        return X_transformed.reshape(T, N, C)
    def random_flip(self, X):
        if not self.left_parts or not self.right_parts: return X
        T, N, C = X.shape
        if C != 3: return X
        flipped_X = X.copy()
        flipped_X[..., 0] *= -1
        for l, r in zip(self.left_parts, self.right_parts):
            if 0 <= l < N and 0 <= r < N:
                temp_l_data = flipped_X[:, l, :].copy()
                flipped_X[:, l, :] = flipped_X[:, r, :]
                flipped_X[:, r, :] = temp_l_data
        return flipped_X
    def random_coordinate_dropout(self, X, prob=0.1, axis=None):
        T, N, C = X.shape;
        if C != 3: return X;
        if random.random() < prob:
            axis_to_drop = axis if axis in [0, 1, 2] else random.randint(0, C - 1);
            X[..., axis_to_drop] = 0;
        return X
    def random_joint_dropout(self, X, prob=0.1):
        T, N, C = X.shape;
        if random.random() < prob:
            num_joints_to_drop = random.randint(1, N // 4 + 1);
            joints_to_drop = random.sample(range(N), num_joints_to_drop);
            X[:, joints_to_drop, :] = 0;
        return X

    # --- 时间采样和衍生模态计算 ---
    def temporal_sampling(self, data_numpy):
        T_orig = data_numpy.shape[0]
        if T_orig == 0: return np.zeros((self.max_len, self.num_nodes, data_numpy.shape[-1]), dtype=data_numpy.dtype)
        if T_orig <= self.max_len: return data_numpy
        if self.random_choose:
             indices = random.sample(range(T_orig), self.max_len); indices.sort();
             return data_numpy[indices, :, :]
        else:
             indices = np.linspace(0, T_orig - 1, self.max_len).round().astype(int);
             indices = np.clip(indices, 0, T_orig - 1);
             return data_numpy[indices, :, :]
    def _get_bone_data(self, joint_data):
        bone_data = np.zeros_like(joint_data);
        for v1, v2 in self.bone_pairs:
             if 0 <= v1 < self.num_nodes and 0 <= v2 < self.num_nodes:
                  bone_data[:, v1, :] = joint_data[:, v1, :] - joint_data[:, v2, :]
        return bone_data
    def _get_motion_data(self, data):
        motion_data = np.zeros_like(data); T = data.shape[0];
        if T > 1: motion_data[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]; motion_data[-1, :, :] = motion_data[-2, :, :];
        return motion_data

    def __len__(self):
        return len(self.sample_info) * self.repeat

    def __getitem__(self, index):
        true_index = index % len(self.sample_info)

        # --- 获取预加载的数据和标签 ---
        label = self.label[true_index]
        joint_data_orig = self.data[true_index]
        sample_id = self.sample_info[true_index]['id'] # 使用基础 ID

        # --- 应用增强和处理 ---
        joint_data = joint_data_orig.copy()
        if self.center_joint_idx is not None and 0 <= self.center_joint_idx < self.num_nodes and joint_data.shape[0] > 0:
            center_coord = joint_data[:, self.center_joint_idx:self.center_joint_idx+1, :].copy()
            joint_data = joint_data - center_coord
        if self.split == 'train':
            if self.apply_random_shear: joint_data = self.random_shear(joint_data, self.shear_amplitude)
            if self.apply_rand_view_transform: joint_data = self.rand_view_transform(joint_data)
            if self.apply_random_flip and random.random() < 0.5: joint_data = self.random_flip(joint_data)
            if self.apply_coord_drop: joint_data = self.random_coordinate_dropout(joint_data, self.coord_drop_prob, self.coord_drop_axis)
            if self.apply_joint_drop: joint_data = self.random_joint_dropout(joint_data, self.joint_drop_prob)

        # --- 时间采样 ---
        joint_data_sampled = self.temporal_sampling(joint_data)
        if joint_data_sampled.shape[0] == 0:
            logger.warning(f"Temporal sampling resulted in empty data for index {true_index} (sample_id: {sample_id}). Returning zero sample.")
            zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float); zero_label = torch.tensor(label, dtype=torch.long); zero_mask = torch.zeros(self.max_len, dtype=torch.bool); return zero_data, zero_label, zero_mask, true_index

        # --- 计算衍生模态和拼接 ---
        modal_data_list = []; data_bone_sampled = None
        try:
            for modality in self.modalities:
                if modality == 'joint': modal_data_list.append(joint_data_sampled)
                elif modality == 'bone': data_bone_sampled = self._get_bone_data(joint_data_sampled); modal_data_list.append(data_bone_sampled)
                elif modality == 'joint_motion': modal_data_list.append(self._get_motion_data(joint_data_sampled))
                elif modality == 'bone_motion':
                    if data_bone_sampled is None: data_bone_sampled = self._get_bone_data(joint_data_sampled)
                    modal_data_list.append(self._get_motion_data(data_bone_sampled))
            if not modal_data_list: raise ValueError("No modalities generated.")
            data_concatenated = np.concatenate(modal_data_list, axis=-1)
            if data_concatenated.shape[-1] != self.num_input_dim:
                raise ValueError(f"Concatenated channel ({data_concatenated.shape[-1]}) != expected ({self.num_input_dim})")
        except Exception as e:
             logger.error(f"样本 {sample_id}: 计算或拼接模态时出错: {e}. Returning zero sample.")
             zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float); zero_label = torch.tensor(label, dtype=torch.long); zero_mask = torch.zeros(self.max_len, dtype=torch.bool); return zero_data, zero_label, zero_mask, true_index

        # --- 填充与掩码 ---
        data_padded, mask_np = pad_sequence_with_mask(data_concatenated, self.max_len, self.num_nodes, self.num_input_dim)

        # --- 转换为 Tensor ---
        try:
            data_tensor = torch.from_numpy(data_padded).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool()
        except Exception as e:
            logger.error(f"将样本 {sample_id} 转换为 Tensor 时出错: {e}. Returning zero sample.")
            zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float); zero_label = torch.tensor(label, dtype=torch.long); zero_mask = torch.zeros(self.max_len, dtype=torch.bool); return zero_data, zero_label, zero_mask, true_index

        # 返回 true_index
        return data_tensor, label_tensor, mask_tensor, true_index

    # --- top_k 方法 ---
    def top_k(self, score, top_k):
        label_np = np.array(self.label)
        valid_indices = np.where(label_np != -1)[0]
        if len(valid_indices) == 0: return 0.0
        if score.shape[0] == len(valid_indices):
            valid_labels = label_np[valid_indices]
            valid_rankings = score.argsort(axis=1, descending=True)
        elif score.shape[0] == len(label_np):
            valid_scores = score[valid_indices, :]
            valid_labels = label_np[valid_indices]
            valid_rankings = valid_scores.argsort(axis=1, descending=True)
        else:
             logger.error(f"top_k: score shape {score.shape} 无法与 label count {len(label_np)} 或 valid label count {len(valid_indices)} 匹配！")
             return 0.0
        hit_count = 0
        for i in range(len(valid_labels)):
            if valid_labels[i] in valid_rankings[i, :top_k]: hit_count += 1
        return hit_count * 1.0 / (len(valid_labels) + 1e-6)


# --- 单元测试 ---
if __name__ == '__main__':
    # 设置日志级别为 INFO 或 DEBUG
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')
    logger.info("测试 Feeder for DHG14-28 (v2.0 - 完全修正 LOSO 加载逻辑)...")

    # --- !!! 修改为你的实际路径 !!! ---
    test_root_dir = 'data/DHG14-28/DHG14-28_sample_json'
    test_subject_for_val = 1
    # ---------------------------------

    if not os.path.isdir(test_root_dir): logger.error("测试根目录无效!"); sys.exit(1)

    def collate_fn_filter_none(batch):
        batch = [item for item in batch if item is not None and item[1] is not None and item[1].item() != -1]
        if not batch: return None
        try:
             from torch.utils.data.dataloader import default_collate; return default_collate(batch)
        except Exception as e: logger.error(f"CollateFn 错误: {e}"); return None

    try:
        logger.info(f"\n--- 测试训练集 (留出 Subject {test_subject_for_val}) ---")
        train_args = {'root_dir': test_root_dir, 'subject_idx': test_subject_for_val, 'split': 'train','label_type': 'label_14', 'max_len': 120, 'data_path': 'joint,bone', 'num_classes': 14, 'debug': False}
        train_dataset = Feeder(**train_args)
        logger.info(f"训练集样本数: {len(train_dataset)}")
        if len(train_dataset) > 0:
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn_filter_none, pin_memory=True)
            logger.info("开始迭代训练数据 (最多5批)...")
            batch_count = 0
            for i, batch_data in enumerate(tqdm(train_loader, desc="Train Batches")):
                if batch_data is None: logger.warning(f"训练 Batch {i} 为 None"); continue
                xb, lb, mb, idxb = batch_data; batch_count += 1
                if i < 2: logger.info(f" Batch {i}: X={xb.shape}, L={lb.shape}, M={mb.shape}, Idx={idxb}")
                if batch_count >= 5: break
            logger.info(f"训练数据迭代完成 ({batch_count} 批)。")
        else: logger.warning("训练集为空！")

        logger.info(f"\n--- 测试验证集 (Subject {test_subject_for_val}) ---")
        val_args = {'root_dir': test_root_dir, 'subject_idx': test_subject_for_val, 'split': 'val', 'label_type': 'label_28', 'max_len': 150, 'data_path': 'joint', 'num_classes': 28, 'debug': False}
        val_dataset = Feeder(**val_args)
        logger.info(f"验证集样本数: {len(val_dataset)}")
        if len(val_dataset) > 0:
            from torch.utils.data import DataLoader
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn_filter_none, pin_memory=True)
            logger.info("开始迭代验证数据 (最多5批)...")
            batch_count = 0
            for i, batch_data in enumerate(tqdm(val_loader, desc="Val Batches")):
                if batch_data is None: logger.warning(f"验证 Batch {i} 为 None"); continue
                xb_v, lb_v, mb_v, idxb_v = batch_data; batch_count += 1
                if i < 2: logger.info(f" Batch {i}: X={xb_v.shape}, L={lb_v.shape}, M={mb_v.shape}, Idx={idxb_v}")
                if batch_count >= 5: break
            logger.info(f"验证数据迭代完成 ({batch_count} 批)。")
        else: logger.warning("验证集为空！")

    except Exception as e:
         logger.error(f"单元测试过程中出错: {e}", exc_info=True)