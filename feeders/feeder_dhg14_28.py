# 文件名: feeders/feeder_dhg14_28.py (适配 SDT-GRU 模型)

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
import glob # 导入 glob
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- 辅助函数：填充/截断序列并生成掩码 ---
def pad_sequence_with_mask(seq, max_len, num_nodes, num_channels, pad_value=0.0):
    """将序列填充/截断到指定长度，并返回序列和掩码"""
    if seq is None or seq.size == 0:
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask

    seq_len = seq.shape[0]

    if seq_len == 0:
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask

    # 检查输入维度是否基本正确 (T, N, C_base or C_total)
    if seq.ndim < 3 or seq.shape[1] != num_nodes:
         logger.warning(f"Pad input seq shape {seq.shape} mismatch expected node count {num_nodes}. Returning zeros.")
         padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
         mask = np.zeros(max_len, dtype=bool)
         return padded_seq, mask
    # 允许通道数是 C_base 或 C_total (拼接后)
    current_channels = seq.shape[2]
    if current_channels != num_channels:
        logger.warning(f"Pad input seq channel {current_channels} mismatch expected {num_channels}. Padding/truncating channels if possible or returning zeros.")
        # 尝试处理通道不匹配，简单填充/截断 (或者报错)
        if current_channels < num_channels:
            padding_channels = np.full((seq_len, num_nodes, num_channels - current_channels), pad_value, dtype=seq.dtype)
            seq = np.concatenate([seq, padding_channels], axis=-1)
        else:
            seq = seq[..., :num_channels]
        # 如果无法处理，还是返回零吧
        # padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        # mask = np.zeros(max_len, dtype=bool)
        # return padded_seq, mask


    if seq_len < max_len:
        pad_len = max_len - seq_len
        # 使用正确的通道数创建 padding
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
    用于 DHG14-28 数据集的 Feeder 类 (适配 SDT-GRU 模型)。
    支持留一交叉验证 (LOSO) 结构。
    """
    def __init__(self, root_dir, subject_idx, split, # <<<--- 使用 subject_idx
                 label_type='label_28',
                 max_len=150,
                 data_path='joint',
                 base_channel=3,
                 repeat=1,
                 random_choose=False,
                 # --- 通用增强参数 ---
                 apply_rand_view_transform=True, # 旋转缩放
                 apply_random_shear=False, shear_amplitude=0.5, # 剪切
                 apply_random_flip=False, # <<<--- DHG 默认关闭翻转
                 apply_coord_drop=False, coord_drop_prob=0.1, coord_drop_axis=None,
                 apply_joint_drop=False, joint_drop_prob=0.1,
                 center_joint_idx=0, # <<<--- DHG 中心点索引，默认为 0 (需确认)
                 debug=False,
                 **kwargs): # 接收 num_classes 等其他参数

        super().__init__()

        # 参数校验
        if not os.path.isdir(root_dir): raise ValueError(f"无效的数据集根目录: {root_dir}")
        if split not in ['train', 'val', 'test']: raise ValueError(f"无效的 split: {split}")
        if not (1 <= subject_idx <= 20): raise ValueError(f"无效的 subject_idx: {subject_idx}, 应在 1 到 20 之间")

        self.root_dir = root_dir
        self.subject_idx = subject_idx
        self.split = split
        self.train_val = 'train' if split == 'train' else 'val'
        self.label_type = label_type
        self.max_len = max_len
        self.repeat = repeat if self.train_val == 'train' else 1
        self.random_choose = random_choose if self.train_val == 'train' else False
        self.center_joint_idx = center_joint_idx # 可以为 None 来禁用中心化
        self.debug = debug

        # --- DHG14-28 特定参数 ---
        self.num_nodes = 22
        self.base_channel = base_channel
        # 骨骼连接 (0-based)
        self.bone_pairs = [(0, 1), (2, 0), (3, 2), (4, 3), (5, 4), (6, 1), (7, 6), (8, 7), (9, 8), (10, 1), (11, 10),
                           (12, 11), (13, 12), (14, 1), (15, 14), (16, 15), (17, 16), (18, 1), (19, 18), (20, 19), (21, 20)]
        # 左右手对应关节点 (需要根据 DHG 定义更新，但默认不使用翻转)
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
        self.apply_rand_view_transform = apply_rand_view_transform if self.train_val == 'train' else False
        self.apply_random_shear = apply_random_shear if self.train_val == 'train' else False
        self.shear_amplitude = shear_amplitude
        self.apply_random_flip = apply_random_flip if self.train_val == 'train' else False
        self.apply_coord_drop = apply_coord_drop if self.train_val == 'train' else False
        self.coord_drop_prob = coord_drop_prob
        self.coord_drop_axis = coord_drop_axis
        self.apply_joint_drop = apply_joint_drop if self.train_val == 'train' else False
        self.joint_drop_prob = joint_drop_prob

        logger.info(f"初始化 Feeder for DHG14-28 (Subject {subject_idx} as {split}): root={root_dir}")
        logger.info(f"标签类型: {label_type}, 目标序列长度: {max_len}")
        logger.info(f"加载模态: {self.modalities}, 总输入维度: {self.num_input_dim}")
        if self.train_val == 'train':
            logger.info(f"训练时增强: RandViewTransform={self.apply_rand_view_transform}, RandShear={self.apply_random_shear}, RandFlip={self.apply_random_flip}, CoordDrop={self.apply_coord_drop}, JointDrop={self.apply_joint_drop}")

        # --- 加载数据 ---
        self.sample_info = [] # 存储 {'id': ...}
        self.label = []       # 存储 0-based 标签
        self.data = {}        # 字典存储数据，键为 sample_id

        self._load_data_loso()

        # 过滤无效样本并转换为列表
        valid_indices = [i for i, info in enumerate(self.sample_info) if info['id'] in self.data and self.data[info['id']] is not None and self.label[i] != -1]
        self.sample_info = [self.sample_info[i] for i in valid_indices]
        self.label = [self.label[i] for i in valid_indices]
        self.data = [self.data[info['id']] for info in self.sample_info] # 转为列表

        if not self.sample_info: raise RuntimeError(f"未能加载任何有效的样本信息 for split '{split}' (subject {subject_idx})")
        if len(self.sample_info) != len(self.label) or len(self.sample_info) != len(self.data):
             raise RuntimeError(f"样本信息、标签、数据数量不匹配！ Split: {split}, Subject: {subject_idx}")

        # 推断类别数
        self.num_classes = kwargs.get('num_classes', 0) # 尝试从 kwargs 获取
        if self.num_classes == 0 and self.label:
            self.num_classes = max(self.label) + 1
        if self.num_classes == 0:
             # 尝试从 label_type 推断
             self.num_classes = 14 if '14' in self.label_type else 28
             logger.warning(f"无法从标签列表推断类别数，根据 label_type 猜测为 {self.num_classes}")
        logger.info(f"使用的类别数: {self.num_classes}")


        if self.debug:
             limit = min(100, len(self.sample_info)); self.sample_info=self.sample_info[:limit]; self.label=self.label[:limit]; self.data=self.data[:limit]
             logger.warning(f"!!! DEBUG 模式开启，只使用前 {limit} 个样本 !!!")

        logger.info(f"成功加载 {len(self.sample_info)} 个有效样本用于 '{split}' split (Subject {subject_idx} as {'validation' if split != 'train' else 'training'})。")


    def _load_data_loso(self):
        """根据 LOSO 结构加载数据列表和数据"""
        # 构造列表文件和数据目录路径
        list_file_path = os.path.join(self.root_dir, str(self.subject_idx), f"{self.subject_idx}{self.train_val}_samples.json")
        json_data_dir = os.path.join(self.root_dir, str(self.subject_idx), self.train_val)

        if not os.path.exists(list_file_path): raise FileNotFoundError(f"找不到列表文件: {list_file_path}")
        if not os.path.isdir(json_data_dir): raise FileNotFoundError(f"找不到 JSON 数据目录: {json_data_dir}")

        logger.info(f"从 {list_file_path} 加载样本列表...")
        try:
            with open(list_file_path, 'r') as f:
                sample_info_raw = json.load(f) # 先加载原始列表
        except Exception as e:
            logger.error(f"无法加载或解析列表文件 {list_file_path}: {e}"); raise e

        logger.info(f"开始从 {json_data_dir} 加载骨骼数据...")
        loaded_count = 0; skipped_label = 0; skipped_data = 0

        # 临时列表存储有效信息
        temp_sample_info = []
        temp_label = []

        for info in tqdm(sample_info_raw, desc=f"加载 S{self.subject_idx} {self.split} 数据"): # 添加进度条
            sample_id = info.get('file_name')
            if not sample_id: continue

            # 处理标签
            try:
                label_val = int(info[self.label_type])
                if label_val <= 0: raise ValueError("标签需为正")
                final_label = label_val - 1
            except (KeyError, ValueError, TypeError):
                 final_label = -1; skipped_label += 1

            # 读取对应的 JSON 数据文件
            json_path = os.path.join(json_data_dir, sample_id + '.json')
            data_numpy = self._read_dhg_json(json_path, sample_id)

            # 只有当数据和标签都有效时才加入
            if data_numpy is not None and final_label != -1:
                temp_sample_info.append({'id': sample_id}) # 简化存储
                temp_label.append(final_label)
                self.data[sample_id] = data_numpy # 仍然用字典临时存储
                loaded_count += 1
            else:
                skipped_data += 1
                if final_label != -1 and data_numpy is None: logger.debug(f"样本 {sample_id} 标签有效但数据加载失败。")
                elif final_label == -1 and data_numpy is not None: logger.debug(f"样本 {sample_id} 数据有效但标签无效。")
                # else: 两者都无效

        self.sample_info = temp_sample_info
        self.label = temp_label
        logger.info(f"数据加载完成。加载成功: {loaded_count}, 跳过(无效标签 {skipped_label}, 数据错误/缺失 {skipped_data})。")

    def _read_dhg_json(self, filepath, sample_id_for_debug):
        """读取单个 DHG JSON 文件并提取骨骼数据 (基于样本结构)"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f: json_data = json.load(f)
            skeletons_data = json_data.get('skeletons')
            if not skeletons_data or not isinstance(skeletons_data, list): raise ValueError("找不到 'skeletons' 列表或格式错误")

            sequence = []
            for frame_idx, frame_joints_coords in enumerate(skeletons_data):
                if not isinstance(frame_joints_coords, list) or len(frame_joints_coords) == 0:
                    if sequence: sequence.append(sequence[-1]); continue
                    else: sequence.append(np.zeros((self.num_nodes, self.base_channel), dtype=np.float32)); continue

                # 处理节点数
                current_frame_joints_list = frame_joints_coords
                if len(current_frame_joints_list) != self.num_nodes:
                    if len(current_frame_joints_list) < self.num_nodes:
                        padding = [[0.0] * self.base_channel] * (self.num_nodes - len(current_frame_joints_list))
                        current_frame_joints_list.extend(padding)
                    else:
                        current_frame_joints_list = current_frame_joints_list[:self.num_nodes]

                frame_data = []
                valid_joint_count = 0
                for joint_coords in current_frame_joints_list:
                    if isinstance(joint_coords, list) and len(joint_coords) >= self.base_channel and all(isinstance(c, (int, float)) and np.isfinite(c) for c in joint_coords[:self.base_channel]):
                        frame_data.append(joint_coords[:self.base_channel])
                        valid_joint_count += 1
                    else:
                        frame_data.append([0.0] * self.base_channel) # 无效坐标补零

                # 基于有效关节点比例决定是否使用该帧
                if valid_joint_count < self.num_nodes * 0.5:
                     if sequence: sequence.append(sequence[-1])
                     else: sequence.append(np.zeros((self.num_nodes, self.base_channel), dtype=np.float32))
                else:
                     sequence.append(np.array(frame_data, dtype=np.float32))

            if not sequence: return None
            data_numpy = np.stack(sequence, axis=0)
            if not np.any(data_numpy): return None # 检查是否全零
            return data_numpy

        except FileNotFoundError: logger.warning(f"找不到 JSON 文件: {filepath}"); return None
        except Exception as e: logger.warning(f"加载或处理 DHG 文件 {filepath} 时出错 ({e})"); return None

    # --- 数据增强函数 ---
    # (rand_view_transform, random_shear, random_flip(需谨慎使用), random_coordinate_dropout, random_joint_dropout)
    # (代码与 v1.3 版本相同，此处省略以减少篇幅，请确保它们存在)
    def rand_view_transform(self, X, agx=None, agy=None, s=None):
        if agx is None: agx = random.randint(-60, 60);
        if agy is None: agy = random.randint(-60, 60);
        if s is None: s = random.uniform(0.7, 1.3);
        agx_rad = math.radians(agx); agy_rad = math.radians(agy);
        Rx = np.array([[1,0,0], [0,math.cos(agx_rad),math.sin(agx_rad)], [0,-math.sin(agx_rad),math.cos(agx_rad)]], dtype=X.dtype);
        Ry = np.array([[math.cos(agy_rad),0,-math.sin(agy_rad)], [0,1,0], [math.sin(agy_rad),0,math.cos(agy_rad)]], dtype=X.dtype);
        Ss = np.array([[s,0,0],[0,s,0],[0,0,s]], dtype=X.dtype);
        orig_shape = X.shape;
        if X.shape[-1] != 3: return X;
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
        """随机水平翻转骨架 (如果 left/right parts 定义了)"""
        logger.warning("DHG 的 Random Flip 未验证是否适用！请检查 left/right parts 定义。") # 添加更明确的警告

        # 检查左右部分是否已定义，如果没有定义则直接返回
        if not self.left_parts or not self.right_parts:
            logger.debug("未定义 left_parts 或 right_parts，跳过翻转。") # 添加 debug 信息
            return X

        # 获取维度并检查通道数
        T, N, C = X.shape
        if C != 3: # 只对 3D 坐标有效
            logger.debug(f"输入通道数不为 3 ({C})，跳过翻转。")
            return X

        # 复制数据以避免修改原始数据
        flipped_X = X.copy()
        # 翻转 X 坐标 (假设 X 轴是水平轴)
        flipped_X[..., 0] *= -1

        # 交换左右对应的关节点
        for l, r in zip(self.left_parts, self.right_parts):
            # 确保索引有效
            if 0 <= l < N and 0 <= r < N:
                # 使用临时变量或直接交换，避免数据覆盖问题 (虽然 numpy 可能处理得当，但这样更清晰)
                temp_l_data = flipped_X[:, l, :].copy() # 显式复制
                flipped_X[:, l, :] = flipped_X[:, r, :]
                flipped_X[:, r, :] = temp_l_data
            else:
                 logger.warning(f"翻转时索引越界: left={l}, right={r}, N={N}")

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

    # --- 时间采样和填充函数 ---
    def temporal_sampling(self, data_numpy):
        T_orig = data_numpy.shape[0]
        if T_orig == 0: return np.zeros((self.max_len, self.num_nodes, data_numpy.shape[-1]), dtype=data_numpy.dtype)
        if T_orig <= self.max_len: return data_numpy

        if self.random_choose:
             indices = random.sample(range(T_orig), self.max_len); indices.sort(); return data_numpy[indices, :, :]
        else:
             indices = np.linspace(0, T_orig - 1, self.max_len).round().astype(int); indices = np.clip(indices, 0, T_orig - 1); return data_numpy[indices, :, :]

    # --- 衍生模态计算函数 ---
    def _get_bone_data(self, joint_data):
        bone_data = np.zeros_like(joint_data);
        for v1, v2 in self.bone_pairs: bone_data[:, v1, :] = joint_data[:, v1, :] - joint_data[:, v2, :];
        return bone_data
    def _get_motion_data(self, data):
        motion_data = np.zeros_like(data); T = data.shape[0];
        if T > 1: motion_data[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]; motion_data[-1, :, :] = motion_data[-2, :, :];
        return motion_data

    def __len__(self):
        return len(self.sample_info) * self.repeat

    def __getitem__(self, index):
        true_index = index % len(self.sample_info)

        label = self.label[true_index]
        joint_data_orig = self.data[true_index] # 直接从加载好的列表获取

        if joint_data_orig is None or label == -1:
             # 返回零样本
             zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float)
             zero_label = torch.tensor(-1, dtype=torch.long)
             zero_mask = torch.zeros(self.max_len, dtype=torch.bool)
             return zero_data, zero_label, zero_mask, true_index

        # 应用增强和处理
        joint_data = joint_data_orig.copy()
        if self.train_val == 'train':
            # 中心化
            center_joint_dhg = getattr(self, 'center_joint_idx', 0) # 使用配置或默认值 0
            if center_joint_dhg is not None and 0 <= center_joint_dhg < self.num_nodes and joint_data.shape[0] > 0:
                center_coord = joint_data[:, center_joint_dhg:center_joint_dhg+1, :]; joint_data = joint_data - center_coord
            # 其他增强
            if self.apply_random_shear: joint_data = self.random_shear(joint_data, self.shear_amplitude)
            if self.apply_rand_view_transform: joint_data = self.rand_view_transform(joint_data)
            if self.apply_random_flip and random.random() < 0.5: joint_data = self.random_flip(joint_data)
            if self.apply_coord_drop: joint_data = self.random_coordinate_dropout(joint_data, self.coord_drop_prob, self.coord_drop_axis)
            if self.apply_joint_drop: joint_data = self.random_joint_dropout(joint_data, self.joint_drop_prob)
        else: # 验证/测试时只中心化
            center_joint_dhg = getattr(self, 'center_joint_idx', 0)
            if center_joint_dhg is not None and 0 <= center_joint_dhg < self.num_nodes and joint_data.shape[0] > 0:
                 center_coord = joint_data[:, center_joint_dhg:center_joint_dhg+1, :]; joint_data = joint_data - center_coord

        # 时间采样
        joint_data_sampled = self.temporal_sampling(joint_data)

        # 计算衍生模态和拼接
        modal_data_list = []; data_bone_sampled = None
        for modality in self.modalities:
            if modality == 'joint': modal_data_list.append(joint_data_sampled)
            elif modality == 'bone': data_bone_sampled = self._get_bone_data(joint_data_sampled); modal_data_list.append(data_bone_sampled)
            elif modality == 'joint_motion': modal_data_list.append(self._get_motion_data(joint_data_sampled))
            elif modality == 'bone_motion':
                if data_bone_sampled is None: data_bone_sampled = self._get_bone_data(joint_data_sampled)
                modal_data_list.append(self._get_motion_data(data_bone_sampled))

        if not modal_data_list: # 返回零样本
             zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float); zero_label = torch.tensor(-1, dtype=torch.long); zero_mask = torch.zeros(self.max_len, dtype=torch.bool); return zero_data, zero_label, zero_mask, true_index

        try: data_concatenated = np.concatenate(modal_data_list, axis=-1)
        except ValueError as e: # 返回零样本
             logger.error(f"样本 {self.sample_info[true_index]['id']} 拼接模态时出错: {e}"); zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float); zero_label = torch.tensor(-1, dtype=torch.long); zero_mask = torch.zeros(self.max_len, dtype=torch.bool); return zero_data, zero_label, zero_mask, true_index

        # 填充与掩码
        data_padded, mask_np = pad_sequence_with_mask(data_concatenated, self.max_len, self.num_nodes, self.num_input_dim)

        # 转换为 Tensor
        try:
            data_tensor = torch.from_numpy(data_padded).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool()
        except Exception as e: # 返回零样本
            logger.error(f"将样本 {self.sample_info[true_index]['id']} 转换为 Tensor 时出错: {e}"); zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float); zero_label = torch.tensor(-1, dtype=torch.long); zero_mask = torch.zeros(self.max_len, dtype=torch.bool); return zero_data, zero_label, zero_mask, true_index

        return data_tensor, label_tensor, mask_tensor, true_index

    # --- top_k 方法 ---
    def top_k(self, score, top_k):
        """计算 Top-K 准确率 (处理 0-based 和无效标签)"""
        # (代码与 v1.3 版本相同)
        rank = score.argsort(axis=1, descending=True)
        label_np = np.array(self.label)
        valid_indices = np.where(label_np != -1)[0]
        if len(valid_indices) == 0: return 0.0
        valid_labels = label_np[valid_indices]; valid_rankings = rank[valid_indices, :]
        hit_count = 0
        for i in range(len(valid_labels)):
            if valid_labels[i] in valid_rankings[i, :top_k]: hit_count += 1
        return hit_count * 1.0 / len(valid_labels)