# 文件名: feeders/feeder_shrec17.py (适配 SDT-GRU 模型)

import json
import os
import sys
import numpy as np
import random
import pickle
import logging
import torch # <--- 确保导入 torch
from torch.utils.data import Dataset

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

    # 确保输入维度正确
    if seq.shape[1] != num_nodes or seq.shape[2] != num_channels:
         # 尝试处理维度不匹配，如果无法处理则返回错误填充
         logger.warning(f"Pad input seq shape {seq.shape} mismatch expected (T, {num_nodes}, {num_channels}). Trying to adapt or returning zeros.")
         # 简单的适配或返回零逻辑，根据你的数据具体情况决定
         # 这里返回零作为安全 fallback
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
    用于 SHREC'17 数据集的 Feeder 类 (适配 SDT-GRU 模型)。
    """
    def __init__(self, root_dir, list_file, split,
                 label_type='label_28',
                 max_len=180,           # <<<--- SDT-GRU 需要固定长度
                 data_path='joint',     # 支持多模态 ('joint', 'bone', 'motion')
                 base_channel=3,        # 基础模态通道数
                 repeat=1,
                 random_choose=False,   # 时间采样策略
                 apply_random_translation=True, # 随机平移增强
                 debug=False,
                 **kwargs): # 接收其他可能的参数

        super().__init__()

        if not os.path.isdir(root_dir): raise ValueError(f"无效的数据集根目录: {root_dir}")
        if not os.path.exists(list_file): raise ValueError(f"找不到列表文件: {list_file}")
        if split not in ['train', 'val', 'test']: raise ValueError(f"无效的 split: {split}")

        self.root_dir = root_dir
        self.list_file = list_file
        self.split = split
        self.train_val = 'train' if split == 'train' else 'val' # 用于区分增强
        self.label_type = label_type
        self.max_len = max_len
        self.repeat = repeat if self.train_val == 'train' else 1
        self.random_choose = random_choose if self.train_val == 'train' else False # 时间采样
        self.apply_random_translation = apply_random_translation if self.train_val == 'train' else False # 平移增强
        self.debug = debug

        # --- SHREC'17 特定参数 ---
        self.num_nodes = 22
        self.base_channel = base_channel # 通常是 3 (x, y, z)
        # 定义骨骼连接 (0-based)，用于计算 bone 模态
        self.bone_pairs = [(0, 1), (2, 0), (3, 2), (4, 3), (5, 4), (6, 1), (7, 6), (8, 7), (9, 8), (10, 1), (11, 10),
                           (12, 11), (13, 12), (14, 1), (15, 14), (16, 15), (17, 16), (18, 1), (19, 18), (20, 19), (21, 20)]

        # --- 解析多模态 ---
        if isinstance(data_path, str): self.modalities = [m.strip().lower() for m in data_path.split(',') if m.strip()]
        elif isinstance(data_path, list): self.modalities = [m.strip().lower() for m in data_path if isinstance(m, str) and m.strip()]
        else: raise ValueError("data_path 必须是逗号分隔的字符串或字符串列表")
        valid_modalities = ['joint', 'bone', 'joint_motion', 'bone_motion'] # 定义支持的模态
        for m in self.modalities:
            if m not in valid_modalities: raise ValueError(f"不支持的数据模态: '{m}'. 支持: {valid_modalities}")
        if not self.modalities: raise ValueError("必须至少指定一种数据模态 (data_path)")
        self.num_input_dim = self.base_channel * len(self.modalities) # 总输入维度

        logger.info(f"初始化 Feeder for SHREC'17 (for SDT-GRU): split={split}, root={root_dir}")
        logger.info(f"列表文件: {list_file}, 标签类型: {label_type}")
        logger.info(f"目标序列长度 (max_len): {max_len}")
        logger.info(f"加载模态: {self.modalities}, 总输入维度: {self.num_input_dim}")

        self.sample_info = [] # 存储 {'file_name': ..., 'label_14': ..., 'label_28': ...}
        self.label = []       # 存储 0-based 标签
        self.data = []        # 存储加载的 numpy 数据

        self._load_sample_list()
        self._load_data() # 加载数据到内存

        if not self.sample_info or not self.data:
            raise RuntimeError(f"未能加载任何样本或数据 for split '{split}'")
        if len(self.sample_info) != len(self.label) or len(self.sample_info) != len(self.data):
             raise RuntimeError(f"样本信息 ({len(self.sample_info)}), 标签 ({len(self.label)}), 数据 ({len(self.data)}) 数量不匹配！ Split: {split}")

        if self.debug:
            logger.warning(f"!!! DEBUG 模式开启，只使用前 100 个样本 !!!")
            limit = 100
            self.sample_info = self.sample_info[:limit]
            self.label = self.label[:limit]
            self.data = self.data[:limit]

        logger.info(f"成功加载 {len(self.sample_info)} 个样本用于 '{split}' split。")

    def _load_sample_list(self):
        """从 JSON 列表文件加载样本信息和标签"""
        try:
            with open(self.list_file, 'r') as f:
                self.sample_info = json.load(f)
        except Exception as e:
            logger.error(f"无法加载或解析列表文件 {self.list_file}: {e}")
            raise e

        self.label = []
        num_classes = 0
        invalid_label_count = 0
        for info in self.sample_info:
            try:
                label_val = int(info[self.label_type])
                if label_val <= 0: raise ValueError("标签值必须为正整数")
                self.label.append(label_val - 1) # 转为 0-based
                if label_val > num_classes: num_classes = label_val
            except (KeyError, ValueError, TypeError) as e:
                # logger.warning(f"样本 {info.get('file_name', '未知')} 的标签 '{self.label_type}' 无效或缺失 ({e})，标记为 -1")
                self.label.append(-1)
                invalid_label_count += 1

        self.num_classes = num_classes
        logger.info(f"从标签类型 '{self.label_type}' 推断出 {self.num_classes} 个类别。发现 {invalid_label_count} 个无效标签。")
        if self.num_classes == 0 and invalid_label_count < len(self.sample_info):
            logger.warning("未能从有效标签中推断出类别数量！请检查标签文件。")
            # 或者根据配置强制设置 num_classes


    def _load_data(self):
        """加载所有样本的骨骼数据到内存"""
        logger.info("开始加载骨骼数据到内存...")
        self.data = []
        skipped_count = 0
        json_dir = os.path.join(self.root_dir, 'train_jsons' if self.train_val == 'train' else 'test_jsons')
        if not os.path.isdir(json_dir):
             raise FileNotFoundError(f"找不到 JSON 数据目录: {json_dir}")

        for i, info in enumerate(self.sample_info):
            if self.label[i] == -1: # 跳过无效标签的样本数据加载
                self.data.append(None)
                continue

            file_name = info.get('file_name')
            if not file_name:
                logger.warning(f"样本 {i} 缺少 'file_name'，无法加载数据。")
                self.data.append(None); skipped_count += 1
                continue

            json_path = os.path.join(json_dir, file_name + '.json')
            try:
                with open(json_path, 'r') as f:
                    json_file = json.load(f)
                # SHREC JSON 结构可能直接是列表或字典包含 'skeletons'
                skeletons_data = None
                if isinstance(json_file, dict):
                    skeletons_data = json_file.get('skeletons')
                elif isinstance(json_file, list): # 有些格式可能直接是帧列表
                    skeletons_data = json_file

                if not skeletons_data or not isinstance(skeletons_data, list) or len(skeletons_data) == 0:
                    raise ValueError("JSON 中骨骼数据无效或为空")

                sequence = []
                for frame_joints in skeletons_data:
                    # 检查帧数据格式，可能是列表或字典
                    current_frame_joints = []
                    if isinstance(frame_joints, list) and len(frame_joints) > 0:
                        # 假设 frame_joints 是 [[x,y,z], [x,y,z], ...]
                         if isinstance(frame_joints[0], list) and len(frame_joints[0]) >= self.base_channel:
                             current_frame_joints = frame_joints
                         # else: 可能需要其他解析方式
                    elif isinstance(frame_joints, dict) and 'joints' in frame_joints: # 兼容更复杂的帧结构
                        current_frame_joints = frame_joints['joints']
                        # 可能还需要进一步解析 joint 内部结构

                    # 确保关节点数量正确
                    if len(current_frame_joints) != self.num_nodes:
                         # logger.warning(f"文件 {file_name} 某帧节点数 {len(current_frame_joints)} != {self.num_nodes}, 填充/截断.")
                         if len(current_frame_joints) < self.num_nodes:
                             padding = [[0.0] * self.base_channel] * (self.num_nodes - len(current_frame_joints))
                             current_frame_joints.extend(padding)
                         else:
                             current_frame_joints = current_frame_joints[:self.num_nodes]

                    # 提取坐标并检查有效性
                    frame_data = []
                    valid_frame = True
                    for joint in current_frame_joints:
                        if isinstance(joint, list) and len(joint) >= self.base_channel:
                            coords = joint[:self.base_channel]
                            if all(isinstance(c, (int, float)) and np.isfinite(c) for c in coords):
                                frame_data.append(coords)
                            else:
                                # logger.warning(f"文件 {file_name} 某帧包含无效坐标: {joint}, 用零替换。")
                                frame_data.append([0.0] * self.base_channel)
                                # valid_frame = False; break # 或者标记帧无效
                        else:
                            # logger.warning(f"文件 {file_name} 某帧关节点格式无效: {joint}, 用零替换。")
                            frame_data.append([0.0] * self.base_channel)
                            # valid_frame = False; break
                    # if not valid_frame: # 如果决定跳过无效帧
                    #     if sequence: sequence.append(sequence[-1]) # 重复上一帧
                    #     else: sequence.append(np.zeros((self.num_nodes, self.base_channel), dtype=np.float32))
                    # else:
                    sequence.append(np.array(frame_data, dtype=np.float32))


                if not sequence: raise ValueError("处理后序列为空")

                value = np.stack(sequence, axis=0) # (T, N, C_base)
                if value.shape[1] != self.num_nodes or value.shape[2] != self.base_channel:
                    raise ValueError(f"Numpy 数组维度错误: {value.shape}, 期望 T x {self.num_nodes} x {self.base_channel}")

                self.data.append(value)

            except FileNotFoundError:
                # logger.warning(f"找不到 JSON 文件: {json_path}，样本 {i} 数据为 None")
                self.data.append(None); skipped_count += 1
            except Exception as e:
                logger.warning(f"加载或处理文件 {json_path} 时出错 ({e})，样本 {i} 数据为 None")
                self.data.append(None); skipped_count += 1
        logger.info(f"数据加载完成。成功加载 {len(self.data) - skipped_count} 个，跳过 {skipped_count} 个。")

    def random_translation(self, ske_data):
        """应用随机平移"""
        T, N, C = ske_data.shape
        if C != 3: return ske_data

        offset = np.random.uniform(-0.01, 0.01, size=(1, 1, C)).astype(ske_data.dtype)
        data = ske_data + offset # 应用广播
        return data

    def temporal_sampling(self, data_numpy):
        """根据 random_choose 进行时间采样，返回原始长度或采样后的序列"""
        T_orig = data_numpy.shape[0]
        if T_orig <= self.max_len: # 长度足够短，无需采样
            return data_numpy

        if self.random_choose: # 训练时随机采样
             indices = random.sample(range(T_orig), self.max_len)
             indices.sort()
             return data_numpy[indices, :, :]
        else: # 验证/测试时均匀采样
             indices = np.linspace(0, T_orig - 1, self.max_len).round().astype(int)
             indices = np.clip(indices, 0, T_orig - 1) # 确保索引有效
             return data_numpy[indices, :, :]

    # --- 用于计算衍生模态的辅助函数 ---
    def _get_bone_data(self, joint_data):
        """计算骨骼数据"""
        bone_data = np.zeros_like(joint_data)
        for v1, v2 in self.bone_pairs: # 使用 0-based 索引
            bone_data[:, v1, :] = joint_data[:, v1, :] - joint_data[:, v2, :]
        return bone_data

    def _get_motion_data(self, data):
        """计算运动数据（帧间差分）"""
        motion_data = np.zeros_like(data)
        T = data.shape[0]
        if T > 1:
            motion_data[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            motion_data[-1, :, :] = motion_data[-2, :, :] # 重复最后一帧的运动
        return motion_data

    def __len__(self):
        return len(self.sample_info) * self.repeat

    def __getitem__(self, index):
        true_index = index % len(self.sample_info)

        label = self.label[true_index]
        joint_data_orig = self.data[true_index] # 获取原始 joint 数据

        # 处理无效样本
        if joint_data_orig is None or label == -1:
             # 返回一个符合维度的零样本和无效标签，或者让 collate_fn 处理
             logger.debug(f"样本 {true_index} 无效，返回零样本。")
             zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float)
             zero_label = torch.tensor(-1, dtype=torch.long)
             zero_mask = torch.zeros(self.max_len, dtype=torch.bool)
             # 索引还是返回真实的，方便调试
             return zero_data, zero_label, zero_mask, true_index
             # 或者 return None (需要 collate_fn 处理)

        # --- 1. 数据增强 (作用于原始 joint 数据) ---
        joint_data = joint_data_orig.copy()
        if self.apply_random_translation:
            joint_data = self.random_translation(joint_data)
        # 在这里可以添加其他你需要的增强，比如随机旋转等（如果实现）

        # --- 2. 时间采样（作用于增强后的 joint 数据）---
        joint_data_sampled = self.temporal_sampling(joint_data)

        # --- 3. 计算衍生模态 (基于采样后的 joint 数据) ---
        modal_data_list = []
        data_bone_sampled = None # 缓存计算结果

        for modality in self.modalities:
            if modality == 'joint':
                modal_data_list.append(joint_data_sampled)
            elif modality == 'bone':
                data_bone_sampled = self._get_bone_data(joint_data_sampled)
                modal_data_list.append(data_bone_sampled)
            elif modality == 'joint_motion':
                modal_data_list.append(self._get_motion_data(joint_data_sampled))
            elif modality == 'bone_motion':
                if data_bone_sampled is None: # 确保 bone 已计算
                     data_bone_sampled = self._get_bone_data(joint_data_sampled)
                modal_data_list.append(self._get_motion_data(data_bone_sampled))

        if not modal_data_list: # 如果请求的模态都无法计算
            logger.error(f"样本 {true_index} 无法生成任何请求的模态数据！")
            # 返回零样本
            zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float)
            zero_label = torch.tensor(-1, dtype=torch.long)
            zero_mask = torch.zeros(self.max_len, dtype=torch.bool)
            return zero_data, zero_label, zero_mask, true_index

        # --- 4. 拼接多模态特征 ---
        try:
             # 确保所有模态数据形状一致
             first_shape = modal_data_list[0].shape
             for i, mod_data in enumerate(modal_data_list[1:]):
                 if mod_data.shape != first_shape:
                     # 尝试修复（比如填充最后一维），或者报错
                     raise ValueError(f"模态 {self.modalities[i+1]} 的形状 {mod_data.shape} 与第一个模态 {first_shape} 不匹配！")
             data_concatenated = np.concatenate(modal_data_list, axis=-1) # 拼接通道 C 维度
        except ValueError as e:
             logger.error(f"样本 {true_index} 拼接模态时出错: {e}")
             # 返回零样本
             zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float)
             zero_label = torch.tensor(-1, dtype=torch.long)
             zero_mask = torch.zeros(self.max_len, dtype=torch.bool)
             return zero_data, zero_label, zero_mask, true_index


        # --- 5. 填充序列长度并生成掩码 ---
        # 注意：拼接后的数据维度是 (T_sampled, N, C_total)
        # T_sampled 可能是 self.max_len 或更短（如果原始序列就短）
        # pad_sequence_with_mask 需要知道预期的 num_input_dim (C_total)
        data_padded, mask_np = pad_sequence_with_mask(data_concatenated, self.max_len, self.num_nodes, self.num_input_dim)

        # --- 6. 转换为 Tensor ---
        try:
            # 输出格式 (T, N, C) - SDT-GRU 模型内部会处理 B 维度
            data_tensor = torch.from_numpy(data_padded).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool() # (T,)
        except Exception as e:
            logger.error(f"将样本 {true_index} 转换为 Tensor 时出错: {e}")
            # 返回零样本
            zero_data = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float)
            zero_label = torch.tensor(-1, dtype=torch.long)
            zero_mask = torch.zeros(self.max_len, dtype=torch.bool)
            return zero_data, zero_label, zero_mask, true_index


        return data_tensor, label_tensor, mask_tensor, true_index

    def top_k(self, score, top_k):
        """计算 Top-K 准确率 (处理 0-based 和无效标签)"""
        rank = score.argsort(axis=1, descending=True) # score shape: (N, num_class)
        label_np = np.array(self.label)
        valid_indices = np.where(label_np != -1)[0]
        if len(valid_indices) == 0: return 0.0

        valid_labels = label_np[valid_indices]
        valid_rankings = rank[valid_indices, :]

        hit_count = 0
        for i in range(len(valid_labels)):
            if valid_labels[i] in valid_rankings[i, :top_k]:
                hit_count += 1
        return hit_count * 1.0 / len(valid_labels)

# --- 单元测试 (可选) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("测试 Feeder for SHREC'17 (适配 SDT-GRU)...")

    # --- !!! 修改为你的实际路径 !!! ---
    test_root_dir = 'data/shrec/shrec17_jsons'
    train_list = 'data/shrec/shrec17_jsons/train_samples.json'
    test_list = 'data/shrec/shrec17_jsons/test_samples.json'
    # ---------------------------------

    if not os.path.isdir(test_root_dir) or not os.path.exists(train_list) or not os.path.exists(test_list):
        logger.error("请确保测试路径设置正确！")
        sys.exit(1)

    def collate_fn_filter_none(batch):
        """过滤掉返回 None 的样本"""
        batch = [item for item in batch if item is not None and item[0] is not None]
        if not batch: return None
        try:
             # 默认的 collate 应该能处理 (T, N, C) 的数据
             from torch.utils.data.dataloader import default_collate
             return default_collate(batch)
             # 如果需要自定义堆叠，可以像下面这样：
             # xs = torch.stack([item[0] for item in batch], 0) # (B, T, N, C)
             # ls = torch.stack([item[1] for item in batch], 0) # (B,)
             # ms = torch.stack([item[2] for item in batch], 0) # (B, T)
             # idxs = torch.tensor([item[3] for item in batch], dtype=torch.long) # (B,)
             # return xs, ls, ms, idxs
        except Exception as e:
             logger.error(f"CollateFn 错误: {e}")
             for i, item in enumerate(batch):
                 try: logger.error(f" Item {i} shapes: {item[0].shape}, {item[1].shape}, {item[2].shape}, {type(item[3])}")
                 except: logger.error(f" Item {i} is problematic: {item}")
             return None

    try:
        # --- 测试训练集加载 (joint, 28类) ---
        logger.info("\n--- 测试训练集 (joint, 28类) ---")
        train_args = {
            'root_dir': test_root_dir, 'list_file': train_list, 'split': 'train',
            'label_type': 'label_28', 'max_len': 150, 'data_path': 'joint',
            'num_classes': 28, 'debug': True, 'apply_random_translation': True,
            'random_choose': True
        }
        train_dataset = Feeder(**train_args)
        logger.info(f"训练集样本数 (Debug): {len(train_dataset)}")
        logger.info(f"输入维度: {train_dataset.num_input_dim}")
        if len(train_dataset) > 0:
            item1 = train_dataset[0]
            if item1:
                 x1, l1, m1, idx1 = item1
                 logger.info(f"样本 {idx1} - X shape: {x1.shape}, Label: {l1.item()}, Mask shape: {m1.shape}, Mask sum: {m1.sum().item()}")
                 assert x1.shape == (train_args['max_len'], train_dataset.num_nodes, 3)
                 assert m1.shape == (train_args['max_len'],)
            else: logger.warning("第一个训练样本加载失败")

            # 测试 DataLoader
            from torch.utils.data import DataLoader
            # 注意：DataLoader 默认会将 (T, N, C) 堆叠为 (B, T, N, C)，这正是 SDT-GRU 模型期望的输入格式
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_filter_none)
            first_batch = next(iter(train_loader), None)
            if first_batch:
                xb, lb, mb, idxb = first_batch
                logger.info(f"Batch X shape: {xb.shape}, Label shape: {lb.shape}, Mask shape: {mb.shape}")
                assert xb.shape == (4, train_args['max_len'], train_dataset.num_nodes, 3) # B, T, N, C
                assert lb.shape == (4,)
                assert mb.shape == (4, train_args['max_len']) # B, T
            else: logger.warning("训练批次为空或加载失败")

        # --- 测试测试集加载 (joint, bone, 14类) ---
        logger.info("\n--- 测试测试集 (joint, bone, 14类) ---")
        test_args = {
            'root_dir': test_root_dir, 'list_file': test_list, 'split': 'test',
            'label_type': 'label_14', 'max_len': 180, 'data_path': 'joint,bone',
            'num_classes': 14, 'debug': True, 'apply_random_translation': False,
            'random_choose': False
        }
        test_dataset = Feeder(**test_args)
        logger.info(f"测试集样本数 (Debug): {len(test_dataset)}")
        logger.info(f"输入维度: {test_dataset.num_input_dim}") # 应该是 3 * 2 = 6
        if len(test_dataset) > 0:
            item_test = test_dataset[0]
            if item_test:
                 xt, lt, mt, idxt = item_test
                 logger.info(f"样本 {idxt} - X shape: {xt.shape}, Label: {lt.item()}, Mask shape: {mt.shape}, Mask sum: {mt.sum().item()}")
                 assert xt.shape == (test_args['max_len'], test_dataset.num_nodes, 6) # T, N, C=6
                 assert mt.shape == (test_args['max_len'],)
            else: logger.warning("第一个测试样本加载失败")

            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn_filter_none)
            test_batch = next(iter(test_loader), None)
            if test_batch:
                 xb_t, lb_t, mb_t, idxb_t = test_batch
                 logger.info(f"Batch X shape: {xb_t.shape}, Label shape: {lb_t.shape}, Mask shape: {mb_t.shape}")
                 assert xb_t.shape == (2, test_args['max_len'], test_dataset.num_nodes, 6)
                 assert lb_t.shape == (2,)
                 assert mb_t.shape == (2, test_args['max_len'])
            else: logger.warning("测试批次为空或加载失败")

    except Exception as e:
         logger.error(f"单元测试过程中出错: {e}", exc_info=True)