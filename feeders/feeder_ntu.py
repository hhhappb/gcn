# -*- coding: utf-8 -*-
# 文件名: feeders/feeder_ntu.py
# 描述: 适用于 SDT_GRUs_Gesture 模型的 NTU RGB+D 数据集加载器。
#       基于 TD-GCN 的 npz 数据格式进行适配。
#       修改：正确处理 root_dir, data_path, label_path 以构造完整文件路径。

import os
import numpy as np
import pickle
import random
import math
import logging
import sys # 保留，以备测试块使用或将来可能的路径操作
import torch
from torch.utils.data import Dataset, DataLoader # 确保 DataLoader 在这里导入
from tqdm import tqdm

# --- 尝试导入项目根目录下的 utils ---
try:
    # 假设 feeders 目录在项目的子目录中，utils.py 在项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils import collate_fn_filter_none
    # logger_utils_test = logging.getLogger(__name__) # 如果只在测试块用，可以移到那里
    # logger_utils_test.info("成功从项目根目录的 utils.py 导入 collate_fn_filter_none。")
except ImportError as e_import:
    # logger_utils_test = logging.getLogger(__name__)
    logging.warning(f"无法从项目根目录的 utils.py 导入 collate_fn_filter_none (错误: {e_import})。"
                              "如果直接运行此文件进行测试，DataLoader 可能无法正确工作。")
    # 如果希望测试块能独立运行，这里可以定义一个临时的 collate_fn_filter_none
    def collate_fn_filter_none(batch): # pragma: no cover
        batch = [item for item in batch if item is not None]
        if not batch: return None
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

logger = logging.getLogger(__name__)

NTU_PAIRS_TDGCN_STYLE = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (24,25)
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

    # 增加对输入维度的严格检查
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
    return bone_data_tvc

def calculate_motion_data(data_tvc):
    motion_data_tvc = np.zeros_like(data_tvc)
    T = data_tvc.shape[0]
    if T > 1:
        motion_data_tvc[:-1, :, :] = data_tvc[1:, :, :] - data_tvc[:-1, :, :]
        motion_data_tvc[-1, :, :] = motion_data_tvc[-2, :, :] 
    return motion_data_tvc


class Feeder_NTU(Dataset):
    def __init__(self,
                 root_dir,                  # <<--- 接收 root_dir
                 data_path,                 # .npz 数据文件名 (相对于 root_dir)
                 label_path,                # .npz 标签文件名 (相对于 root_dir)
                 split='train',
                 max_len=150,
                 modalities="joint",
                 num_nodes=25,
                 base_channel=3,
                 num_classes=60,
                 random_choose=False,
                 random_rot=False,
                 center_joint_idx=0,
                 debug=False,
                 label_source='from_label_path', # 新增：'from_label_path' 或 'from_data_npz'
                 **kwargs):
        super().__init__()

        self.root_dir = root_dir
        # 将传入的 data_path 和 label_path 视为相对于 root_dir 的文件名或子路径
        self.actual_data_path = os.path.join(self.root_dir, data_path)
        self.actual_label_path = os.path.join(self.root_dir, label_path) # 即使label_source是from_data_npz，这个也先构造
        
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
        self.base_channel = base_channel
        self.num_classes = num_classes
        self.num_input_dim = self.base_channel * len(self.modalities)

        self.random_choose = random_choose if self.split == 'train' else False
        self.random_rot = random_rot if self.split == 'train' else False
        self.center_joint_idx = center_joint_idx 
        self.debug = debug
        
        self.bone_pairs = NTU_PAIRS_TDGCN_STYLE

        # 在 __init__ 的末尾调用 load_data
        self.load_data()

        logger.info(f"Feeder_NTU ({self.split}集) 初始化完成:")
        logger.info(f"  实际数据文件路径: {self.actual_data_path}")
        if self.label_source == 'from_label_path':
            logger.info(f"  实际标签文件路径: {self.actual_label_path}")
        else:
            logger.info(f"  标签将从数据文件 ({self.actual_data_path}) 中加载。")
        logger.info(f"  目标序列长度: {self.target_seq_len}")
        logger.info(f"  加载模态: {self.modalities} (总输入通道数: {self.num_input_dim})")
        logger.info(f"  样本数量: {len(self.labels)}")
        if self.debug:
            logger.warning(f"  DEBUG模式开启，只使用前 {len(self.skeletons_data)} 个样本。")


    def load_data(self):
        logger.info(f"开始从 .npz 文件加载数据和标签 ({self.split}集)...")
        
        if not os.path.exists(self.actual_data_path): # 使用拼接后的路径检查
            raise FileNotFoundError(f"数据文件未找到: {self.actual_data_path}")

        data_npz = np.load(self.actual_data_path)
        data_key = 'x_train' if self.split == 'train' else 'x_test'
        if data_key not in data_npz:
            raise KeyError(f"数据 .npz 文件 ('{self.actual_data_path}') 中未找到键 '{data_key}'")
        
        raw_data_flat = data_npz[data_key] 
        N_samples, T_orig_from_file, Features_flat = raw_data_flat.shape

        expected_features_per_person = self.num_nodes * self.base_channel
        if Features_flat % expected_features_per_person != 0:
            raise ValueError(f"NPZ中的特征维度 ({Features_flat}) 不能被 (节点数*基础通道数={expected_features_per_person}) 整除。请检查数据格式。")
        M_in_file = Features_flat // expected_features_per_person
        
        logger.info(f"  从数据文件推断出原始帧长 T_orig={T_orig_from_file}, 最大人数 M_in_file={M_in_file}.")

        try:
            self.skeletons_data = raw_data_flat.reshape(N_samples, T_orig_from_file, M_in_file, self.num_nodes, self.base_channel)
            self.skeletons_data = self.skeletons_data.transpose(0, 2, 1, 3, 4) # N, M, T, V, C
        except ValueError as e:
            logger.error(f"Reshape 原始数据失败。原始形状: {raw_data_flat.shape}, 目标 M={M_in_file}, V={self.num_nodes}, C={self.base_channel}. 错误: {e}")
            raise

        # 加载标签数据
        if self.label_source == 'from_data_npz':
            label_npz_source = data_npz # 从同一个已加载的 data_npz 获取标签
            source_path_for_log = self.actual_data_path
        else: # from_label_path (默认)
            if not os.path.exists(self.actual_label_path): # 使用拼接后的路径检查
                raise FileNotFoundError(f"标签文件未找到: {self.actual_label_path}")
            label_npz_source = np.load(self.actual_label_path)
            source_path_for_log = self.actual_label_path
            
        label_key = 'y_train' if self.split == 'train' else 'y_test'
        if label_key not in label_npz_source:
            if 'arr_0' in label_npz_source: label_key = 'arr_0'
            elif 'labels' in label_npz_source: label_key = 'labels'
            else: raise KeyError(f"标签源 ('{source_path_for_log}') 中未找到期望的标签键 (尝试过 'y_train'/'y_test', 'arr_0', 'labels')")

        raw_labels = label_npz_source[label_key]
        if raw_labels.ndim == 2 and raw_labels.shape[1] > 1: 
            self.labels = np.argmax(raw_labels, axis=1).astype(int)
            logger.info(f"  标签从one-hot编码转换 (源: {source_path_for_log}, 形状: {raw_labels.shape} -> {self.labels.shape})。")
        elif raw_labels.ndim == 1: 
            self.labels = raw_labels.astype(int)
            logger.info(f"  直接加载类别索引标签 (源: {source_path_for_log}, 形状: {self.labels.shape})。")
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
        if T_orig == 0: 
            return np.zeros((0, self.num_nodes, self.base_channel), dtype=np.float32)
        if T_orig <= target_len: 
            return joint_data_tvc
        else: 
            if self.random_choose: 
                start_idx = random.randint(0, T_orig - target_len)
                return joint_data_tvc[start_idx : start_idx + target_len, :, :]
            else: 
                start_idx = (T_orig - target_len) // 2
                return joint_data_tvc[start_idx : start_idx + target_len, :, :]

    def _random_rotation_augmentation(self, joint_data_tvc, angle_limit_degrees=10): # 添加参数
        if joint_data_tvc.shape[2] != 3: 
            return joint_data_tvc
        theta_x = np.random.uniform(-angle_limit_degrees, angle_limit_degrees) * (math.pi / 180.0)
        theta_y = np.random.uniform(-angle_limit_degrees, angle_limit_degrees) * (math.pi / 180.0)
        theta_z = np.random.uniform(-angle_limit_degrees, angle_limit_degrees) * (math.pi / 180.0)
        Rx = np.array([[1,0,0],[0,math.cos(theta_x),-math.sin(theta_x)],[0,math.sin(theta_x),math.cos(theta_x)]])
        Ry = np.array([[math.cos(theta_y),0,math.sin(theta_y)],[0,1,0],[-math.sin(theta_y),0,math.cos(theta_y)]])
        Rz = np.array([[math.cos(theta_z),-math.sin(theta_z),0],[math.sin(theta_z),math.cos(theta_z),0],[0,0,1]])
        R_combined = np.dot(Rz, np.dot(Ry, Rx))
        original_shape = joint_data_tvc.shape
        joint_data_flat = joint_data_tvc.reshape(-1, 3)
        rotated_data_flat = np.dot(joint_data_flat, R_combined.T) 
        return rotated_data_flat.reshape(original_shape).astype(joint_data_tvc.dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label_0based = self.labels[index]
        if not (0 <= label_0based < self.num_classes):
            logger.error(f"样本 {index}: 标签 {label_0based} 超出有效范围 [0, {self.num_classes-1}]。跳过此样本。")
            return None 

        skeletons_for_sample = self.skeletons_data[index]
        selected_joint_data_tvc = None
        if skeletons_for_sample.shape[0] > 0: 
            selected_joint_data_tvc = skeletons_for_sample[0, :, :, :].copy() 
            if np.sum(np.abs(selected_joint_data_tvc)) < 1e-6:
                 if skeletons_for_sample.shape[0] > 1 and np.sum(np.abs(skeletons_for_sample[1,:,:,:])) > 1e-6 :
                     selected_joint_data_tvc = skeletons_for_sample[1, :, :, :].copy()
                 else:
                     logger.warning(f"样本 {index}: 有效骨骼数据全为零。将使用全零数据。")
                     # 使用 self.target_seq_len 而不是 selected_joint_data_tvc.shape[0] (可能为0)
                     selected_joint_data_tvc = np.zeros((self.target_seq_len if selected_joint_data_tvc.shape[0] == 0 else selected_joint_data_tvc.shape[0], 
                                                        self.num_nodes, self.base_channel), dtype=np.float32)
        else: 
            logger.warning(f"样本 {index}: 无有效骨骼数据。将使用全零数据。")
            t_for_dummy = self.skeletons_data.shape[2] if self.skeletons_data.ndim == 5 and self.skeletons_data.shape[2] > 0 else self.target_seq_len
            selected_joint_data_tvc = np.zeros((t_for_dummy, self.num_nodes, self.base_channel), dtype=np.float32)

        try:
            if self.center_joint_idx is not None and 0 <= self.center_joint_idx < self.num_nodes:
                if selected_joint_data_tvc.shape[0] > 0: 
                    center_coord = selected_joint_data_tvc[0, self.center_joint_idx, :].copy()
                    selected_joint_data_tvc = selected_joint_data_tvc - center_coord
            
            joint_data_sampled_tvc = self._temporal_sampling(selected_joint_data_tvc, self.target_seq_len)
            if joint_data_sampled_tvc.shape[0] == 0 and self.target_seq_len > 0:
                pass 

            if self.random_rot: 
                # 假设你在 __init__ 中添加了 self.rotation_angle_degrees 参数
                angle_limit = getattr(self, 'rotation_angle_degrees', 10) 
                joint_data_sampled_tvc = self._random_rotation_augmentation(joint_data_sampled_tvc, angle_limit_degrees=angle_limit)

            modal_data_list_tvc = [] 
            if joint_data_sampled_tvc.ndim != 3 or joint_data_sampled_tvc.shape[1] != self.num_nodes or joint_data_sampled_tvc.shape[2] != self.base_channel:
                logger.error(f"样本 {index}: 采样/增强后的关节数据维度不正确: {joint_data_sampled_tvc.shape}。跳过。")
                return None

            bone_data_xyz_cache = None
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
            
            data_concatenated_tvc = np.concatenate(modal_data_list_tvc, axis=-1)
            if data_concatenated_tvc.shape[-1] != self.num_input_dim:
                logger.error(f"样本 {index}: 拼接后的通道数 ({data_concatenated_tvc.shape[-1]}) 与预期的总输入维度 ({self.num_input_dim}) 不符。跳过。")
                return None

            data_padded_tvc, mask_np = pad_and_mask_sequence(data_concatenated_tvc, 
                                                             self.target_seq_len, 
                                                             self.num_nodes, 
                                                             self.num_input_dim)

            data_tensor = torch.from_numpy(data_padded_tvc).float()
            label_tensor = torch.tensor(label_0based, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool()

            return data_tensor, label_tensor, mask_tensor, index
        except Exception as e:
            logger.error(f"处理样本 {index} (标签: {label_0based}) 时发生未知错误: {e}", exc_info=True)
            return None

# --- 用于独立测试 Feeder_NTU 的代码块 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    dummy_npz_file = 'dummy_ntu_data_for_feeder_test.npz'
    if not os.path.exists(dummy_npz_file):
        logger.info(f"创建虚拟 .npz 文件: {dummy_npz_file} 用于测试...")
        _N, _T, _M, _V, _C = 20, 70, 2, 25, 3 
        _x_train = np.random.rand(_N, _T, _M * _V * _C).astype(np.float32)
        _y_train_onehot = np.zeros((_N, 60), dtype=np.float32)
        _y_train_idx = np.random.randint(0, 60, size=_N)
        _y_train_onehot[np.arange(_N), _y_train_idx] = 1.0
        _x_test = np.random.rand(_N // 2, _T, _M * _V * _C).astype(np.float32)
        _y_test_onehot = np.zeros((_N // 2, 60), dtype=np.float32)
        _y_test_idx = np.random.randint(0, 60, size=(_N // 2))
        _y_test_onehot[np.arange(_N // 2), _y_test_idx] = 1.0
        np.savez(dummy_npz_file, x_train=_x_train, y_train=_y_train_onehot, x_test=_x_test, y_test=_y_test_onehot)
    
    DUMMY_ROOT_DIR = '.' # 假设 dummy_ntu_data_for_feeder_test.npz 在当前目录
    DUMMY_DATA_FILENAME = dummy_npz_file 
    DUMMY_LABEL_FILENAME = dummy_npz_file

    logger.info("\n--- 测试 Feeder_NTU (使用虚拟数据) ---")
    try:
        train_feeder_params = {
            'root_dir': DUMMY_ROOT_DIR,             # <<--- 提供 root_dir
            'data_path': DUMMY_DATA_FILENAME,      # <<--- 提供文件名
            'label_path': DUMMY_LABEL_FILENAME,    # <<--- 提供文件名
            'label_source': 'from_data_npz',       # <<--- 测试从数据npz加载标签
            'split': 'train',
            'max_len': 100,
            'modalities': "joint,bone", 
            'num_classes': 60,
            'num_nodes': 25,
            'base_channel': 3,
            'random_choose': True,
            'random_rot': True, 
            'center_joint_idx': 0,
            'debug': False 
        }
        feeder_train = Feeder_NTU(**train_feeder_params)
        logger.info(f"虚拟训练集 Feeder 长度: {len(feeder_train)}")

        if len(feeder_train) > 0:
            item_result = feeder_train[0] 
            if item_result is not None:
                data, label, mask, idx = item_result
                logger.info(f"虚拟训练集样本 {idx}: 数据形状 {data.shape}, 标签 {label.item()}, Mask形状 {mask.shape}, Mask有效帧数 {mask.sum().item()}")
                expected_channels = len(train_feeder_params['modalities'].split(',')) * train_feeder_params['base_channel']
                assert data.shape == (train_feeder_params['max_len'], train_feeder_params['num_nodes'], expected_channels)
                assert mask.shape == (train_feeder_params['max_len'],)
                logger.info("第一个训练样本形状和掩码校验通过。")
            else:
                logger.warning("获取的第一个训练样本为 None。")

            if 'collate_fn_filter_none' in globals() and callable(globals()['collate_fn_filter_none']):
                collate_fn_to_use_test = globals()['collate_fn_filter_none']
                train_loader = DataLoader(feeder_train, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_to_use_test)
                first_batch = next(iter(train_loader), None)
                if first_batch:
                    xb, lb, mb, idxb = first_batch
                    logger.info(f"虚拟训练集 DataLoader 第一个批次: X形状 {xb.shape}, L形状 {lb.shape}, M形状 {mb.shape}")
                else:
                    logger.warning("虚拟训练集 DataLoader 返回的第一个批次为空。")
            else:
                logger.warning("无法找到 collate_fn_filter_none，跳过 DataLoader 测试。")
        else:
            logger.warning("虚拟训练集为空。")
    except Exception as e:
        logger.error(f"测试 Feeder_NTU (训练集) 时出错: {e}", exc_info=True)