# -*- coding: utf-8 -*-
# 文件名: feeders/feeder_ucla.py (v1.2 - 支持多模态拼接和类别相关增强)

import torch
import pickle
import numpy as np
import random
import math
import json # <--- 确保导入 json 模块
import os
import glob
import sys
import traceback
import logging
from torch.utils.data import Dataset, DataLoader

# 获取 logger 实例
logger = logging.getLogger(__name__)

# --- 辅助函数 ---

def pad_sequence(seq, max_len, pad_value=0.0):
    """将序列填充/截断到指定长度，并返回序列和掩码"""
    if seq is None or seq.size == 0: # 使用 .size 检查 numpy 数组是否为空
        # 尝试获取维度信息，如果不可能，使用默认值
        # 注意: 这里的默认通道数需要谨慎设置，最好能从外部获取或推断
        num_nodes = 20 # NW-UCLA 默认值
        num_channels = 3 # 基础维度
        # logger.warning(f"pad_sequence 收到空序列或 None，返回全零填充 (默认维度 {num_nodes}x{num_channels})。")
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask

    seq_len = seq.shape[0]
    # 处理 seq_len 可能为 0 的情况
    if seq_len == 0:
        # logger.warning(f"pad_sequence 收到长度为 0 的序列，返回全零填充。")
        num_nodes = seq.shape[1] if seq.ndim > 1 else 20
        num_channels = seq.shape[2] if seq.ndim > 2 else 3 # 获取实际通道数
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask

    num_nodes = seq.shape[1]
    num_channels = seq.shape[2] # 获取实际通道数

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

def joint_to_bone(data_numpy, bone_pairs, num_nodes):
    """根据关节坐标计算骨骼向量"""
    T, N, C = data_numpy.shape
    data_bone = np.zeros_like(data_numpy)
    for v1, v2 in bone_pairs:
        idx1, idx2 = v1 - 1, v2 - 1
        if 0 <= idx1 < num_nodes and 0 <= idx2 < num_nodes:
            data_bone[:, idx1, :] = data_numpy[:, idx1, :] - data_numpy[:, idx2, :]
        else:
             pass # 减少日志噪音
    return data_bone

def joint_to_motion(data_numpy):
    """计算帧间运动向量（差分）"""
    data_motion = np.zeros_like(data_numpy)
    T = data_numpy.shape[0]
    if T > 1:
        data_motion[:T-1, :, :] = data_numpy[1:, :, :] - data_numpy[:-1, :, :]
        data_motion[T-1, :, :] = data_motion[T-2, :, :]
    return data_motion


class Feeder(Dataset):
    """ NW-UCLA 数据集的 Feeder 类 (支持多模态拼接和类别相关增强) """
    def __init__(self, data_path='joint', label_path=None, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False,
                 root_dir=None, split='train', center_joint_idx=1, apply_rand_view_transform=True,
                 val_pkl_path='data/nw-ucla/val_label.pkl', bone_pairs=None,
                 num_classes=10, max_len=64,
                 # --- 类别相关增强参数 ---
                 augment_confused_classes=False, # 默认关闭
                 confused_classes_list=None, # 默认无特定类别
                 confused_rotation_range=(-75, 75), # 默认增强旋转范围
                 confused_scale_range=(0.4, 1.6),   # 默认增强缩放范围
                 add_gaussian_noise=False,        # 默认不添加高斯噪声
                 gaussian_noise_level=0.01        # 默认噪声水平
                 ):
        super().__init__()
        # --- 参数检查与赋值 ---
        if root_dir is None or not os.path.isdir(root_dir): raise ValueError(f"无效的数据集根目录 'root_dir': {root_dir}")
        if split not in ['train', 'val', 'test']: raise ValueError(f"无效的 split 参数: {split}")

        self.root_dir = root_dir
        self.split = 'train' if split == 'train' else 'val'
        self.train_val = split
        self.repeat = repeat if self.train_val == 'train' else 1
        self.max_len = max_len if max_len > 0 else (window_size if window_size > 0 else 64)
        self.num_nodes = 20
        self.num_base_input_dim = 3 # 基础模态 (joint) 的维度
        self.num_classes = num_classes
        self.center_joint_idx = center_joint_idx if isinstance(center_joint_idx, int) and 0 <= center_joint_idx < self.num_nodes else None
        self.apply_rand_view_transform = apply_rand_view_transform if self.train_val == 'train' else False
        self.random_choose = random_choose if self.train_val == 'train' else False
        self.debug = debug

        # --- 解析多模态 ---
        if isinstance(data_path, str): self.modalities = [m.strip().lower() for m in data_path.split(',') if m.strip()]
        elif isinstance(data_path, list): self.modalities = [m.strip().lower() for m in data_path if isinstance(m, str) and m.strip()]
        else: raise ValueError("data_path 必须是逗号分隔的字符串或字符串列表")
        valid_modalities = ['joint', 'bone', 'joint_motion', 'bone_motion']
        for m in self.modalities:
            if m not in valid_modalities: raise ValueError(f"不支持的数据模态: '{m}'. 支持: {valid_modalities}")
        if not self.modalities: raise ValueError("必须至少指定一种数据模态 (data_path)")
        self.num_input_dim = self.num_base_input_dim * len(self.modalities)

        # --- 骨骼连接对 ---
        if bone_pairs is None: self.bone_pairs = [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
        else: self.bone_pairs = bone_pairs
        self.val_pkl_path = val_pkl_path

        # --- 存储类别相关增强参数 ---
        self.augment_confused_classes = augment_confused_classes if self.train_val == 'train' else False
        self.confused_classes_set = set(confused_classes_list) if confused_classes_list else set()
        self.confused_rotation_range = confused_rotation_range
        self.confused_scale_range = confused_scale_range
        self.add_gaussian_noise = add_gaussian_noise if self.train_val == 'train' else False
        self.gaussian_noise_level = gaussian_noise_level
        if self.augment_confused_classes:
            logger.info(f"训练时将对类别 {list(self.confused_classes_set)} 应用特殊增强 (旋转范围: {self.confused_rotation_range}, 缩放范围: {self.confused_scale_range}, 添加高斯噪声: {self.add_gaussian_noise})")

        # --- 打印初始化信息 ---
        logger.info(f"初始化 Feeder for NW-UCLA: split={self.train_val}, root_dir={self.root_dir}")
        logger.info(f"加载模态: {self.modalities}, 目标序列长度: {self.max_len}")
        logger.info(f"基础输入维度: {self.num_base_input_dim}, 拼接后总维度: {self.num_input_dim}")
        logger.info(f"类别数: {self.num_classes}")
        if self.split == 'val': logger.info(f"验证/测试集 PKL 路径: {self.val_pkl_path}")

        # --- 加载数据 ---
        self.sample_info = [] # 存储 {'path': ..., 'id': ...}
        self.label = []       # 存储 0-based 标签
        self._load_data() # 加载样本信息和标签

        # --- 加载后检查 ---
        if not self.sample_info: raise RuntimeError(f"未能加载任何样本信息 for split '{self.train_val}' in '{self.root_dir}'.")
        if len(self.sample_info) != len(self.label):
             logger.error(f"样本信息 ({len(self.sample_info)}) 和标签 ({len(self.label)}) 数量不匹配！ Split: {self.train_val}")
             raise RuntimeError(f"样本信息 ({len(self.sample_info)}) 和标签 ({len(self.label)}) 数量不匹配！ Split: {self.train_val}")

        if self.debug:
            logger.warning(f"!!! DEBUG 模式开启，只使用前 100 个样本 !!!")
            self.sample_info = self.sample_info[:100]
            self.label = self.label[:100]

        logger.info(f"成功加载 {len(self.sample_info)} 个样本用于 '{self.train_val}' split。")

    def _load_data(self):
        """根据 split 加载数据信息和标签"""
        if self.split == 'train':
            self._load_train_samples_and_labels_from_json()
        else: # 'val' or 'test'
            self._load_val_samples_and_labels_from_pkl()

    def _load_train_samples_and_labels_from_json(self):
        """扫描 root_dir，根据文件名划分视角，从 JSON 文件内部读取标签 (用于训练集)。"""
        logger.info(f"扫描目录 '{self.root_dir}' 并根据视角划分训练样本 (View 1 & 2)，从 JSON 读取标签...")
        json_files_pattern = os.path.join(self.root_dir, 'a*_s*_e*_v*.json')
        json_files = glob.glob(json_files_pattern)
        if not json_files: logger.error(f"在 '{self.root_dir}' 中找不到任何符合模式 'aX*_sX*_eX*_vX*.json' 的 JSON 文件。"); return

        self.sample_info = []
        self.label = []
        loaded_count = 0; skipped_json_error = 0; skipped_label_error = 0
        skipped_view = 0; skipped_parsing = 0; debug_limit = 50

        logger.info(f"找到 {len(json_files)} 个潜在 JSON 文件，开始解析...")
        for filepath in sorted(json_files):
            filename = os.path.basename(filepath)
            sample_id = filename.replace('.json', '')
            try:
                parts = sample_id.split('_'); view_part = parts[-1]
                if len(parts) != 4 or not view_part.startswith('v') or not view_part[1:].isdigit(): raise ValueError("文件名或视角部分格式错误")
                view_id = int(view_part[1:])
            except Exception as e:
                if skipped_parsing < debug_limit: logger.warning(f"[解析失败] 文件名 '{filename}' 无法解析视角ID ({e})，跳过。")
                skipped_parsing += 1; continue

            is_train_sample = view_id == 1 or view_id == 2
            if not is_train_sample:
                if skipped_view < debug_limit: pass # logger.debug(...)
                skipped_view += 1; continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f: json_data = json.load(f)
                label_key = "label"; possible_keys = ["action_id", "action_label", "category_id"]; label_orig = None
                if label_key in json_data: label_orig = json_data[label_key]
                else:
                    for pk in possible_keys:
                        if pk in json_data: label_key = pk; label_orig = json_data[label_key]; break
                if label_orig is None: raise KeyError(f"JSON 文件中找不到标签键 '{label_key}' 或 {possible_keys}")
                if not isinstance(label_orig, int): raise TypeError(f"标签值 '{label_orig}' 不是整数")
                if not (1 <= label_orig <= self.num_classes):
                    if skipped_label_error < debug_limit: logger.warning(f"[跳过 - JSON标签超范围] 文件: {filename}, JSON标签 {label_orig} 超出范围 [1, {self.num_classes}]")
                    skipped_label_error += 1; continue

                final_label = label_orig - 1
                self.sample_info.append({'path': filepath, 'id': sample_id})
                self.label.append(final_label)
                loaded_count += 1
                # if self.debug and loaded_count <= debug_limit: print(...) # 减少打印

            except Exception as e:
                 if skipped_json_error < debug_limit or skipped_label_error < debug_limit : logger.warning(f"[跳过 - JSON/标签错误] 文件: {filename}, 原因: {e}")
                 if isinstance(e, (FileNotFoundError, json.JSONDecodeError)): skipped_json_error += 1
                 else: skipped_label_error += 1
                 continue

        logger.info(f"训练样本扫描完毕。成功加载: {loaded_count}。跳过(JSON错误 {skipped_json_error}, 标签错误 {skipped_label_error}, 非训练视角 {skipped_view}, 解析失败 {skipped_parsing})。")
        if loaded_count == 0: logger.error("错误：没有加载到任何有效的训练样本！")


    def _load_val_samples_and_labels_from_pkl(self):
        """从 pkl 文件加载验证集/测试集样本信息和标签。"""
        logger.info(f"从 pkl 文件加载验证/测试集样本和标签: {self.val_pkl_path}")
        if not os.path.exists(self.val_pkl_path): logger.error(f"PKL 文件未找到: {self.val_pkl_path}"); return
        try:
            with open(self.val_pkl_path, 'rb') as f: pkl_data_raw = pickle.load(f)
            pkl_data = []
            # (保持对不同 PKL 格式的兼容处理)
            if isinstance(pkl_data_raw, list) and pkl_data_raw and isinstance(pkl_data_raw[0], dict) and 'file_name' in pkl_data_raw[0] and 'label' in pkl_data_raw[0]: pkl_data = pkl_data_raw
            elif isinstance(pkl_data_raw, dict):
                 if all(isinstance(k, str) and isinstance(v, int) for k, v in pkl_data_raw.items()): pkl_data = [{'file_name': k, 'label': v} for k, v in pkl_data_raw.items()]
                 elif pkl_data_raw and isinstance(list(pkl_data_raw.values())[0], dict) and 'label' in list(pkl_data_raw.values())[0]: pkl_data = [{'file_name': k, 'label': v['label']} for k, v in pkl_data_raw.items()]
                 else: logger.error(f"无法识别的 PKL 字典格式 in {self.val_pkl_path}"); return
            else: logger.error(f"无法识别的 PKL 文件格式 in {self.val_pkl_path}"); return
            if not pkl_data: logger.error(f"从 PKL 加载/解析后数据为空: {self.val_pkl_path}"); return

            self.sample_info = []; self.label = []
            loaded_count = 0; skipped_count = 0; missing_files = 0; debug_limit = 50
            for item in pkl_data:
                file_name = item.get('file_name'); label_orig = item.get('label')
                if file_name is None or label_orig is None: skipped_count += 1; continue
                if not isinstance(label_orig, int): skipped_count += 1; continue
                if not (1 <= label_orig <= self.num_classes):
                    if skipped_count < 10 or self.debug: logger.warning(f"PKL 标签无效: {label_orig} for {file_name}，跳过！")
                    skipped_count += 1; continue
                json_file_path = os.path.join(self.root_dir, f"{file_name}.json")
                if os.path.exists(json_file_path):
                     self.sample_info.append({'path': json_file_path, 'id': file_name})
                     final_label = label_orig - 1
                     if not (0 <= final_label < self.num_classes): # Double check
                          logger.error(f"内部错误：PKL标签转换后无效 {final_label} for {file_name}，跳过！")
                          self.sample_info.pop(); skipped_count += 1; continue
                     self.label.append(final_label)
                     loaded_count += 1
                     # if self.debug and loaded_count <= debug_limit: print(...) # Reduce print
                else:
                     if missing_files < 10: logger.warning(f"找不到验证/测试集 JSON 文件: {json_file_path}，跳过。")
                     missing_files += 1; skipped_count += 1
            logger.info(f"验证/测试样本加载完毕。加载: {loaded_count}, 跳过: {skipped_count} (缺失 JSON: {missing_files})。")
            if loaded_count == 0: logger.error("错误：没有加载到任何有效的验证/测试样本！")
        except Exception as e: logger.error(f"加载或处理 PKL 文件 '{self.val_pkl_path}' 失败: {e}", exc_info=True)

    def rand_view_transform(self, X, agx, agy, s):
        """应用随机视角变换"""
        agx_rad = math.radians(agx); agy_rad = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx_rad),math.sin(agx_rad)], [0,-math.sin(agx_rad),math.cos(agx_rad)]])
        Ry = np.asarray([[math.cos(agy_rad),0,-math.sin(agy_rad)], [0,1,0], [math.sin(agy_rad),0,math.cos(agy_rad)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        orig_shape = X.shape; C_dim = orig_shape[-1]
        if C_dim != 3: return X
        is_reshaped = False
        if X.ndim > 2 : X = np.reshape(X,(-1,3)); is_reshaped = True
        R = np.dot(Rx, Ry)
        X_transformed = np.dot(X, np.dot(R, Ss))
        if is_reshaped : X_transformed = np.reshape(X_transformed, orig_shape)
        return X_transformed

    def __getitem__(self, index):
        """获取并处理单个样本，返回拼接后的多模态数据和标签"""
        true_index = index % len(self.sample_info)
        if true_index >= len(self.sample_info) or true_index >= len(self.label): return None

        info = self.sample_info[true_index]
        label = self.label[true_index] # 0-based label
        json_file_path = info['path']
        sample_id = info['id']

        # --- 读取 JSON 数据 (获取骨骼坐标) ---
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f: json_data = json.load(f)
            frame_list_from_json = []
            potential_keys = ['frames', 'data', 'skeletons']
            for key in potential_keys:
                if key in json_data and isinstance(json_data[key], list): frame_list_from_json = json_data[key]; break
            if not frame_list_from_json and isinstance(json_data, list): frame_list_from_json = json_data
            if not frame_list_from_json: raise ValueError("JSON 数据列表为空")
        except Exception as e: return None # 让 collate_fn 处理

        # --- 提取基础关节序列 (data_numpy) ---
        skeleton_sequence = []
        for frame_idx, frame_info in enumerate(frame_list_from_json):
            skeletons = []; target_skeleton = None; joints = []
            if isinstance(frame_info, dict):
                skeletons = frame_info.get('skeletons', [])
                if not skeletons and 'joints' in frame_info: skeletons = [frame_info]
            elif isinstance(frame_info, list): skeletons = [{'joints': frame_info}]
            if skeletons: target_skeleton = skeletons[0]; joints = target_skeleton.get('joints', [])
            if not isinstance(joints, list): joints = []
            current_frame_joints_np = None
            if len(joints) == self.num_nodes:
                 frame_joints_list = []; valid_frame = True
                 for joint_idx, joint_info in enumerate(joints):
                      pos = None
                      if isinstance(joint_info, list) and len(joint_info) >= self.num_base_input_dim: pos = joint_info[:self.num_base_input_dim]
                      elif isinstance(joint_info, dict):
                           pos_candidate = joint_info.get('position', joint_info.get('pos'))
                           if isinstance(pos_candidate, list) and len(pos_candidate) >= self.num_base_input_dim: pos = pos_candidate[:self.num_base_input_dim]
                      if pos is None or not all(isinstance(c, (int, float)) and math.isfinite(c) for c in pos): valid_frame = False; break
                      frame_joints_list.append(pos)
                 if valid_frame:
                     try: current_frame_joints_np = np.array(frame_joints_list, dtype=np.float32)
                     except ValueError: valid_frame = False
            if current_frame_joints_np is None or not valid_frame:
                 if skeleton_sequence: skeleton_sequence.append(skeleton_sequence[-1])
                 else: skeleton_sequence.append(np.zeros((self.num_nodes, self.num_base_input_dim), dtype=np.float32))
            else: skeleton_sequence.append(current_frame_joints_np)
        if not skeleton_sequence: return None
        try:
            data_numpy = np.stack(skeleton_sequence, axis=0) # (T_orig, N, C_base)
            if not np.all(np.isfinite(data_numpy)): data_numpy = np.nan_to_num(data_numpy, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(data_numpy)): return None
        except Exception as e: return None

        # --- 数据预处理和增强 (应用在原始 joint 数据上) ---
        # 1. 中心化
        if self.center_joint_idx is not None: # 检查是否为 None
            if data_numpy.shape[0] > 0 and data_numpy.shape[1] > self.center_joint_idx:
                center = data_numpy[:, self.center_joint_idx:self.center_joint_idx+1, :]
                data_numpy = data_numpy - center

        # 2. 应用数据增强 (只在训练时)
        if self.train_val == 'train':
            apply_stronger_aug = self.augment_confused_classes and (label in self.confused_classes_set)

            # 2.a 随机视角变换
            if self.apply_rand_view_transform:
                if apply_stronger_aug: # 对特定类别应用更强的变换
                    rot_range = self.confused_rotation_range; scale_range = self.confused_scale_range
                else: # 默认变换范围
                    rot_range = (-60, 60); scale_range = (0.5, 1.5)
                agx = random.randint(rot_range[0], rot_range[1])
                agy = random.randint(rot_range[0], rot_range[1])
                s = random.uniform(scale_range[0], scale_range[1])
                if data_numpy.shape[0] > 0 and data_numpy.shape[-1] == 3:
                    data_numpy = self.rand_view_transform(data_numpy, agx, agy, s)

            # 2.b 对易混淆类别添加高斯噪声 (可选)
            if apply_stronger_aug and self.add_gaussian_noise and self.gaussian_noise_level > 0:
                noise = np.random.normal(scale=self.gaussian_noise_level, size=data_numpy.shape)
                data_numpy = data_numpy + noise.astype(data_numpy.dtype)

            # 2.c 在这里可以添加其他训练时增强, e.g., Joint Dropout
            # if apply_stronger_aug and self.apply_joint_dropout:
            #    data_numpy = self.random_joint_dropout(data_numpy, p=0.1)

        # --- 根据请求的模态计算并拼接 ---
        modal_data_list = []
        data_bone = None # 缓存 bone
        try:
            for modality in self.modalities:
                if modality == 'joint': modal_data_list.append(data_numpy.copy())
                elif modality == 'bone':
                    if data_bone is None: data_bone = joint_to_bone(data_numpy, self.bone_pairs, self.num_nodes)
                    modal_data_list.append(data_bone.copy())
                elif modality == 'joint_motion': modal_data_list.append(joint_to_motion(data_numpy))
                elif modality == 'bone_motion':
                    if data_bone is None: data_bone = joint_to_bone(data_numpy, self.bone_pairs, self.num_nodes)
                    modal_data_list.append(joint_to_motion(data_bone))
            if not modal_data_list: return None # 没有有效模态
            data_concatenated = np.concatenate(modal_data_list, axis=-1) # (T_orig, N, C_total)
        except Exception as e: # 捕捉计算或拼接中的错误
            # logger.error(f"样本 {sample_id}: 计算或拼接模态时出错: {e}")
            return None

        # --- 时间步采样/插值 和 Padding ---
        data_sampled = self.temporal_windowing(data_concatenated, sample_id)
        if data_sampled is None or data_sampled.size == 0: return None

        # *** 确保 pad_sequence 处理的是正确维度的数组 ***
        data_padded, mask_np = pad_sequence(data_sampled, self.max_len)

        # --- 转换为 Tensor ---
        try:
            x_tensor = torch.from_numpy(data_padded).float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask_np).bool()
        except Exception as e: return None

        return x_tensor, label_tensor, mask_tensor, true_index


    def temporal_windowing(self, data_numpy, sample_id_for_debug):
        """处理时间维度：采样或保持原样，交由 pad_sequence 处理最终长度"""
        if data_numpy is None or data_numpy.shape[0] == 0: return None # 处理空输入
        T_orig = data_numpy.shape[0]
        target_len = self.max_len
        if target_len <= 0: return data_numpy
        if T_orig == target_len: return data_numpy
        elif T_orig < target_len: return data_numpy # 由 pad_sequence 填充
        else: # T_orig > target_len
            if self.train_val == 'train' and self.random_choose:
                indices_pool = list(np.arange(T_orig))
                k = min(target_len, T_orig)
                if k <= 0: return None
                random_idx = random.sample(indices_pool, k); random_idx.sort()
                return data_numpy[random_idx, :, :]
            else:
                idx = np.linspace(0, T_orig - 1, target_len).round().astype(int)
                idx = np.clip(idx, 0, T_orig - 1)
                return data_numpy[idx, :, :]

    def __len__(self):
        return len(self.sample_info) * self.repeat

    def top_k(self, score, top_k):
        pass # 通常在 Processor 中计算

# (单元测试代码保持不变，或根据需要更新)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("测试 Feeder (支持多模态拼接和类别相关增强)...")

    # --- !!! 修改为你本地 NW-UCLA 数据集的实际根目录和 PKL 文件路径 !!! ---
    test_root_dir = '/path/to/your/nw-ucla/data-json' # <--- 修改 JSON 文件所在目录
    test_val_pkl = '/path/to/your/nw-ucla/val_label.pkl' # <--- 修改验证集 PKL 文件路径

    if not os.path.isdir(test_root_dir):
         logger.error(f"测试根目录无效: {test_root_dir}"); sys.exit(1)
    # if not os.path.exists(test_val_pkl): logger.warning(f"测试 PKL 文件无效: {test_val_pkl}") # 只警告

    def collate_fn_filter_none(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        try:
             xs = torch.stack([item[0] for item in batch], 0)
             ls = torch.stack([item[1] for item in batch], 0)
             ms = torch.stack([item[2] for item in batch], 0)
             idxs = torch.tensor([item[3] for item in batch], dtype=torch.long)
             return xs, ls, ms, idxs
        except Exception as e:
             logger.error(f"CollateFn 错误: {e}")
             for i, item in enumerate(batch):
                 try: logger.error(f" Item {i} shapes: {item[0].shape}, {item[1].shape}, {item[2].shape}, {type(item[3])}")
                 except: logger.error(f" Item {i} is problematic: {item}")
             return None

    try:
        # --- 测试训练集加载 (joint, bone) ---
        logger.info("\n--- 测试训练集 (joint, bone) ---")
        train_feeder_jb_args = {
            'root_dir': test_root_dir, 'split': 'train', 'data_path': 'joint,bone',
            'max_len': 100, 'num_classes': 10, 'debug': True, 'apply_rand_view_transform': True
        }
        train_feeder_jb = Feeder(**train_feeder_jb_args)
        logger.info(f"训练集 (joint, bone) 样本数: {len(train_feeder_jb)}")
        logger.info(f"预期输入维度: {train_feeder_jb.num_input_dim}")
        if len(train_feeder_jb.sample_info) > 0:
            item1 = train_feeder_jb[0]
            if item1:
                 x1, l1, m1, idx1 = item1
                 logger.info(f"样本 {idx1} - X shape: {x1.shape}, Label: {l1.item()}, Mask sum: {m1.sum().item()}")
                 assert x1.shape == (100, 20, 6)
            else: logger.warning("第一个样本加载失败")
            train_loader_jb = DataLoader(train_feeder_jb, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_filter_none)
            batch1 = next(iter(train_loader_jb), None)
            if batch1:
                xb, lb, mb, idxb = batch1
                logger.info(f"Batch X shape: {xb.shape}, Label shape: {lb.shape}, Mask shape: {mb.shape}")
                assert xb.shape == (4, 100, 20, 6)
            else: logger.warning("批次为空")

        # --- 测试类别相关增强 ---
        logger.info("\n--- 测试类别相关增强 ---")
        confused_args = {
            'root_dir': test_root_dir, 'split': 'train', 'data_path': 'joint', 'max_len': 64,
            'num_classes': 10, 'debug': True, 'apply_rand_view_transform': True,
            'augment_confused_classes': True, 'confused_classes_list': [0, 8],
            'confused_rotation_range': (-90, 90), 'confused_scale_range': (0.1, 2.0),
            'add_gaussian_noise': True, 'gaussian_noise_level': 0.1
        }
        feeder_aug = Feeder(**confused_args)
        logger.info(f"类别增强 Feeder 样本数: {len(feeder_aug)}")
        if len(feeder_aug.sample_info) > 0:
             idx_cls0, idx_cls8, idx_other = -1, -1, -1
             for i in range(len(feeder_aug.label)):
                 lbl = feeder_aug.label[i]
                 if lbl == 0 and idx_cls0 == -1: idx_cls0 = i
                 elif lbl == 8 and idx_cls8 == -1: idx_cls8 = i
                 elif lbl not in [0, 8] and idx_other == -1: idx_other = i
                 if idx_cls0 != -1 and idx_cls8 != -1 and idx_other != -1: break
             print(f"找到用于测试增强的索引: cls0={idx_cls0}, cls8={idx_cls8}, other={idx_other}")
             if idx_cls0 != -1: item0 = feeder_aug[idx_cls0]; print(f" 类别 0 样本加载成功") if item0 else print(" 类别 0 样本加载失败")
             if idx_cls8 != -1: item8 = feeder_aug[idx_cls8]; print(f" 类别 8 样本加载成功") if item8 else print(" 类别 8 样本加载失败")
             if idx_other != -1: item_other = feeder_aug[idx_other]; print(f" 其他类别 ( {feeder_aug.label[idx_other]} ) 样本加载成功") if item_other else print(" 其他类别样本加载失败")

    except Exception as e:
         logger.error(f"单元测试过程中出错: {e}", exc_info=True)