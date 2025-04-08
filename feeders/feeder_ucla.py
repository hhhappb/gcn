# 文件名: feeders/feeder_ucla.py (修正版 v4 - 完整代码，包含导入)
import torch
from torch.utils.data import Dataset, DataLoader # <--- 导入 DataLoader
import json
import os
import glob
import numpy as np
import random
import math
from tqdm import tqdm
import logging
import traceback # <--- 导入 traceback

logger = logging.getLogger(__name__) # 创建 logger

# --- 填充函数 ---
def pad_sequence(seq, max_len, pad_value=0.0):
    """将序列填充/截断到指定长度"""
    seq_len = seq.shape[0]
    if seq_len == 0: # 处理空序列的情况
        num_nodes = 20 # 假设 NW-UCLA 默认值
        num_channels = 3 # 假设 3D 坐标
        logger.warning(f"遇到长度为 0 的序列，将返回全零填充和全 False 掩码。")
        padded_seq = np.full((max_len, num_nodes, num_channels), pad_value, dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        return padded_seq, mask

    num_nodes = seq.shape[1]
    num_channels = seq.shape[2]

    if seq_len < max_len:
        pad_len = max_len - seq_len
        padding = np.full((pad_len, num_nodes, num_channels), pad_value, dtype=seq.dtype)
        padded_seq = np.concatenate([seq, padding], axis=0)
        mask = np.concatenate([np.ones(seq_len, dtype=bool), np.zeros(pad_len, dtype=bool)], axis=0)
    else: # 长度大于等于 max_len
        padded_seq = seq[:max_len, :, :]
        mask = np.ones(max_len, dtype=bool)
    return padded_seq, mask

class Feeder(Dataset):
    """
    适用于 NW-UCLA 数据集的 PyTorch Dataset 类。
    内置 Cross-View 划分逻辑 (v1, v2 训练, v3 测试/验证)。
    """
    def __init__(self, data_path, label_path=None, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 # --- 核心参数 ---
                 root_dir=None,       # 数据集 JSON 文件所在的根目录 (必需)
                 split='train',       # 明确指定要加载哪个 split ('train' 或 'val'/'test')
                 center_joint_idx=1,  # 用于中心化的关节点索引 (0-based)
                 apply_normalization=False, # 是否应用内部 MinMax 归一化 (建议 False)
                 apply_rand_view_transform=True # 训练时是否应用随机视角变换
                 ):
        """
        初始化数据集。
        Args:
            data_path (str): 特征类型 ('joint', 'bone', 'motion')。
            label_path (str, optional): 不再直接用于区分 split。
            repeat (int): 训练时重复数据集的次数。
            random_choose (bool): 训练时是否随机采样时间步。
            window_size (int): 目标序列长度 (即 max_len)。
            root_dir (str): 包含 JSON 文件的目录路径 (必需)。
            split (str): 'train' 或 'val'/'test' (必需)。
            center_joint_idx (int): 中心化时使用的关节点索引。
            apply_normalization (bool): 是否在内部应用 Min-Max 归一化。
            apply_rand_view_transform (bool): 训练时是否应用随机旋转/缩放。
            # 其他参数保留以兼容旧配置，但部分可能未使用
        """
        super().__init__()

        if root_dir is None or not os.path.isdir(root_dir):
            raise ValueError(f"'root_dir' 参数必须提供且为有效目录路径，当前为: {root_dir}")
        if split not in ['train', 'val', 'test']:
             raise ValueError(f"无效的 'split' 参数: {split}。应为 'train', 'val', 或 'test'。")

        self.root_dir = root_dir
        self.split = 'train' if split == 'train' else 'val' # 将 'test' 也视为 'val' 处理
        self.repeat = repeat if self.split == 'train' else 1

        # --- 从参数获取配置 ---
        self.max_len = window_size if window_size > 0 else 52
        self.num_nodes = 20
        self.num_input_dim = 3
        self.center_joint_idx = center_joint_idx
        self.apply_normalization = apply_normalization
        self.apply_rand_view_transform = apply_rand_view_transform if self.split == 'train' else False
        self.random_choose = random_choose if self.split == 'train' else False

        self.debug = debug
        self.data_path_flag = data_path
        self.bone_pairs = [(0, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                           (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18)]

        logger.info(f"加载 NW-UCLA 数据集, split: {self.split}, root_dir: {self.root_dir}")
        logger.info(f"目标序列长度 (max_len/window_size): {self.max_len}")
        logger.info(f"特征类型: {self.data_path_flag}")
        logger.info(f"训练时随机采样时间步: {self.random_choose}")
        logger.info(f"训练时随机视角变换: {self.apply_rand_view_transform}")
        logger.info(f"内部 MinMax 归一化: {self.apply_normalization}")
        logger.info(f"中心化关节点索引: {self.center_joint_idx}")

        # --- 内置划分逻辑: 加载样本列表 ---
        self.sample_info = []
        self._load_samples_internal()

        if not self.sample_info:
            raise RuntimeError(f"在目录 '{self.root_dir}' 中没有为 '{self.split}' split 找到有效的样本。")

        if self.debug: self.sample_info = self.sample_info[:100]
        logger.info(f"共加载 {len(self.sample_info)} 个样本用于 '{self.split}' split。")


    def _load_samples_internal(self):
        """扫描 root_dir，根据文件名内置 Cross-View 划分逻辑。"""
        logger.info(f"扫描目录 '{self.root_dir}' 并根据视角划分样本...")
        # 使用 glob 查找所有匹配模式的 JSON 文件
        json_files_pattern = os.path.join(self.root_dir, 'a*_s*_e*_v*.json')
        json_files = glob.glob(json_files_pattern)

        if not json_files:
             logger.error(f"在 '{self.root_dir}' 中找不到任何匹配 '{os.path.basename(json_files_pattern)}' 模式的文件。")
             return

        missing_labels = 0
        unknown_views = 0

        for filepath in tqdm(json_files, desc=f"处理 {self.split} 样本", leave=False):
            filename = os.path.basename(filepath)
            sample_id = filename.replace('.json', '')

            try:
                parts = sample_id.split('_')
                action_part = parts[0]
                view_part = parts[-1]
                action_id = int(action_part[1:]) - 1 # aXX -> XX-1
                view_id = int(view_part[1:])       # vVV -> VV
                label = action_id
                # 可选：检查标签范围
                # num_classes = 10 # NW-UCLA
                # if not (0 <= label < num_classes): raise ValueError("标签超出范围")
            except (IndexError, ValueError):
                logger.warning(f"无法从文件名 '{filename}' 解析动作或视角信息，跳过。")
                missing_labels += 1
                continue

            # --- 根据 split 和 view_id 决定是否添加样本 ---
            is_train_sample = view_id == 1 or view_id == 2
            is_val_sample = view_id == 3

            if self.split == 'train' and is_train_sample:
                self.sample_info.append({'path': filepath, 'label': label, 'id': sample_id})
            elif self.split == 'val' and is_val_sample: # 'val' 和 'test' 都加载 view 3
                self.sample_info.append({'path': filepath, 'label': label, 'id': sample_id})
            elif view_id not in [1, 2, 3]: # 记录未知视角
                 logger.warning(f"文件名 '{filename}' 包含未知视角 ID '{view_id}'，跳过。")
                 unknown_views += 1

        if missing_labels > 0: logger.warning(f"共跳过 {missing_labels} 个无法解析标签/视角的样本。")
        if unknown_views > 0: logger.warning(f"共跳过 {unknown_views} 个包含未知视角的样本。")


    def __len__(self):
        """返回数据集的样本数量 (考虑 repeat)"""
        return len(self.sample_info) * self.repeat


    def rand_view_transform(self, X, agx, agy, s):
        """应用随机旋转和缩放"""
        agx = math.radians(agx); agy = math.radians(agy)
        Rx = np.array([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0,-math.sin(agx),math.cos(agx)]])
        Ry = np.array([[math.cos(agy),0,-math.sin(agy)], [0,1,0], [math.sin(agy),0,math.cos(agy)]])
        transform_matrix = Ry @ Rx
        if s != 1.0: transform_matrix = np.diag([s]*3) @ transform_matrix
        X_flat = X.reshape(-1, X.shape[-1])
        X_transformed = X_flat @ transform_matrix.T
        return X_transformed.reshape(X.shape)


    def __getitem__(self, index):
        """获取并处理单个数据样本"""
        true_index = index % len(self.sample_info) # 处理 repeat
        info = self.sample_info[true_index]
        json_file_path = info['path']
        label = info['label']
        sample_id = info['id']

        # --- 读取 JSON 数据 ---
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f: json_data = json.load(f) # 指定编码
            frames_data = json_data.get('frames', [])
            if not frames_data: raise ValueError("JSON 文件中 'frames' 数据为空。")
        except Exception as e:
            logger.error(f"样本 {sample_id}: 读取/解析 JSON {json_file_path} 失败: {e}")
            return None # 返回 None，由 collate_fn 过滤

        # --- 提取骨骼序列 ---
        skeleton_sequence = []
        for frame_idx, frame in enumerate(frames_data):
            skeletons = frame.get('skeletons', [])
            if not skeletons:
                if skeleton_sequence: skeleton_sequence.append(skeleton_sequence[-1])
                else: skeleton_sequence.append(np.zeros((self.num_nodes, self.num_input_dim), dtype=np.float32))
                continue
            target_skeleton = skeletons[0]
            joints = target_skeleton.get('joints', [])
            if len(joints) != self.num_nodes:
                if skeleton_sequence: skeleton_sequence.append(skeleton_sequence[-1])
                else: skeleton_sequence.append(np.zeros((self.num_nodes, self.num_input_dim), dtype=np.float32))
                continue
            frame_joints = []
            try:
                for joint in joints:
                    pos = joint.get('position', [0.0]*self.num_input_dim)[:self.num_input_dim]
                    if len(pos) < self.num_input_dim: pos.extend([0.0] * (self.num_input_dim - len(pos)))
                    frame_joints.append(pos)
                skeleton_sequence.append(np.array(frame_joints, dtype=np.float32))
            except Exception as e:
                 logger.warning(f"样本 {sample_id} 帧 {frame_idx}: 解析关节点失败: {e}。跳过此帧。")
                 if skeleton_sequence: skeleton_sequence.append(skeleton_sequence[-1])
                 else: skeleton_sequence.append(np.zeros((self.num_nodes, self.num_input_dim), dtype=np.float32))

        if not skeleton_sequence:
            logger.error(f"样本 {sample_id}: 处理后骨骼序列为空。")
            return None

        data_numpy = np.stack(skeleton_sequence, axis=0) # (T_orig, N, C)

        # --- 数据预处理和增强 ---
        # 1. 中心化
        if self.center_joint_idx is not None and 0 <= self.center_joint_idx < self.num_nodes:
             center = data_numpy[:, self.center_joint_idx:self.center_joint_idx+1, :]
             data_numpy = data_numpy - center
        # 2. 训练时随机视角变换
        if self.split == 'train' and self.apply_rand_view_transform:
            agx = random.randint(-60, 60); agy = random.randint(-60, 60); s = random.uniform(0.5, 1.5)
            data_numpy = self.rand_view_transform(data_numpy, agx, agy, s)
        # 3. (可选) 内部 MinMax 归一化 (不推荐)
        if self.apply_normalization:
             original_shape = data_numpy.shape
             data_flat = data_numpy.reshape(-1, self.num_input_dim)
             data_min = np.min(data_flat, axis=0); data_max = np.max(data_flat, axis=0)
             denominator = data_max - data_min; denominator[denominator == 0] = 1e-6 # 避免除零
             data_normalized = (data_flat - data_min) / denominator * 2 - 1
             data_numpy = data_normalized.reshape(original_shape)
             if not np.all(np.isfinite(data_numpy)): logger.warning(f"样本 {sample_id}: 归一化后含非有限值！")

        # --- 时间步采样/插值 和 Padding ---
        # 先进行采样/插值，再 padding/截断
        current_len = data_numpy.shape[0]
        target_len = self.max_len
        if current_len == 0: # 再次检查空序列
            logger.error(f"样本 {sample_id}: 预处理后序列长度为0。")
            return None

        data_temporal = np.zeros((target_len, self.num_nodes, self.num_input_dim), dtype=np.float32)
        mask_np = np.zeros(target_len, dtype=bool)

        if self.split == 'train' and self.random_choose:
            # 随机过采样
            indices_pool = list(np.arange(current_len)) * math.ceil(target_len / current_len * 2)
            random_idx = random.sample(indices_pool, min(target_len, len(indices_pool))) # 最多取 target_len 个
            random_idx.sort()
            effective_len = len(random_idx)
            data_temporal[:effective_len] = data_numpy[random_idx, :, :]
            mask_np[:effective_len] = True # 标记有效帧
        else:
            # 线性插值/重采样
            if current_len == 1: # 只有一帧，重复
                data_temporal = np.tile(data_numpy, (target_len, 1, 1))
                mask_np[:] = True # 所有帧都有效（因为是重复的）
            elif current_len > 1:
                # 使用 linspace 生成索引，确保长度为 target_len
                idx = np.linspace(0, current_len - 1, target_len).astype(int)
                data_temporal = data_numpy[idx, :, :]
                mask_np[:] = True # 所有插值/采样后的帧都视为有效
            # else current_len == 0 的情况在前面处理了

        # --- (可选) 计算 Bone 或 Motion 特征 ---
        data_final = data_temporal
        if 'bone' in self.data_path_flag:
            data_bone = np.zeros_like(data_final)
            for i, (v1, v2) in enumerate(self.bone_pairs):
                data_bone[:, v1, :] = data_final[:, v1, :] - data_final[:, v2, :]
            data_final = data_bone
            logger.debug(f"样本 {sample_id}: 已计算 Bone 特征。")
        elif 'motion' in self.data_path_flag:
            data_motion = np.zeros_like(data_final)
            data_motion[:-1, :, :] = data_final[1:, :, :] - data_final[:-1, :, :]
            data_final = data_motion
            logger.debug(f"样本 {sample_id}: 已计算 Motion 特征。")

        # --- 转换为 Tensor ---
        x_tensor = torch.from_numpy(data_final).float()       # (T, N, C)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 标量
        mask_tensor = torch.from_numpy(mask_np).bool()        # (T,)

        # 返回 (data, label, mask, index)
        return x_tensor, label_tensor, mask_tensor, true_index


# --- 单元测试 ---
if __name__ == '__main__':
    print("测试 Feeder (内置划分)...")
    # 配置 logging 以便在测试时看到日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dummy_feeder_args = {
        'root_dir': 'data/nw-ucla/all_sqe/all_sqe/',  # <--- !!! 修改为你的 JSON 目录 !!!
        # 'split_file': None, # 不需要了
        'split': 'train',
        'data_path': 'joint',
        'window_size': 100,
        'random_choose': True,
        'apply_rand_view_transform': True,
        'apply_normalization': False,
        'repeat': 1,
        'debug': True, # 只加载少量数据测试
    }
    try:
        train_dataset = Feeder(**dummy_feeder_args)
        print(f"训练集样本数 (debug): {len(train_dataset)}")
        if len(train_dataset) > 0:
            item = train_dataset[0]
            if item is not None:
                x, label, mask, index = item
                print(f"第一个样本 X 形状: {x.shape}")      # 期望: (100, 20, 3)
                print(f"第一个样本 Label: {label.item()}")
                print(f"第一个样本 Mask 形状: {mask.shape}")   # 期望: (100,)
                print(f"第一个样本 Mask True 数量: {mask.sum().item()}") # 期望 <= 100
                print(f"第一个样本原始索引: {index}")

                # --- 测试 DataLoader ---
                def collate_fn_filter_none(batch):
                    batch = list(filter(lambda x: x is not None, batch))
                    if not batch: return None
                    return torch.utils.data.dataloader.default_collate(batch)

                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_filter_none)
                print("\n尝试加载第一个 Batch...")
                try:
                    first_batch = next(iter(train_loader))
                    if first_batch:
                        x_batch, label_batch, mask_batch, index_batch = first_batch
                        print(f"第一个 Batch X 形状: {x_batch.shape}") # 期望: (B, 100, 20, 3)
                        print(f"第一个 Batch Label 形状: {label_batch.shape}")
                        print(f"第一个 Batch Mask 形状: {mask_batch.shape}")
                        print(f"第一个 Batch 索引: {index_batch}")
                    else: print("DataLoader 返回了空 Batch。")
                except StopIteration:
                    print("DataLoader 为空，无法获取 Batch。")

            else: print("获取第一个样本失败 (返回 None)。")
        else: print("训练集为空。")

    except (ValueError, FileNotFoundError) as e: print(f"\n错误: {e}")
    except Exception as e: print(f"\n发生未知错误: {e}"); traceback.print_exc()