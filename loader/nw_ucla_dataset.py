# 文件名: loader/nw_ucla_dataset.py
import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from tqdm import tqdm # 用于显示进度条 (可选)

# --- 你可能需要从 utils.py 导入一些辅助函数，或者在这里定义 ---
# 例如：填充函数
def pad_sequence(seq, max_len, pad_value=0.0):
    """将序列填充到指定长度"""
    seq_len = seq.shape[0]
    if seq_len < max_len:
        # 计算需要填充的长度
        pad_len = max_len - seq_len
        # 创建填充部分 (T_pad, N, C)
        # 注意：填充的值和方式可能需要调整
        padding = np.full((pad_len, seq.shape[1], seq.shape[2]), pad_value, dtype=seq.dtype)
        # 拼接原序列和填充部分
        padded_seq = np.concatenate([seq, padding], axis=0)
        # 创建掩码 (True 表示有效帧, False 表示填充帧)
        mask = np.concatenate([np.ones(seq_len, dtype=bool), np.zeros(pad_len, dtype=bool)], axis=0)
    elif seq_len > max_len:
        # 如果序列过长，则截断
        padded_seq = seq[:max_len, :, :]
        mask = np.ones(max_len, dtype=bool)
    else:
        # 长度正好
        padded_seq = seq
        mask = np.ones(max_len, dtype=bool)
    return padded_seq, mask


class NWUCLADataset(Dataset):
    """
    用于加载 NW-UCLA 数据集的 PyTorch Dataset 类。
    假设 NW-UCLA 数据以 JSON 文件形式存储，每个文件包含一个动作序列。
    """
    def __init__(self, dataset_args, split):
        """
        初始化数据集。
        Args:
            dataset_args (dict): 数据集相关的配置参数，至少包含 'root' 路径。
                                 可能还需要 'max_len', 'num_nodes', 'num_input_dim',
                                 以及指定训练/测试 split 的文件路径。
            split (str): 'train' 或 'test'。
        """
        super().__init__()
        self.root_dir = dataset_args['root'] # 数据集根目录，例如 'data/NW-UCLA'
        self.split = split
        self.max_len = dataset_args.get('max_len', 100) # 序列最大长度 (用于填充/截断)
        self.num_nodes = dataset_args.get('num_nodes', 20) # NW-UCLA 通常是 20 个关节点
        self.num_input_dim = dataset_args.get('num_input_dim', 3) # 通常是 x, y, z

        print(f"加载 NW-UCLA 数据集, split: {split}, 根目录: {self.root_dir}")

        # --- 核心步骤 1: 加载样本列表和标签 ---
        self.samples = [] # 用于存储 (json_file_path, label) 对
        self._load_samples()

        if not self.samples:
            raise FileNotFoundError(f"在 '{self.root_dir}' 中没有找到 '{split}' split 的数据文件。请检查路径和文件结构。")

        print(f"找到 {len(self.samples)} 个样本用于 '{split}' split。")

    def _load_samples(self):
        """
        根据 split ('train' or 'test') 加载对应的样本文件路径和标签。
        你需要根据 NW-UCLA 数据集的具体组织方式来实现这个逻辑。

        常见的实现方式：
        1.  可能有 train.list / test.list 文件指定了每个 split 包含哪些样本 ID 或文件名。
        2.  或者根据文件名或子文件夹来区分训练/测试样本。
        3.  标签信息通常也需要从文件名、文件夹名或单独的标签文件中获取。
        """
        # 示例逻辑 (需要你根据实际情况修改)
        split_file_path = os.path.join(self.root_dir, f"{self.split}_list.txt") # 假设有列表文件
        if not os.path.exists(split_file_path):
             print(f"警告：找不到 split 文件 '{split_file_path}'。尝试直接扫描 JSON 文件...")
             # 备选：直接扫描所有 .json 文件，可能需要其他方式区分 split
             all_files = [f for f in os.listdir(self.root_dir) if f.endswith('.json')]
             # 这里需要添加逻辑来确定哪些文件属于当前 split 以及它们的标签
             # 例如，如果文件名包含 's01', 's02' 等表示主题，可以用主题ID划分
             # 例如，如果文件名包含 'a01', 'a02' 等表示动作，用动作ID做标签
             print("错误：直接扫描 JSON 的逻辑未实现，请根据你的数据组织方式添加。")
             return # 或者抛出错误

        # --- 如果使用列表文件 ---
        print(f"从 {split_file_path} 加载样本列表...")
        with open(split_file_path, 'r') as f:
            for line in f:
                sample_id = line.strip() # 例如 'a01_s01_e01'
                json_file_name = f"{sample_id}.json" # 假设 JSON 文件名与 ID 对应
                json_file_path = os.path.join(self.root_dir, json_file_name)

                if os.path.exists(json_file_path):
                    # --- 提取标签 ---
                    # 假设动作标签可以从 sample_id 中提取，例如 'aXX'
                    try:
                        # 动作类别通常从 1 开始，转换为从 0 开始的索引
                        action_id = int(sample_id.split('_')[0][1:]) - 1
                        label = action_id
                        self.samples.append((json_file_path, label))
                    except (IndexError, ValueError):
                        print(f"警告：无法从样本 ID '{sample_id}' 中提取有效的标签。跳过此样本。")
                else:
                    print(f"警告：找不到对应的 JSON 文件 '{json_file_path}'。跳过此样本。")

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取并处理单个数据样本。
        Args:
            idx (int): 样本索引。
        Returns:
            tuple: (x_tensor, label_tensor, mask_tensor)
                   x_tensor: 处理后的骨骼序列 (SeqLen, NumNodes, InputDim)
                   label_tensor: 类别标签 (标量)
                   mask_tensor: 有效帧掩码 (SeqLen,)
        """
        json_file_path, label = self.samples[idx]

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"错误：无法读取或解析 JSON 文件 {json_file_path}: {e}")
            # 返回一个虚拟的错误样本或抛出异常
            # 这里简单返回 None，DataLoader 的 collate_fn 需要处理这种情况 (或者直接报错)
            # return None, None, None
            # 或者返回一个全零的样本？需要确保维度正确
            dummy_x = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long) # 使用无效标签
            dummy_mask = torch.zeros(self.max_len, dtype=torch.bool)
            return dummy_x, dummy_label, dummy_mask


        # --- 核心步骤 2: 从 JSON 数据中提取骨骼序列 ---
        frames_data = data.get('frames', []) # 获取 'frames' 列表，如果不存在则返回空列表
        if not frames_data:
            print(f"警告：JSON 文件 {json_file_path} 中没有 'frames' 数据或为空。")
            # 返回错误样本
            dummy_x = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long) # 使用无效标签
            dummy_mask = torch.zeros(self.max_len, dtype=torch.bool)
            return dummy_x, dummy_label, dummy_mask

        skeleton_sequence = []
        for frame in frames_data:
            skeletons = frame.get('skeletons', [])
            if not skeletons:
                # print(f"警告: {json_file_path} 的某一帧没有骨骼数据，可能需要处理。")
                # 可以选择跳过此帧，或用上一帧/零填充
                # 简单的处理：如果序列已有数据，用上一帧；否则用零
                if skeleton_sequence:
                    skeleton_sequence.append(skeleton_sequence[-1]) # 重复上一帧的关节点
                else:
                    # 如果是第一帧就没有，则添加零帧
                    zero_joints = np.zeros((self.num_nodes, self.num_input_dim), dtype=np.float32)
                    skeleton_sequence.append(zero_joints)
                continue # 处理下一帧

            # --- 处理单个骨骼 (NW-UCLA 通常每帧只有一个主要骨骼) ---
            # 如果可能有多个骨骼，你需要决定如何选择或合并
            target_skeleton = skeletons[0] # 简单取第一个骨骼
            joints = target_skeleton.get('joints', [])

            if len(joints) != self.num_nodes:
                 print(f"警告：{json_file_path} 某帧的关节点数量 ({len(joints)}) 与期望 ({self.num_nodes}) 不符。可能需要填充或截断。")
                 # 进行填充或截断处理，这里简化为跳过此帧（如果需要更鲁棒，应处理）
                 if skeleton_sequence:
                     skeleton_sequence.append(skeleton_sequence[-1])
                 else:
                     zero_joints = np.zeros((self.num_nodes, self.num_input_dim), dtype=np.float32)
                     skeleton_sequence.append(zero_joints)
                 continue

            frame_joints = []
            for joint in joints:
                # 提取坐标，你需要根据 JSON 的具体字段名调整
                # 常见的字段名可能是 'x', 'y', 'z' 或 'position' [x, y, z]
                # 假设是 'position' 列表
                pos = joint.get('position', [0.0, 0.0, 0.0])[:self.num_input_dim] # 取前 N 个维度
                if len(pos) < self.num_input_dim: # 如果维度不够，补零
                    pos.extend([0.0] * (self.num_input_dim - len(pos)))
                frame_joints.append(pos)

            skeleton_sequence.append(np.array(frame_joints, dtype=np.float32)) # (NumNodes, InputDim)

        if not skeleton_sequence:
             print(f"错误：处理完 {json_file_path} 后骨骼序列为空。")
             # 返回错误样本
             dummy_x = torch.zeros((self.max_len, self.num_nodes, self.num_input_dim), dtype=torch.float32)
             dummy_label = torch.tensor(-1, dtype=torch.long) # 使用无效标签
             dummy_mask = torch.zeros(self.max_len, dtype=torch.bool)
             return dummy_x, dummy_label, dummy_mask

        # 将列表转换为 NumPy 数组 (SeqLen, NumNodes, InputDim)
        x_np = np.stack(skeleton_sequence, axis=0)

        # --- 核心步骤 3: 序列填充/截断 和 生成 Mask ---
        x_padded, mask = pad_sequence(x_np, self.max_len)

        # --- 核心步骤 4: 转换为 Tensor ---
        x_tensor = torch.from_numpy(x_padded).float()
        # 标签转换为 LongTensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        # Mask 转换为 BoolTensor
        mask_tensor = torch.from_numpy(mask).bool()

        # --- 可选：在这里进行数据归一化或变换 ---
        # 例如，减去根节点坐标（中心化）
        # root_joint_idx = 0 # 假设根节点索引为 0
        # root_coords = x_tensor[:, root_joint_idx:root_joint_idx+1, :] # (T, 1, C)
        # x_tensor = x_tensor - root_coords # 广播减法

        return x_tensor, label_tensor, mask_tensor

    # --- 可选：添加计算 mean/std 的方法 ---
    # 注意：这可能非常耗时，通常建议预计算并保存
    def calculate_mean_std(self, first_n_samples=1000):
        """ 估算数据集的均值和标准差 (通常在训练集上计算) """
        if self.split != 'train':
            print("警告：通常只在训练集上计算 mean/std。")
            # 可以返回 0 和 1，或者加载预计算的值

        print(f"计算前 {first_n_samples} 个样本的 mean/std (可能较慢)...")
        all_frames = []
        num_samples_to_use = min(first_n_samples, len(self))
        for i in tqdm(range(num_samples_to_use), desc="计算 Mean/Std"):
             x, _, mask = self[i] # 获取样本
             # 只使用有效帧进行计算
             valid_frames = x[mask] # (NumValidFrames, N, C)
             if valid_frames.numel() > 0:
                 all_frames.append(valid_frames.reshape(-1, self.num_input_dim)) # (NumValidFrames*N, C)

        if not all_frames:
            print("错误：无法收集有效帧来计算 mean/std。")
            return torch.zeros(self.num_input_dim), torch.ones(self.num_input_dim)

        all_frames_tensor = torch.cat(all_frames, dim=0) # (TotalValidFrames*N, C)
        mean = torch.mean(all_frames_tensor, dim=0)    # (C,)
        std = torch.std(all_frames_tensor, dim=0)      # (C,)
        # 防止 std 为 0
        std[std == 0] = 1.0
        print(f"计算得到的 Mean: {mean.numpy()}, Std: {std.numpy()}")
        return mean, std

# --- 单元测试 (可选) ---
if __name__ == '__main__':
    print("测试 NWUCLADataset...")
    # 创建一个虚拟的配置文件字典
    dummy_dataset_args = {
        # --- !!! 修改为你本地 NW-UCLA 数据集的实际根目录 !!! ---
        'root': '/path/to/your/nw-ucla-data', # 例如 'data/NW-UCLA'
        'max_len': 150, # 示例最大长度
        'num_nodes': 20,
        'num_input_dim': 3,
        # --- !!! 如果你的 split 文件名不同，在这里修改 !!! ---
        # 'split_file_pattern': '{split}_list.txt' # 或者其他模式
    }

    # --- !!! 确保你的数据目录下有对应的 JSON 文件和 split 列表文件 !!! ---
    # 例如，在 /path/to/your/nw-ucla-data 下应该有 a01_s01_e01.json 等文件
    # 以及 train_list.txt 和 test_list.txt 文件，内容是样本 ID 列表

    try:
        # 测试训练集加载
        train_dataset = NWUCLADataset(dummy_dataset_args, split='train')
        print(f"训练集样本数: {len(train_dataset)}")

        # 获取第一个样本
        if len(train_dataset) > 0:
            x, label, mask = train_dataset[0]
            print(f"第一个样本 X 形状: {x.shape}")      # 期望: (max_len, num_nodes, num_input_dim)
            print(f"第一个样本 Label: {label.item()}")
            print(f"第一个样本 Mask 形状: {mask.shape}")   # 期望: (max_len,)
            print(f"第一个样本有效帧数: {mask.sum().item()}")

            # 测试 DataLoader
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            first_batch = next(iter(train_loader))
            x_batch, label_batch, mask_batch = first_batch
            print(f"\n第一个 Batch X 形状: {x_batch.shape}") # 期望: (batch_size, max_len, num_nodes, num_input_dim)
            print(f"第一个 Batch Label 形状: {label_batch.shape}") # 期望: (batch_size,)
            print(f"第一个 Batch Mask 形状: {mask_batch.shape}")   # 期望: (batch_size, max_len,)

            # # 测试计算 mean/std (如果需要且实现了)
            # mean, std = train_dataset.calculate_mean_std(first_n_samples=10)
            # print(f"\n估算的 Mean: {mean}, Std: {std}")

        else:
            print("训练集为空，无法获取样本。")

        # 测试测试集加载 (类似)
        test_dataset = NWUCLADataset(dummy_dataset_args, split='test')
        print(f"\n测试集样本数: {len(test_dataset)}")
        if len(test_dataset) > 0:
             x_test, label_test, mask_test = test_dataset[0]
             # ... 打印形状 ...

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保 'root' 路径设置正确，并且包含 NW-UCLA 数据集文件和对应的 split 列表文件。")
    except NotImplementedError as e:
         print(f"\n错误: {e}")
         print("请在 _load_samples 方法中实现适合你的数据组织的逻辑。")