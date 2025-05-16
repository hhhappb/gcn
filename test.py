import numpy as np
import os

# --- 你需要根据实际情况修改这些参数 ---
NPZ_FILE_PATH = '/root/autodl-tmp/my-gcn/gcn/data/ntu/NTU60_CS.npz' # 替换成你的 .npz 文件路径
EXPECTED_NUM_NODES = 25           # 你期望的关节点数量
EXPECTED_BASE_CHANNEL = 3         # 你期望的每个关节点的通道数 (通常是 x,y,z -> 3)
# 对于 M_in_file (每帧最大人数)，我们先不预设，尝试从数据推断或假设为1或2
# --- 参数修改结束 ---

if not os.path.exists(NPZ_FILE_PATH):
    print(f"错误: 文件 '{NPZ_FILE_PATH}' 不存在！")
else:
    print(f"开始检查文件: {NPZ_FILE_PATH}")
    data_npz = np.load(NPZ_FILE_PATH)
    
    print("\n文件中包含的键 (keys):", list(data_npz.keys()))

    # --- 检查训练数据 (假设键是 'x_train') ---
    if 'x_train' in data_npz:
        x_train_raw = data_npz['x_train']
        print(f"\n--- 训练数据 ('x_train') ---")
        print(f"原始 'x_train' 形状: {x_train_raw.shape}")
        
        N_samples_train, T_orig_train, Features_flat_train = x_train_raw.shape
        print(f"推断: N_samples={N_samples_train}, T_orig={T_orig_train}, Features_flat={Features_flat_train}")

        # 尝试按照你的 feeder 代码的逻辑反推 M_in_file
        expected_features_per_person = EXPECTED_NUM_NODES * EXPECTED_BASE_CHANNEL
        if Features_flat_train % expected_features_per_person == 0:
            M_inferred_train = Features_flat_train // expected_features_per_person
            print(f"基于 Features_flat 和 V*C，推断的 M_in_file (人数): {M_inferred_train}")

            # 关键：打印第一个样本、第一帧的数据，看看它的结构
            # 我们假设先 reshape 成 (M, V, C) 来观察
            print(f"\n尝试查看第一个样本、第一帧的数据，假设的 M={M_inferred_train}, V={EXPECTED_NUM_NODES}, C={EXPECTED_BASE_CHANNEL}:")
            try:
                first_sample_first_frame_flat = x_train_raw[0, 0, :]
                # 尝试直接 reshape，如果这里报错，说明 Features_flat 的组织方式不是简单的 M * V * C
                reshaped_frame_data = first_sample_first_frame_flat.reshape(M_inferred_train, EXPECTED_NUM_NODES, EXPECTED_BASE_CHANNEL)
                print("成功 reshape 第一个样本的第一帧数据为 (M, V, C) 结构。")
                print("第一个人的数据 (V, C):\n", reshaped_frame_data[0, :, :])
                if M_inferred_train > 1:
                    print("第二个人的数据 (V, C) (如果存在且非零):\n", reshaped_frame_data[1, :, :])
            except ValueError as e:
                print(f"!!!! Reshape 失败: {e} !!!!")
                print("这通常意味着 Features_flat 的内部结构不是简单的 (M * V * C) 顺序。")
                print("例如，它可能是 (V * C * M) 或其他交错方式。")
                print("你需要仔细检查 .npz 文件的来源或生成方式，了解其确切维度排列。")

        else:
            print(f"!!!! 警告: Features_flat_train ({Features_flat_train}) 不能被 EXPECTED_NUM_NODES * EXPECTED_BASE_CHANNEL ({expected_features_per_person}) 整除！")
            print("这表明你的 EXPECTED_NUM_NODES 或 EXPECTED_BASE_CHANNEL 设置可能与数据不符，或者数据格式非常特殊。")

    else:
        print("\n文件中未找到 'x_train' 键。")

    # --- 检查标签 (假设键是 'y_train') ---
    if 'y_train' in data_npz:
        y_train_raw = data_npz['y_train']
        print(f"\n--- 训练标签 ('y_train') ---")
        print(f"原始 'y_train' 形状: {y_train_raw.shape}")
        if y_train_raw.ndim == 1:
            print("标签是一维的 (类别索引)。")
            print("前10个标签:", y_train_raw[:10])
            print("唯一标签值及其数量:", np.unique(y_train_raw, return_counts=True))
        elif y_train_raw.ndim == 2:
            print("标签是二维的 (可能是一hot编码)。")
            print("前3个标签:\n", y_train_raw[:3])
            # 如果是一hot，尝试转换为类别索引
            y_train_idx = np.argmax(y_train_raw, axis=1)
            print("转换为类别索引后的前10个标签:", y_train_idx[:10])
            print("唯一类别索引及其数量:", np.unique(y_train_idx, return_counts=True))
        else:
            print("标签的维度不寻常。")
    else:
        print("\n文件中未找到 'y_train' 键。")
        
    # (可以对 'x_test', 'y_test' 进行类似检查)
    
    data_npz.close()