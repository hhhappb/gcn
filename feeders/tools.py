import random
import matplotlib.pyplot as plt # 导入 matplotlib 用于绘图 (当前代码中未使用)
import numpy as np # 导入 numpy 用于数值计算
import pdb # 导入 Python 调试器 (当前代码中未使用)

import torch # 导入 PyTorch
import torch.nn.functional as F # 导入 PyTorch 的函数式接口，例如插值函数

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M # 输入数据维度注释：通道, 时间, 节点, 人数
    C, T, V, M = data_numpy.shape # 获取输入数据的维度
    begin = 0 # 有效帧的起始索引（假设从0开始）
    end = valid_frame_num # 有效帧的结束索引（不包含此帧）
    valid_size = end - begin # 有效帧的总数量

    #crop # 裁剪操作
    if len(p_interval) == 1: # 如果 p_interval 只包含一个值 (通常用于测试/验证，固定裁剪比例)
        p = p_interval[0] # 获取裁剪比例
        bias = int((1-p) * valid_size/2) # 计算中心裁剪的偏移量
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop # 执行中心裁剪
        cropped_length = data.shape[1] # 裁剪后的长度
    else: # 如果 p_interval 是一个范围 [min_p, max_p] (通常用于训练，随机裁剪比例)
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0] # 在范围内随机选择一个裁剪比例 p
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64 # 计算裁剪长度，并确保其在 [64, valid_size] 之间
        bias = np.random.randint(0,valid_size-cropped_length+1) # 随机选择裁剪的起始位置
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :] # 执行随机裁剪
        if data.shape[1] == 0: # 如果裁剪后长度为0 (异常情况)
            print(cropped_length, bias, valid_size) # 打印调试信息

    # resize # 缩放操作
    data = torch.tensor(data,dtype=torch.float) # 将numpy数组转换为PyTorch张量
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length) # 维度变换和展平以适应插值函数
    data = data[None, None, :, :] # 增加维度以适应 F.interpolate
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample # 使用双线性插值调整时间维度到 window 大小
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy() # 恢复维度并转回numpy

    return data

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M # 输入数据维度注释
    begin = np.random.randint(step) if random_sample else 0 # 如果随机采样，则随机选择起始帧索引
    return data_numpy[:, begin::step, :, :] # 按步长进行下采样

def temporal_slice(data_numpy, step):
    # input: C,T,V,M # 输入数据维度注释
    C, T, V, M = data_numpy.shape # 获取维度
    # 将时间维度T切分成 T/step 个片段，每个片段长度为 step，并重排维度
    return data_numpy.reshape(C, T // step, step, V, M).transpose( # 使用整数除法确保维度正确
        (0, 1, 3, 2, 4)).reshape(C, T // step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M # 输入数据维度注释
    # naive version # 简易版本的均值减法
    if mean == 0: # 如果均值为0，则不进行操作
        return
    C, T, V, M = data_numpy.shape # 获取维度
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0 # 判断哪些帧是有效帧（非全零）
    begin = valid_frame.argmax() # 第一个有效帧的索引
    end = len(valid_frame) - valid_frame[::-1].argmax() # 最后一个有效帧的后一个索引
    data_numpy[:, begin:end, :, :] = data_numpy[:, begin:end, :, :] - mean # 仅对有效帧部分减去均值
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    # 输入: C,T,V,M
    # 自动填充函数，将时间维度 T 填充到目标大小 size
    C, T, V, M = data_numpy.shape
    if T < size: # 如果当前时间长度小于目标大小
        begin = random.randint(0, size - T) if random_pad else 0 # 如果是随机填充，则随机选择填充的起始位置
        data_numpy_paded = np.zeros((C, size, V, M)) # 创建一个全零的目标大小数组
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy # 将原始数据复制到填充数组的指定位置
        return data_numpy_paded
    else: # 如果长度不小于目标大小，则直接返回原始数据
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0 # 输入数据维度注释，并指出此方法可能存在的问题
    C, T, V, M = data_numpy.shape # 获取维度
    if T == size: # 如果当前长度等于目标长度，直接返回
        return data_numpy
    elif T < size: # 如果当前长度小于目标长度
        if auto_pad: # 如果允许自动填充
            return auto_pading(data_numpy, size, random_pad=True) # 调用自动填充函数
        else: # 否则直接返回（长度不足）
            return data_numpy
    else: # 如果当前长度大于目标长度
        begin = random.randint(0, T - size) # 随机选择一个起始点
        return data_numpy[:, begin:begin + size, :, :] # 返回选定的时间片段

def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.], # 候选旋转角度列表 (度)
                scale_candidate=[0.9, 1.0, 1.1],         # 候选缩放因子列表
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2], # 候选平移量列表
                move_time_candidate=[1]):               # 变换发生的次数 (将序列分成几段进行变换)
    # input: C,T,V,M # 输入数据维度注释
    # 对XY平面进行随机的仿射变换 (旋转、缩放、平移)，变换参数在时间上平滑过渡
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate) # 随机选择变换发生的段数
    # 生成变换的关键帧节点索引
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T) # 确保包含序列的结束点
    num_node = len(node) # 关键帧的数量

    # 为每个关键帧随机选择变换参数
    A = np.random.choice(angle_candidate, num_node) # 旋转角度
    S = np.random.choice(scale_candidate, num_node)   # 缩放因子
    T_x = np.random.choice(transform_candidate, num_node) # X方向平移
    T_y = np.random.choice(transform_candidate, num_node) # Y方向平移

    # 初始化每个时间步的变换参数数组
    a = np.zeros(T) # 角度 (弧度)
    s = np.zeros(T) # 缩放
    t_x = np.zeros(T) # X平移
    t_y = np.zeros(T) # Y平移

    # linspace # 在关键帧之间线性插值变换参数
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180 # 角度转为弧度
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    # 构建每个时间步的2D旋转和缩放矩阵
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s], # (2, T)
                      [np.sin(a) * s, np.cos(a) * s]]) # (2, T) -> theta (2, 2, T)

    # perform transformation # 执行变换
    for i_frame in range(T): # 遍历每一帧
        xy = data_numpy[0:2, i_frame, :, :] # 取出当前帧的XY坐标数据 (2, V, M)
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1)) # 应用旋转和缩放
        new_xy[0] += t_x[i_frame] # 应用X方向平移
        new_xy[1] += t_y[i_frame] # 应用Y方向平移
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M) # 将变换后的XY坐标写回

    return data_numpy


def random_shift(data_numpy):
    # 输入: C,T,V,M
    # 将数据中的有效帧片段随机平移到序列中的其他位置，其余部分用零填充
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape) # 创建一个全零的目标数组
    # 找到有效帧的起始和结束索引
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin # 有效帧片段的长度
    if size <=0 : return data_numpy # 如果没有有效帧，则不进行操作
    bias = random.randint(0, T - size) # 随机选择新的起始位置
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :] # 将有效帧片段复制到新位置

    return data_shift


def _rot(rot):
    """
    rot: T,3 # 输入旋转角度张量，形状 (T,3)，T是时间步数，3对应绕X,Y,Z轴的旋转角度(弧度)
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3 # 计算余弦和正弦值
    zeros = torch.zeros(rot.shape[0], 1, device=rot.device)  # T,1 # 创建零张量，确保在同一设备
    ones = torch.ones(rot.shape[0], 1, device=rot.device)   # T,1 # 创建一张量

    # 构建绕X轴的旋转矩阵 Rx (T,3,3)
    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3 # Rx 的第一行
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # Rx 的第二行
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1) # Rx 的第三行
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    # 构建绕Y轴的旋转矩阵 Ry (T,3,3)
    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1) # Ry 的第一行
    r2 = torch.stack((zeros, ones, zeros),dim=-1) # Ry 的第二行
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1) # Ry 的第三行
    ry = torch.cat((ry1, r2, ry3), dim = 1) # T,3,3

    # 构建绕Z轴的旋转矩阵 Rz (T,3,3)
    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1) # Rz 的第一行
    r3 = torch.stack((zeros, zeros, ones),dim=-1) # Rz 的第三行 (注意这里变量名是r3)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1) # Rz 的第二行
    rz = torch.cat((rz1, rz2, r3), dim = 1) # T,3,3

    rot_matrix = rz.matmul(ry).matmul(rx) # 组合旋转矩阵 R = Rz * Ry * Rx
    return rot_matrix


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M # 输入数据，C通常为3 (XYZ)
    theta: 随机旋转角度的范围 [-theta, theta] (弧度)
    """
    data_torch = torch.from_numpy(data_numpy) # 将numpy数组转换为PyTorch张量
    C, T, V, M = data_torch.shape
    # 维度变换以方便矩阵乘法: (C,T,V,M) -> (T,C,V,M) -> (T,C,V*M)
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)
    # 为每个轴生成一个随机旋转角度，并在所有时间步上共享这个角度
    rot_angles_xyz = torch.zeros(3, device=data_torch.device).uniform_(-theta, theta) # (3,)
    # 将旋转角度复制到每个时间步
    rot_angles_per_timestep = torch.stack([rot_angles_xyz, ] * T, dim=0) # (T,3)
    rotation_matrix = _rot(rot_angles_per_timestep)  # (T,3,3) 获取每个时间步的旋转矩阵
    # 应用旋转: (T,3,3) @ (T,3,V*M) -> (T,3,V*M)
    # 这里假设 C=3 (XYZ坐标)。如果C不是3，matmul操作会失败。
    data_rotated = torch.matmul(rotation_matrix, data_torch)
    # 恢复原始维度顺序: (T,C,V*M) -> (T,C,V,M) -> (C,T,V,M)
    data_rotated = data_rotated.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_rotated.numpy() # 转换回numpy数组

def openpose_match(data_numpy):
    # 输入: C,T,V,M
    # 此函数尝试通过匹配相邻帧中的姿态来处理多目标跟踪中人物ID可能发生跳变的问题。
    # 逻辑较为复杂，依赖于骨骼点坐标和可能的置信度分数。
    C, T, V, M = data_numpy.shape
    assert (C == 3) # 假设处理的是XYZ数据，其中第三个通道可能包含置信度或被用于计算分数
    score = data_numpy[2, :, :, :].sum(axis=1) # 使用第三个通道(如Z坐标或置信度)在节点维度上求和作为每个人的分数 (T, M)
    # the rank of body confidence in each frame (shape: T-1, M) # 计算除最后一帧外，每一帧中人物的置信度排名
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1 # 准备前一帧的XY坐标
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2 # 准备当前帧的XY坐标
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M) # 计算相邻帧之间所有人物配对的欧氏距离平方和
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose # 姿态匹配
    forward_map = np.zeros((T, M), dtype=int) - 1 # 初始化前向ID映射表，-1表示未匹配
    forward_map[0] = range(M) # 第一帧的ID保持不变
    for m in range(M): # 遍历每个人物ID (作为前一帧的ID)
        choose = (rank == m) # 找到在前一帧中置信度排名为m的人物
        forward = distance[choose].argmin(axis=1) # 在对应的下一帧中，为这些人找到距离最近的人物作为匹配
        for t in range(T - 1): # 将已匹配的下一帧人物的距离设为无穷大，避免被重复匹配
            distance[t, :, forward[t]] = np.inf # 此处原始代码的索引可能存在问题，需谨慎
        forward_map[1:][choose] = forward # 更新下一帧的ID映射
    assert (np.all(forward_map >= 0)) # 断言确保所有ID都已成功匹配

    # string data # 串联ID映射，确保ID在整个序列中的一致性
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]] # 根据前一帧的映射更新当前帧的映射

    # generate data # 根据最终的ID映射重新组织数据
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0) # 按新ID顺序排列人物数据
    data_numpy = new_data_numpy

    # score sort # 根据每个人物在整个序列中的总置信度得分，对人物进行最终排序
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0) # 计算总分
    rank = (-trace_score).argsort() # 按总分降序排列
    data_numpy = data_numpy[:, :, :, rank] # 按排序后的人物顺序重新排列数据

    return data_numpy