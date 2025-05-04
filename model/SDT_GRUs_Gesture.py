# -*- coding: utf-8 -*-
# 文件名: model/SDT_GRUs_Gesture.py (v12.2 - DropPath in EncoderLayer, based on v12.1)
# 修改说明：
# 1. 保持 v12.1 的整体结构和模块。
# 2. 在 EncoderLayer 中，将残差连接上的 nn.Dropout 替换为 DropPath。
# 3. 确保 DropPath 正确导入或定义。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
import inspect
import yaml
import logging
logger = logging.getLogger(__name__)
# 确保从 timm.models.layers 导入 DropPath 和 trunc_normal_
try:
    from timm.models.layers import trunc_normal_, DropPath
    logger.info("成功从 timm.models.layers 导入 DropPath 和 trunc_normal_。")
except ImportError:
    print("无法导入 timm.models.layers 中的 DropPath 或 trunc_normal_。")
    print("请确保已安装 timm 库: pip install timm")
    # 提供一个简单的 DropPath 替代实现（如果 timm 未安装）
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            if drop_prob is not None and not (0.0 <= drop_prob <= 1.0):
                 raise ValueError("drop_prob must be between 0 and 1")

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            # 获取形状 (B, ...) -> (B, 1, 1, ...)
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            # 生成随机张量并二值化
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            # 应用 DropPath 并调整尺度
            output = x.div(keep_prob) * random_tensor
            return output
    # 提供一个简单的 trunc_normal_ 替代实现
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        # 简化的截断正态分布初始化
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(min=a, max=b)
    print("警告: 无法从 timm 导入 DropPath 或 trunc_normal_，将使用内置的简化版本。")


logger = logging.getLogger(__name__)

print(f"PyTorch 版本: {torch.__version__}")

# --- 位置编码 (PositionalEmbedding) ---
class PositionalEmbedding(nn.Module):
    """标准的 Sinusoidal Positional Embedding (用于时间维度)"""
    def __init__(self, d_model, max_len=500):
        """
        初始化位置编码层。
        Args:
            d_model (int): 特征维度。
            max_len (int): 预计算的最大序列长度。
        """
        super(PositionalEmbedding, self).__init__()
        # 创建一个 (max_len, d_model) 的零矩阵，用于存储位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False # 位置编码不需要梯度

        # 创建位置索引 (0, 1, ..., max_len-1) 并增加一个维度 -> (max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算用于 sin 和 cos 的分母项中的指数部分
        # div_term 形式如 [1/10000^(0/d), 1/10000^(2/d), 1/10000^(4/d), ...]
        div_term_exponent = torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        div_term = torch.exp(div_term_exponent)

        # 计算偶数维度的位置编码 (sin)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算奇数维度的位置编码 (cos)
        if d_model % 2 == 0:
             pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # 处理 d_model 是奇数的情况，最后一个 cos 使用少一个 div_term 元素
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # 增加一个批次维度 -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 将 pe 注册为 buffer，这样它会随模型保存和加载，但不会被视为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        向前传播，为输入 x 添加位置编码。
        Args:
            x (Tensor): 输入张量，形状通常为 (B, T, N, D) 或 (B, T, D)。
                        函数需要根据 T 维度截取位置编码。
        Returns:
            Tensor: 截取后的位置编码，形状为 (1, T, D)。
        """
        # 假设时间维度 T 总是在 dim 1
        time_dim_idx = 1
        # 返回 pe 中前 T 个时间步的位置编码
        return self.pe[:, :x.size(time_dim_idx)]

# --- 空间注意力层 (AttentionLayer - Scaled Dot-Product) ---
class AttentionLayer(nn.Module):
    """
    空间注意力层，使用标准的 Scaled Dot-Product Attention。
    支持使用 Conv1D 或 Linear 进行 QKV 投影，并可加入相对位置偏置。
    """
    def __init__(self, d_model, n_heads, dropout, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20):
        """
        初始化空间注意力层。
        Args:
            d_model (int): 输入和输出特征维度。
            n_heads (int): 注意力头数。d_model 必须能被 n_heads 整除。
            dropout (float): 应用于注意力权重的 dropout 率。
            output_attention (bool): 是否返回注意力权重图。
            qkv_bias (bool): QKV 投影层是否使用偏置。
            use_conv_proj (bool): 是否使用 Conv1D 进行 QKV 投影 (True)，否则使用 Linear (False)。
            conv_kernel_size (int): 如果使用 Conv1D，其卷积核大小。
            num_nodes (int): 图中的节点数量。
        """
        super().__init__()
        if d_model % n_heads != 0: raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads  # 每个头的 key/query 维度
        self.d_values = d_model // n_heads # 每个头的 value 维度
        self.output_attention = output_attention
        self.use_conv_proj = use_conv_proj
        self.conv_kernel_size = conv_kernel_size
        self.num_nodes = num_nodes

        # 定义 QKV 投影层
        if use_conv_proj:
            padding = conv_kernel_size // 2
            self.query_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.key_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.value_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            # 可选：用于 Conv1D 投影后的残差连接的 LayerNorm
            self.resid_norm_q = nn.LayerNorm(d_model); self.resid_norm_k = nn.LayerNorm(d_model); self.resid_norm_v = nn.LayerNorm(d_model)
        else:
            self.query_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.key_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.value_projection = nn.Linear(d_model, d_model, bias=qkv_bias)

        # 定义相对位置偏置表和索引
        # 表的大小为 (2*N-1, H)，N 是节点数，H 是头数
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * num_nodes - 1), n_heads))
        # 初始化偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # 计算相对坐标索引矩阵 (N, N)
        coords_n = torch.arange(num_nodes)
        relative_coords_n = coords_n[:, None] - coords_n[None, :] # 值范围 [-N+1, N-1]
        relative_coords_n += num_nodes - 1 # 转换为非负索引 [0, 2N-2]
        # 注册为 buffer
        self.register_buffer("relative_position_index", relative_coords_n, persistent=False)

        # 输出投影层
        self.out_projection = nn.Linear(d_model, d_model)
        # 用于注意力权重的 Dropout 层
        self.dropout = nn.Dropout(dropout)

    def _get_relative_positional_bias(self) -> torch.Tensor:
        """根据相对位置索引从偏置表中查找并重塑偏置。"""
        # 使用索引从表中查找偏置 (N*N, H) -> (N, N, H)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_nodes, self.num_nodes, -1)
        # 调整维度顺序以匹配注意力分数 (H, N, N)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 增加批次维度 (1, H, N, N)
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        """
        AttentionLayer 的前向传播。
        Args:
            x (Tensor): 输入张量，形状为 (B, N, D)。
        Returns:
            Tensor: 注意力层的输出，形状为 (B, N, D)。
            Tensor | None: 注意力权重图 (如果 output_attention=True)，形状为 (B, H, N, N)。
        """
        B, N, D = x.shape; H = self.n_heads
        if N != self.num_nodes: raise ValueError(f"Input node dim {N} != expected {self.num_nodes}")

        # 1. 计算 Q, K, V
        if self.use_conv_proj:
            # 输入 (B, N, D) -> (B, D, N) 以适应 Conv1D
            x_permuted = x.permute(0, 2, 1).contiguous()
            # 应用 Conv1D 投影
            q_conv = self.query_projection(x_permuted); k_conv = self.key_projection(x_permuted); v_conv = self.value_projection(x_permuted)
            # 可选：应用 LayerNorm 和残差连接
            q_conv_norm = self.resid_norm_q(q_conv.permute(0, 2, 1)).permute(0, 2, 1)
            k_conv_norm = self.resid_norm_k(k_conv.permute(0, 2, 1)).permute(0, 2, 1)
            v_conv_norm = self.resid_norm_v(v_conv.permute(0, 2, 1)).permute(0, 2, 1)
            # (B, D, N) -> (B, N, D)
            queries_proj = x + q_conv_norm.permute(0, 2, 1).contiguous()
            keys_proj = x + k_conv_norm.permute(0, 2, 1).contiguous()
            values_proj = x + v_conv_norm.permute(0, 2, 1).contiguous()
        else:
            # 应用 Linear 投影
             queries_proj = self.query_projection(x); keys_proj = self.key_projection(x); values_proj = self.value_projection(x)

        # 2. 调整形状以支持多头: (B, N, D) -> (B, H, N, d_k or d_v)
        queries = queries_proj.view(B, N, H, self.d_keys).transpose(1, 2)
        keys = keys_proj.view(B, N, H, self.d_keys).transpose(1, 2)
        values = values_proj.view(B, N, H, self.d_values).transpose(1, 2)

        # 3. 计算 Scaled Dot-Product Attention 分数
        # (B, H, N, dk) @ (B, H, dk, N) -> (B, H, N, N)
        scale = 1. / math.sqrt(self.d_keys)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        # 4. 添加相对位置偏置
        relative_position_bias = self._get_relative_positional_bias() # (1, H, N, N)
        scores = scores + relative_position_bias

        # 5. Softmax 归一化得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1) # (B, H, N, N)

        # 6. 应用 Dropout 到注意力权重
        attn_weights_dropped = self.dropout(attn_weights)

        # 7. 用加权的注意力权重聚合 Value
        # (B, H, N, N) @ (B, H, N, dv) -> (B, H, N, dv)
        weighted_values = torch.matmul(attn_weights_dropped, values)

        # 8. 整合多头输出并进行最终投影
        # (B, H, N, dv) -> (B, N, H, dv) -> (B, N, H*dv=D)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_projection(weighted_values) # (B, N, D)

        return output, attn_weights.detach() if self.output_attention else None

# --- 前馈网络模块 (FeedForwardModule) ---
class FeedForwardModule(nn.Module):
    """标准的前馈网络模块 (Linear -> Activation -> Dropout -> Linear)"""
    def __init__(self, d_model, ffn_dim, activation=F.gelu, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout) # FFN 内部的 Dropout
        self.activation = activation
        self.linear2 = nn.Linear(ffn_dim, d_model)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# --- 空间注意力编码器层 (EncoderLayer - 使用 DropPath) ---
class EncoderLayer(nn.Module):
    """
    空间注意力的基础构建块，包含一个自注意力层和一个前馈网络层。
    使用 DropPath 进行残差连接的正则化。
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.gelu, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20):
        super().__init__()
        # 实例化空间自注意力层
        self.self_attn = AttentionLayer(d_model, n_heads, dropout_rate, output_attention, qkv_bias=qkv_bias, use_conv_proj=use_conv_proj, conv_kernel_size=conv_kernel_size, num_nodes=num_nodes)
        # 实例前馈网络
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate)
        # Layer Normalization 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 使用 DropPath 替代残差连接中的 Dropout
        self.drop_path1 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(self, x):
        # 1. 自注意力 + 残差连接 + LayerNorm
        attn_output, attn_weights = self.self_attn(x)
        x = x + self.drop_path1(attn_output) # 应用 DropPath
        x = self.norm1(x)
        # 2. 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = x + self.drop_path2(ffn_output) # 应用 DropPath
        x = self.norm2(x)
        return x, attn_weights

# --- 空间注意力模块 (SpatialAttention) ---
class SpatialAttention(nn.Module):
    """包含多个 EncoderLayer 的空间注意力模块"""
    def __init__(self, d_model, num_nodes, n_heads, ffn_dim, st_layers, dropout_rate, output_attention,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.st_layers = st_layers # 空间注意力层数
        self.output_attention = output_attention
        self.use_node_pos_emb = False # 这个版本不使用显式的节点位置嵌入

        # 创建指定层数的 EncoderLayer
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim,
                         dropout_rate=dropout_rate, output_attention=output_attention,
                         qkv_bias=qkv_bias, use_conv_proj=use_conv_proj,
                         conv_kernel_size=conv_kernel_size, num_nodes=num_nodes)
            for _ in range(st_layers)
        ])

    def forward(self, x):
        attn_list = []
        # 依次通过每个 EncoderLayer
        for encoder in self.encoders:
            x, attn = encoder(x)
            if self.output_attention and attn is not None:
                 attn_list.append(attn)
        # 只返回最后一层的注意力图（如果需要）
        final_attn = attn_list[-1] if self.output_attention and attn_list else None
        return x, final_attn

# --- 时空 GRU 单元 (SpatialTemporalCell) ---
class SpatialTemporalCell(nn.Module):
    """
    结合了空间注意力 (SpatialAttention) 和 GRU 的时空处理单元。
    在每个时间步，先对输入和隐藏状态进行空间注意力处理，然后送入 GRU 进行更新。
    """
    def __init__(self, in_channels, out_channels, num_nodes, n_heads, ffn_dim, st_layers, st_dropout_rate, output_attention,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3):
        super().__init__()
        # 输入输出通道数必须相等，因为 GRU 的隐藏状态维度与输入/输出维度一致
        if in_channels != out_channels: raise ValueError("SpatialTemporalCell 的 in_channels 必须等于 out_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.output_attention = output_attention

        # 为输入 x 和隐藏状态 h 分别实例化空间注意力模块
        self.spatial_attn_i = SpatialAttention(d_model=out_channels, num_nodes=num_nodes, n_heads=n_heads, ffn_dim=ffn_dim, st_layers=st_layers, dropout_rate=st_dropout_rate, output_attention=output_attention, qkv_bias=qkv_bias, use_conv_proj=use_conv_proj, conv_kernel_size=conv_kernel_size)
        self.spatial_attn_h = SpatialAttention(d_model=out_channels, num_nodes=num_nodes, n_heads=n_heads, ffn_dim=ffn_dim, st_layers=st_layers, dropout_rate=st_dropout_rate, output_attention=output_attention, qkv_bias=qkv_bias, use_conv_proj=use_conv_proj, conv_kernel_size=conv_kernel_size)

        # GRU 的线性投影层
        # 将空间处理后的输入/隐藏状态投影到 3 倍维度，用于计算 reset, update, new 门
        gru_input_dim = out_channels
        self.gru_projection_i = nn.Linear(gru_input_dim, out_channels * 3)
        self.gru_projection_h = nn.Linear(gru_input_dim, out_channels * 3)

        # GRU 的偏置项 (每个门对应一个)
        self.bias_ir = Parameter(torch.Tensor(out_channels)); self.bias_ii = Parameter(torch.Tensor(out_channels)); self.bias_in = Parameter(torch.Tensor(out_channels))
        self.bias_hr = Parameter(torch.Tensor(out_channels)); self.bias_hi = Parameter(torch.Tensor(out_channels)); self.bias_hn = Parameter(torch.Tensor(out_channels))

        # 对 GRU 的输出隐藏状态进行 Layer Normalization
        self.ln = nn.LayerNorm(out_channels)
        # 初始化偏置参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 GRU 偏置"""
        stdv = 1. / math.sqrt(self.out_channels)
        for weight in [self.bias_ir, self.bias_ii, self.bias_in, self.bias_hr, self.bias_hi, self.bias_hn]:
            if weight is not None:
                init.uniform_(weight, -stdv, stdv)

    def forward(self, x, prev_hidden=None):
        """
        SpatialTemporalCell 的前向传播。
        Args:
            x (Tensor): 当前时间步的输入特征，形状为 (B, N, C)。
            prev_hidden (Tensor, optional): 上一时间步的隐藏状态，形状为 (B, N, C)。默认为 None (会初始化为零)。
        Returns:
            Tensor: 当前时间步的输出特征 (通常是归一化后的隐藏状态)，形状为 (B, N, C)。
            Tensor: 当前时间步更新后的隐藏状态，形状为 (B, N, C)。
            Tensor | None: 当前时间步的空间注意力图 (如果 output_attention=True)。
        """
        B, N, C = x.shape
        if C != self.out_channels: raise ValueError(f"输入通道 {C} != 预期 {self.out_channels}")

        # 如果没有提供上一时刻隐藏状态，则初始化为零
        if prev_hidden is None:
            prev_hidden = torch.zeros_like(x)
        elif prev_hidden.shape != x.shape:
            raise ValueError(f"隐藏状态形状 {prev_hidden.shape} != 输入形状 {x.shape}")

        # 1. 对当前输入 x 和上一时刻隐藏状态 prev_hidden 分别应用空间注意力
        input_sp_attn, input_attn_map = self.spatial_attn_i(x)
        hidden_sp_attn, hidden_attn_map = self.spatial_attn_h(prev_hidden)

        # 融合特征（在这个实现中，空间注意力直接替换了原始输入/隐藏状态）
        input_fused = input_sp_attn
        hidden_fused = hidden_sp_attn

        # 2. GRU 门计算准备：线性投影
        # 将输入和隐藏状态分别投影到 3 倍维度，然后切分成对应 r, i, n 门的部分
        input_r, input_i, input_n = self.gru_projection_i(input_fused).chunk(3, dim=-1)
        hidden_r, hidden_i, hidden_n = self.gru_projection_h(hidden_fused).chunk(3, dim=-1)

        # 3. 计算 GRU 门
        # 重置门 (reset gate): 控制上一时刻隐藏状态有多少要被忽略
        reset_gate = torch.sigmoid(input_r + self.bias_ir + hidden_r + self.bias_hr)
        # 更新门 (update gate): 控制当前隐藏状态有多少来自上一时刻，有多少来自新计算的状态
        update_gate = torch.sigmoid(input_i + self.bias_ii + hidden_i + self.bias_hi)
        # 新状态候选 (new gate): 计算基于当前输入和部分上一时刻隐藏状态的新信息
        new_gate = torch.tanh(input_n + self.bias_in + reset_gate * (hidden_n + self.bias_hn))

        # 4. 计算当前时刻的隐藏状态
        # H_t = (1 - z_t) * n_t + z_t * H_{t-1}
        next_hidden = (1.0 - update_gate) * new_gate + update_gate * prev_hidden

        # 5. 对输出的隐藏状态进行 Layer Normalization
        next_hidden_normalized = self.ln(next_hidden)
        # 通常将归一化后的隐藏状态作为当前时间步的输出
        output = next_hidden_normalized

        # 如果需要，返回空间注意力图（通常用输入的注意力图）
        final_attn = input_attn_map if self.output_attention and input_attn_map is not None else (hidden_attn_map if self.output_attention else None)

        return output, next_hidden, final_attn

# --- 时间 Transformer 块 (TemporalTransformerBlock) ---
class TemporalTransformerBlock(nn.Module):
    """
    一个简单的 Temporal Transformer Encoder Layer。
    在 GRU 处理完序列后，可选地用于捕捉全局时间依赖。
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(d_model)
        # 时间维度的多头自注意力 (batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True)
        self.drop_path1 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model)
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate)
        self.drop_path2 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        输入 x 的形状应为 (B, T, D)
        """
        # 1. Self-Attention + 残差 + Norm
        shortcut1 = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x) # Q, K, V 都是 x
        x = shortcut1 + self.drop_path1(attn_output)

        # 2. FFN + 残差 + Norm
        shortcut2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut2 + self.drop_path2(x)
        return x

# --- 主模型 (SDT_GRU_Classifier) ---
class SDT_GRU_Classifier(nn.Module):
    """
    时空 GRU 分类器，结合了 SpatialTemporalCell 和可选的 TemporalTransformerBlock。
    """
    def __init__(self, model_cfg):
        """
        初始化模型。
        Args:
            model_cfg (dict): 包含模型配置参数的字典。
        """
        super().__init__()
        print("初始化 SDT_GRU_Classifier (Scaled Dot-Product Attention with DropPath)...")
        try:
            # 打印配置信息以便调试
            print(yaml.dump(model_cfg, default_flow_style=None, sort_keys=False))
        except Exception:
            print(model_cfg)

        # --- 解析通用参数 ---
        self.num_input_dim = model_cfg['num_input_dim']
        self.num_nodes = model_cfg['num_nodes']
        self.num_classes = model_cfg['num_classes']
        self.num_rnn_layers = model_cfg['num_rnn_layers']
        self.num_rnn_units = model_cfg['num_rnn_units'] # GRU/内部特征维度 D
        self.n_heads = model_cfg['n_heads'] # 空间注意力头数
        self.ffn_dim = model_cfg['ffn_dim'] # 空间注意力 FFN 维度
        self.st_layers = model_cfg['st_layers'] # 每个 SpatialTemporalCell 内的空间注意力层数
        self.st_dropout_rate = model_cfg.get('st_dropout_rate', 0.1) # 传递给空间注意力的 DropPath
        self.rnn_dropout_rate = model_cfg.get('rnn_dropout_rate', 0.1) # GRU 层间的 Dropout
        self.classifier_dropout = model_cfg.get('classifier_dropout', 0.5) # 分类器前的 Dropout
        self.output_attention = model_cfg.get('output_attention', False) # 是否输出注意力图
        self.use_gap = model_cfg.get('use_gap', True) # 是否在最后聚合时使用全局平均池化
        self.classifier_hidden_dim = model_cfg.get('classifier_hidden_dim', 0) # 分类器中间隐藏层维度
        self.qkv_bias = model_cfg.get('qkv_bias', False) # 空间注意力 QKV 是否用偏置
        self.use_conv_proj = model_cfg.get('use_conv_proj', True) # 空间注意力是否用 Conv1D 投影
        self.conv_kernel_size = model_cfg.get('conv_kernel_size', 3) # Conv1D 核大小

        # --- 解析时间注意力参数 ---
        self.use_temporal_attn = model_cfg.get('use_temporal_attn', False)
        self.num_temporal_layers = model_cfg.get('num_temporal_layers', 1)
        self.temporal_n_heads = model_cfg.get('temporal_n_heads', 8)
        self.temporal_ffn_dim = model_cfg.get('temporal_ffn_dim', 256)
        self.temporal_dropout_rate = model_cfg.get('temporal_dropout_rate', 0.1) # 传递给时间注意力的 DropPath

        # --- 输入嵌入和位置编码 ---
        self.input_embedding = nn.Linear(self.num_input_dim, self.num_rnn_units)
        self.input_ln = nn.LayerNorm(self.num_rnn_units)
        self.input_dropout = nn.Dropout(self.st_dropout_rate) # 输入部分的标准 Dropout
        self.pos_encoder = PositionalEmbedding(self.num_rnn_units, max_len=model_cfg.get('max_seq_len', 200))
        print(f"输入维度: {self.num_input_dim}, RNN/特征维度: {self.num_rnn_units}")
        print(f"时间位置编码器 (max_len={model_cfg.get('max_seq_len', 200)}, d_model={self.num_rnn_units})")

        # --- 核心 GRU 编码器单元 ---
        self.encoder_cells = nn.ModuleList()
        for i in range(self.num_rnn_layers):
            # 使用恢复后的 SpatialTemporalCell (内部是 Scaled Dot-Product Attention)
            self.encoder_cells.append(
                SpatialTemporalCell(
                    in_channels=self.num_rnn_units, out_channels=self.num_rnn_units,
                    num_nodes=self.num_nodes, n_heads=self.n_heads, ffn_dim=self.ffn_dim,
                    st_layers=self.st_layers, st_dropout_rate=self.st_dropout_rate, # 传递给内部 DropPath
                    output_attention=self.output_attention, qkv_bias=self.qkv_bias,
                    use_conv_proj=self.use_conv_proj, conv_kernel_size=self.conv_kernel_size
                )
            )
        print(f"已创建 {self.num_rnn_layers} 个 SpatialTemporalCell (使用 Scaled Dot-Product 空间注意力)")

        # --- 可选：时间 Transformer 块 ---
        self.temporal_attn_blocks = None
        if self.use_temporal_attn:
            if self.num_rnn_units % self.temporal_n_heads != 0:
                 raise ValueError(f"Temporal Attention: num_rnn_units ({self.num_rnn_units}) 必须能被 temporal_n_heads ({self.temporal_n_heads}) 整除")
            print(f"启用 Temporal Attention ({self.num_temporal_layers} 层, {self.temporal_n_heads} 头, FFN dim {self.temporal_ffn_dim})")
            # TemporalTransformerBlock 内部已经使用了 DropPath
            self.temporal_attn_blocks = nn.ModuleList([
                TemporalTransformerBlock(
                    d_model=self.num_rnn_units,
                    n_heads=self.temporal_n_heads,
                    ffn_dim=self.temporal_ffn_dim,
                    dropout_rate=self.temporal_dropout_rate # 传递给内部 DropPath
                ) for _ in range(self.num_temporal_layers)
            ])
        else:
            print("Temporal Attention 未启用。")

        # --- 分类头 ---
        self.final_dropout = nn.Dropout(self.classifier_dropout) # 分类器前的 Dropout
        in_features_classifier = self.num_rnn_units
        classifier_layers = []
        if self.classifier_hidden_dim > 0:
             # 带隐藏层的分类器
             classifier_layers.append(nn.Linear(in_features_classifier, self.classifier_hidden_dim))
             classifier_layers.append(nn.ReLU()) # 或者其他激活函数
             classifier_layers.append(nn.Dropout(self.classifier_dropout)) # 隐藏层后的 Dropout
             classifier_layers.append(nn.Linear(self.classifier_hidden_dim, self.num_classes))
        else:
             # 直接线性分类器
             classifier_layers.append(nn.Linear(in_features_classifier, self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        print(f"分类器输入维度: {in_features_classifier}")

        self.global_step = 0 # 用于 TensorBoard 记录步数
        print("应用权重初始化...")
        self.apply(self._init_weights) # 应用权重初始化
        print("权重初始化完成。")

    def _init_weights(self, module):
        """递归地初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 对线性层使用截断正态分布初始化
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                # 偏置初始化为 0
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 的仿射参数初始化
            if module.elementwise_affine:
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
             # 对卷积层使用 Kaiming 初始化
             nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
             if module.bias is not None:
                 nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
             # 对多头注意力层的投影权重进行初始化
             if module.in_proj_weight is not None:
                 nn.init.xavier_uniform_(module.in_proj_weight)
             if module.out_proj.weight is not None:
                 nn.init.xavier_uniform_(module.out_proj.weight)
             # 对偏置初始化为 0
             if module.in_proj_bias is not None:
                 nn.init.constant_(module.in_proj_bias, 0.)
             if module.out_proj.bias is not None:
                 nn.init.constant_(module.out_proj.bias, 0.)

    def forward(self, x, mask=None):
        """
        模型的前向传播。
        Args:
            x (Tensor): 输入骨骼序列，形状 (B, T, N, C)。
            mask (Tensor, optional): 布尔掩码，标记有效时间步，形状 (B, T)。默认为 None。
        Returns:
            Tensor: 分类 logits，形状 (B, num_classes)。
            Tensor | None: 最后一层或最后一个时间步的空间注意力图 (如果 output_attention=True)。
        """
        B, T_in, N, C_in = x.shape
        # 检查输入维度是否与配置匹配
        if N != self.num_nodes: raise ValueError(f"输入节点数 {N} != 预期 {self.num_nodes}")
        if C_in != self.num_input_dim: raise ValueError(f"输入维度 {C_in} != 预期 {self.num_input_dim}")

        # 1. 输入处理：嵌入、归一化、Dropout、位置编码
        x_flat = x.view(-1, C_in)
        x_embedded_flat = self.input_embedding(x_flat)
        x_embedded = x_embedded_flat.view(B, T_in, N, self.num_rnn_units)
        x_processed = self.input_ln(x_embedded)
        x_processed = self.input_dropout(x_processed)
        T_current = x_processed.size(1)
        time_pe = self.pos_encoder(x_processed)
        time_pe_expanded = time_pe.unsqueeze(2).expand(-1, -1, N, -1) # (1, T, N, D)
        x_processed = x_processed + time_pe_expanded # 添加位置编码
        # 调整维度顺序以适应 GRU 处理：(T, B, N, D)
        x_processed = x_processed.permute(1, 0, 2, 3).contiguous()
        T_proc = x_processed.size(0)

        # 2. 核心 GRU 编码（带空间注意力）
        hidden_states = [None] * self.num_rnn_layers # 初始化每层 GRU 的隐藏状态
        all_outputs = [] # 存储每个时间步最后一层 GRU 的输出
        all_attns = [] if self.output_attention else None # 存储空间注意力图

        current_input_seq = x_processed # (T, B, N, D)

        # 按时间步循环
        for t in range(T_proc):
            input_t_layer = current_input_seq[t] # 获取当前时间步输入 (B, N, D)
            layer_attns = [] if self.output_attention else None

            # 逐层通过 SpatialTemporalCell
            for i, rnn_cell_module in enumerate(self.encoder_cells):
                prev_hidden_state = hidden_states[i]
                # 调用 SpatialTemporalCell 进行计算
                output_cell, next_hidden_cell, attn_cell = rnn_cell_module(input_t_layer, prev_hidden_state)
                # 更新隐藏状态
                hidden_states[i] = next_hidden_cell
                # 当前层的输出作为下一层的输入
                input_t_layer = output_cell
                # 在 GRU 层之间应用 Dropout (除最后一层外)
                if i < self.num_rnn_layers - 1 and self.rnn_dropout_rate > 0.0:
                    input_t_layer = F.dropout(input_t_layer, p=self.rnn_dropout_rate, training=self.training)
                # 收集空间注意力图（如果需要）
                if self.output_attention and attn_cell is not None:
                    layer_attns.append(attn_cell)

            # 存储最后一层 GRU 在当前时间步的输出
            all_outputs.append(input_t_layer)
            # 存储当前时间步所有层的空间注意力图
            if self.output_attention and layer_attns:
                all_attns.append(torch.stack(layer_attns, dim=1)) # (B, num_rnn_layers, H, N, N)

        # 将所有时间步的输出堆叠起来 (T, B, N, D)
        stacked_outputs = torch.stack(all_outputs, dim=0)

        # 3. 掩码处理：获取最后一个有效时间步的输出
        final_encoder_output_gru = None
        if mask is not None:
            # 确保掩码长度与处理后的序列长度一致
            if mask.shape[1] != T_proc:
                 if mask.shape[1] > T_proc:
                     mask = mask[:, :T_proc]
                 else: # 掩码比序列短，这通常不应该发生，忽略掩码
                     mask = None
                 if mask is None:
                     logger.warning(f"Mask length mismatch with sequence length ({T_proc}). Ignoring mask.")

            if mask is not None:
                # 计算每个样本的有效长度
                valid_lengths = mask.sum(dim=1) # (B,)
                # 处理可能存在的长度为 0 的情况
                if (valid_lengths == 0).any():
                    logger.warning("部分样本的有效长度为 0，将使用第一个时间步的输出。")
                    # 对于长度为 0 的样本，索引设为 0
                    last_valid_indices = torch.zeros_like(valid_lengths)
                    # 对于长度大于 0 的样本，计算最后一个有效索引
                    non_zero_mask = valid_lengths > 0
                    last_valid_indices[non_zero_mask] = (valid_lengths[non_zero_mask] - 1).clamp(min=0).long()
                else:
                    # 计算最后一个有效时间步的索引
                    last_valid_indices = (valid_lengths - 1).clamp(min=0).long() # (B,)

                # 使用高级索引从 stacked_outputs 中提取对应时间步的输出
                batch_indices = torch.arange(B, device=x.device)
                try:
                   # (T, B, N, D) -> (B, N, D)
                   final_encoder_output_gru = stacked_outputs[last_valid_indices, batch_indices, :, :]
                except IndexError as e:
                    logger.error(f"使用 last_valid_indices 访问 stacked_outputs 时索引错误。形状: stacked={stacked_outputs.shape}, indices={last_valid_indices.shape}, batch_indices={batch_indices.shape}. Error: {e}")
                    # 回退到使用最后一个时间步
                    final_encoder_output_gru = stacked_outputs[-1]
            else: # 如果掩码无效或未提供
                final_encoder_output_gru = stacked_outputs[-1] # 使用最后一个时间步
        else: # 如果未提供掩码
            final_encoder_output_gru = stacked_outputs[-1] # 使用最后一个时间步

        # 4. 可选的时间注意力处理
        final_representation = None # 初始化最终用于分类的特征向量
        if self.use_temporal_attn and self.temporal_attn_blocks is not None:
            # 需要将 GRU 输出调整为 (B, T, D) 格式送入时间注意力
            # 先转置 (B, T, N, D)
            gru_output_btnd = stacked_outputs.permute(1, 0, 2, 3)
            # 在节点维度上进行聚合（例如平均）
            if mask is not None:
                # 只聚合有效时间步的节点信息
                mask_expanded_btn1 = mask.unsqueeze(-1).float() # (B, T, 1)
                temporal_input_unmasked = gru_output_btnd.mean(dim=2) # (B, T, D)
                # 将填充帧的特征置零
                temporal_input = temporal_input_unmasked * mask_expanded_btn1
            else:
                temporal_input = gru_output_btnd.mean(dim=2) # (B, T, D)

            # 通过时间 Transformer 块
            temporal_output = temporal_input
            for temp_block in self.temporal_attn_blocks:
                temporal_output = temp_block(temporal_output) # (B, T, D)

            # 聚合时间维度特征，得到最终表示 (B, D)
            if mask is not None:
                # 使用掩码进行加权平均池化
                mask_t = mask.unsqueeze(-1).float() # (B, T, 1)
                masked_temporal_output = temporal_output * mask_t
                # 计算有效时间步数量，避免除零
                time_step_count = mask_t.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1, 1)
                final_representation = masked_temporal_output.sum(dim=1) / time_step_count.squeeze(-1) # (B, D)
            else:
                # 如果没有掩码，则直接对时间维度求平均
                final_representation = temporal_output.mean(dim=1) # (B, D)

        else: # 如果不使用时间注意力
            # 使用 GRU 最后一个有效时间步的输出进行聚合
            # final_encoder_output_gru 的形状是 (B, N, D)
            if self.use_gap:
                # 对节点维度进行全局平均池化
                final_representation = final_encoder_output_gru.mean(dim=1) # (B, D)
            else:
                # 如果不用 GAP，通常会将节点和特征维度展平 (B, N*D)
                # 但这与后续分类器期望的输入维度 D 不符，所以通常还是用 GAP
                logger.warning("use_temporal_attn=False 且 use_gap=False。强制在分类器前使用 GAP。")
                final_representation = final_encoder_output_gru.mean(dim=1) # (B, D)

        # 5. 分类
        # 应用最后的 Dropout
        final_representation = self.final_dropout(final_representation)
        # 送入分类器得到 logits
        logits = self.classifier(final_representation) # (B, num_classes)

        # 6. 处理注意力图输出
        final_attns = None
        # 如果需要输出注意力图，并且已经收集了它们
        if self.output_attention and all_attns is not None and all_attns:
             # all_attns 是一个列表，每个元素是形状 (B, num_rnn_layers, H, N, N) 的张量 (对应一个时间步)
             # 将它们沿着时间维度 (dim=1) 堆叠起来
             try:
                 # 最终形状 (B, T, num_rnn_layers, H, N, N)
                 final_attns = torch.stack(all_attns, dim=1)
             except Exception as e:
                 # 如果堆叠失败（例如，因为某些时间步没有注意力图），记录警告
                 logger.warning(f"无法堆叠空间注意力图: {e}")
                 final_attns = None

        # 训练时更新全局步数
        if self.training:
             # 检查 self 是否有 global_step 属性，以防万一
             if hasattr(self, 'global_step'):
                 self.global_step += 1
             else:
                 # 如果没有，可以初始化一个，但这通常应该在 __init__ 中完成
                 self.global_step = 1

        return logits, final_attns


# --- 示例用法 ---
if __name__ == "__main__":
    print("测试 SDT_GRU_Classifier 结构 (Scaled Dot-Product Attention with DropPath)...")
    # 定义一个基础配置字典用于测试
    cfg_base = {
        'num_input_dim': 12, 'num_nodes': 20, 'num_classes': 10, 'num_rnn_layers': 2, 'num_rnn_units': 128,
        'n_heads': 8, 'ffn_dim': 256, 'st_layers': 2, 'st_dropout_rate': 0.15, 'rnn_dropout_rate': 0.15,
        'classifier_dropout': 0.4, 'max_seq_len': 64, 'use_conv_proj': True, 'conv_kernel_size': 3, 'qkv_bias': False,
        'use_temporal_attn': True, 'num_temporal_layers': 1, 'temporal_n_heads': 8, 'temporal_ffn_dim': 512, 'temporal_dropout_rate': 0.15,
        'use_gap': True, 'classifier_hidden_dim': 64, 'output_attention': False
    }
    # 实例化模型
    model_base = SDT_GRU_Classifier(cfg_base)
    print("\n模型结构 (Scaled Dot-Product 空间注意力):")
    # 计算并打印可训练参数量
    param_count_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"\n模型可训练参数量: {param_count_base:,}")

    # 创建虚拟输入数据进行测试
    B, T, N, C_in = 4, 64, 20, 12 # 批次, 时间, 节点, 输入通道
    dummy_input = torch.randn(B, T, N, C_in)
    # 创建虚拟掩码 (第一个样本最后 10 帧是填充)
    dummy_mask = torch.ones(B, T, dtype=torch.bool)
    dummy_mask[0, -10:] = False
    print(f"\n输入形状: {dummy_input.shape}")

    # 设置为评估模式并进行前向传播
    model_base.eval()
    with torch.no_grad():
        logits_base, _ = model_base(dummy_input, mask=dummy_mask)
    # 打印输出形状
    print(f"输出 logits 形状: {logits_base.shape}") # 应该等于 (B, num_classes) -> (4, 10)