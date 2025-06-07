# -*- coding: utf-8 -*-
# 文件名: model/SDT_GRUs_Gesture.py (v15.1 - 删除TA模块)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
import yaml # 保留yaml用于打印配置
import logging
from einops import rearrange
from typing import Optional, Tuple, List # 用于类型注解

# 尝试导入 timm 的 DropPath 和 trunc_normal_，如果失败则使用内置替代
try:
    from timm.models.layers import trunc_normal_, DropPath
except ImportError:
    print("警告: 无法从 timm 导入 DropPath 或 trunc_normal_。将使用内置简化版本。")
    class DropPath(nn.Module): # 定义一个简单的DropPath替代品
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0. or not self.training: return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, 1, 1, ...)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_() # Binarize
            output = x.div(keep_prob) * random_tensor
            return output
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.): # 定义一个简单的截断正态分布初始化替代品
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(min=a, max=b)

model_logger = logging.getLogger("模型.SDT_BiGRU") # 为模型创建一个特定的日志记录器

# --- 位置编码 (PositionalEmbedding) ---
class PositionalEmbedding(nn.Module):
    """标准的正弦/余弦位置编码 (用于时间维度)"""
    def __init__(self, d_model, max_len=500): # d_model: 特征维度, max_len: 最大序列长度
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float() # 创建一个 (max_len, d_model) 的零张量
        pe.requires_grad = False # 位置编码通常是不可训练的

        position = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)，表示位置索引
        # 计算频率项的分母部分，基于Transformer论文中的公式
        div_term_exponent = torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        div_term = torch.exp(div_term_exponent) # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度使用sin
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度使用cos
        else:
            # 如果d_model是奇数，cos部分少一个维度
            if d_model // 2 > 0: # 确保至少有一个奇数位
                 pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        # pe 的形状是 (max_len, d_model)
        pe = pe.unsqueeze(0) # 增加批次维度 (1, max_len, d_model)
        self.register_buffer('pe', pe) # 将pe注册为模型的buffer，它会被保存但不会被视为模型参数

    def forward(self, x): # x 的形状通常是 (B, T, ...) 或 (B, T, N, D)
        # 我们需要截取与输入序列长度T相匹配的位置编码
        # x.size(1) 通常是时间维度 T
        return self.pe[:, :x.size(1)] # 返回 (1, T, d_model)

# --- 多头注意力层 (AttentionLayer) ---
class AttentionLayer(nn.Module):
    """
    空间注意力层，用于处理单帧内骨骼节点间的关系。
    支持多种配置，如卷积投影、相对位置偏置、局部/全局注意力等。
    """
    def __init__(self, d_model, n_heads, dropout, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20,
                 use_global_spatial_bias=False, attention_type='global', adj_matrix_for_local=None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"特征维度 d_model ({d_model}) 必须能被注意力头数 n_heads ({n_heads}) 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads # 每个头的K维度
        self.d_values = d_model // n_heads # 每个头的V维度
        self.output_attention = output_attention # 是否返回注意力权重图
        self.use_conv_proj = use_conv_proj # 是否用1D卷积生成Q,K,V
        self.conv_kernel_size = conv_kernel_size
        self.num_nodes = num_nodes # 骨骼节点数
        self.use_global_spatial_bias = use_global_spatial_bias # 是否使用全局可学习的空间偏置
        self.attention_type = attention_type # 'global' 或 'local'

        if adj_matrix_for_local is not None and self.attention_type == 'local':
            self.register_buffer("adj_matrix_for_local", adj_matrix_for_local.bool())
        else:
            self.register_buffer("adj_matrix_for_local", None)

        # QKV投影层
        if use_conv_proj: # 使用1D卷积进行QKV投影 (作用在节点序列上)
            padding = conv_kernel_size // 2
            self.query_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.key_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.value_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            # 卷积投影后通常会接LayerNorm和残差连接
            self.resid_norm_q = nn.LayerNorm(d_model)
            self.resid_norm_k = nn.LayerNorm(d_model)
            self.resid_norm_v = nn.LayerNorm(d_model)
        else: # 使用线性层进行QKV投影
            self.query_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.key_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.value_projection = nn.Linear(d_model, d_model, bias=qkv_bias)

        # 相对位置偏置 (学习节点间的相对空间关系)
        if num_nodes > 0:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * num_nodes - 1), n_heads))
            trunc_normal_(self.relative_position_bias_table, std=.02)
            coords_n = torch.arange(num_nodes)
            relative_coords_n = coords_n[:, None] - coords_n[None, :] + num_nodes - 1
            self.register_buffer("relative_position_index", relative_coords_n.long(), persistent=False)
        else:
            self.relative_position_bias_table = None
            self.register_buffer("relative_position_index", None, persistent=False)

        self.out_projection = nn.Linear(d_model, d_model) # 注意力输出后的最终投影
        self.dropout = nn.Dropout(dropout) # 应用于注意力权重 softmax(scores) 之后

        # 全局可学习的空间偏置 (如果启用)
        if self.use_global_spatial_bias and self.attention_type == 'global' and num_nodes > 0:
            self.global_spatial_bias = nn.Parameter(torch.zeros(n_heads, num_nodes, num_nodes))
            self.alpha_global_bias = nn.Parameter(torch.tensor(1.0)) # 偏置的缩放因子
            trunc_normal_(self.global_spatial_bias, std=.02)
        else:
            self.global_spatial_bias = None
            self.alpha_global_bias = None

        # 局部连接偏置 (如果启用，用于局部注意力)
        if self.attention_type == 'local' and num_nodes > 0:
            self.local_connection_bias = nn.Parameter(torch.zeros(self.n_heads, self.num_nodes, self.num_nodes))
        else:
            self.local_connection_bias = None

    def _get_relative_positional_bias(self) -> torch.Tensor:
        """计算并返回相对位置偏置，形状为 (1, H, N, N)"""
        if self.relative_position_bias_table is None or self.relative_position_index is None:
            return 0.0 # 如果没有配置，则偏置为0
        # self.relative_position_index 是 (N, N)
        # self.relative_position_bias_table 是 (2N-1, H)
        idx = self.relative_position_index.view(-1).to(self.relative_position_bias_table.device) # (N*N)
        # 从表中查找偏置
        relative_position_bias = self.relative_position_bias_table[idx] # (N*N, H)
        relative_position_bias = relative_position_bias.view(self.num_nodes, self.num_nodes, self.n_heads) # (N, N, H)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # (H, N, N)
        return relative_position_bias.unsqueeze(0) # (1, H, N, N) 以便广播

    def forward(self, x): # x 输入形状: (B, N, D) - B:批次大小, N:节点数, D:特征维度
        B, N, D = x.shape; H = self.n_heads
        if N != self.num_nodes:
            raise ValueError(f"输入节点维度 {N} 与 AttentionLayer 初始化的 num_nodes ({self.num_nodes}) 不符")

        # 1. QKV投影
        if self.use_conv_proj:
            x_permuted = x.permute(0, 2, 1).contiguous() # (B, D, N) 以适配Conv1d
            q_conv = self.query_projection(x_permuted)   # (B, D, N)
            k_conv = self.key_projection(x_permuted)   # (B, D, N)
            v_conv = self.value_projection(x_permuted)   # (B, D, N)
            # 转换回 (B, N, D) 并进行 LayerNorm 和残差连接
            q_normed = self.resid_norm_q(q_conv.permute(0, 2, 1).contiguous())
            k_normed = self.resid_norm_k(k_conv.permute(0, 2, 1).contiguous())
            v_normed = self.resid_norm_v(v_conv.permute(0, 2, 1).contiguous())
            # 残差连接：原始输入 x 加上经过卷积和归一化处理的变换结果
            queries_proj = x + q_normed
            keys_proj = x + k_normed
            values_proj = x + v_normed
        else: # 使用线性投影
             queries_proj = self.query_projection(x)
             keys_proj = self.key_projection(x)
             values_proj = self.value_projection(x)

        # 2. 将QKV分割到多头
        # (B, N, D) -> (B, N, H, D_head) -> (B, H, N, D_head)
        queries = queries_proj.view(B, N, H, self.d_keys).permute(0, 2, 1, 3)
        keys = keys_proj.view(B, N, H, self.d_keys).permute(0, 2, 1, 3)
        values = values_proj.view(B, N, H, self.d_values).permute(0, 2, 1, 3)

        # 3. 计算注意力分数
        # (B,H,N,D_k) @ (B,H,D_k,N) -> (B,H,N,N)
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_keys) # 缩放因子

        # 4. 添加各种偏置
        if self.relative_position_bias_table is not None:
            scores = scores + self._get_relative_positional_bias() # (1,H,N,N)

        if self.attention_type == 'global' and self.global_spatial_bias is not None:
            # self.global_spatial_bias 是 (H,N,N)，需要扩展到 (1,H,N,N)
            scores = scores + self.global_spatial_bias.unsqueeze(0) * self.alpha_global_bias

        if self.attention_type == 'local' and self.local_connection_bias is not None:
            # self.local_connection_bias 是 (H,N,N)
            scores = scores + self.local_connection_bias.unsqueeze(0)

        # 5. 应用局部注意力掩码 (如果配置)
        if self.attention_type == 'local' and self.adj_matrix_for_local is not None:
            # adj_matrix_for_local 是 (N,N)，True表示连接
            # 我们需要一个掩码，其中非连接处为 -inf。所以对 adj.logical_not() 进行扩展
            local_mask = self.adj_matrix_for_local.logical_not().unsqueeze(0).unsqueeze(0) # (1,1,N,N)
            scores = scores.masked_fill(local_mask, float('-inf'))

        # 6. 计算注意力权重并应用dropout
        attn_weights = torch.softmax(scores, dim=-1) # (B,H,N,N)
        attn_weights_dropped = self.dropout(attn_weights)

        # 7. 加权求和V，并进行输出投影
        # (B,H,N,N) @ (B,H,N,D_v) -> (B,H,N,D_v)
        weighted_values = torch.matmul(attn_weights_dropped, values)
        # (B,H,N,D_v) -> (B,N,H,D_v) -> (B,N, H*D_v = D)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        output = self.out_projection(weighted_values) # (B,N,D)

        return output, attn_weights.detach() if self.output_attention else None

# --- 前馈网络模块 (FeedForwardModule) ---
class FeedForwardModule(nn.Module):
    """标准的两层前馈网络，通常在自注意力之后使用"""
    def __init__(self, d_model, ffn_dim, activation=F.gelu, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)      # 第一层线性变换
        self.activation = activation                    # 激活函数
        self.dropout = nn.Dropout(dropout)              # Dropout层
        self.linear2 = nn.Linear(ffn_dim, d_model)      # 第二层线性变换
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# --- Transformer编码器层 (EncoderLayer) ---
class EncoderLayer(nn.Module):
    """
    一个标准的Transformer编码器层，包含一个自注意力模块和一个前馈网络模块。
    用于空间特征提取，所以这里的“编码器层”是针对单帧内节点而言的。
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.gelu, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20,
                 use_global_spatial_bias=False, attention_type='global', adj_matrix_for_local=None):
        super().__init__()
        # 自注意力模块
        self.self_attn = AttentionLayer(d_model, n_heads, dropout_rate, output_attention,
                                      qkv_bias, use_conv_proj, conv_kernel_size, num_nodes,
                                      use_global_spatial_bias, attention_type, adj_matrix_for_local)
        # 前馈网络模块
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate)
        # LayerNorm层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 残差连接后的Dropout (或DropPath)
        self.dropout_residual1 = nn.Dropout(dropout_rate) # 或者 DropPath(dropout_rate)
        self.dropout_residual2 = nn.Dropout(dropout_rate) # 或者 DropPath(dropout_rate)

    def forward(self, x): # x: (B, N, D)
        # 1. 自注意力部分 (Pre-LN 或 Post-LN 结构，这里是Post-LN后接残差)
        attn_output, attn_weights = self.self_attn(x)
        x_residual_after_attn = x + self.dropout_residual1(attn_output) # 残差连接1
        x_norm1 = self.norm1(x_residual_after_attn)                     # LayerNorm1

        # 2. 前馈网络部分
        ffn_output = self.ffn(x_norm1)
        x_residual_after_ffn = x_norm1 + self.dropout_residual2(ffn_output) # 残差连接2
        x_norm2 = self.norm2(x_residual_after_ffn)                          # LayerNorm2
        return x_norm2, attn_weights

# --- 动态融合门 (DynamicFusionGate) ---
class DynamicFusionGate(nn.Module):
    """用于融合两个特征向量（例如局部和全局空间特征）的动态门控机制"""
    def __init__(self, feature_dim, gate_hidden_dim=None, num_gates=1):
        super().__init__()
        final_gate_hidden_dim = gate_hidden_dim
        if final_gate_hidden_dim is None: # 如果未指定隐藏层维度，则自动计算
            final_gate_hidden_dim = max(16, feature_dim // 4) # 例如取输入特征维度的1/4，最小为16
        # 门控网络：一个小型MLP，输出sigmoid激活的门控值
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, final_gate_hidden_dim), # 输入是两个特征拼接
            nn.ReLU(),
            nn.Linear(final_gate_hidden_dim, num_gates),       # 输出门控值的数量
            nn.Sigmoid()                                       # 确保门控值在0-1之间
        )
    def forward(self, feat1, feat2): # feat1, feat2 形状通常是 (B, D) 或 (B, N, D)
        # 如果输入是 (B,N,D)，通常会先在N维度上池化再输入门控
        gate_input = torch.cat((feat1, feat2), dim=-1) # 在特征维度上拼接
        gate_values = self.gate_network(gate_input)    # (B, num_gates) 或 (B,N,num_gates)
        return gate_values

# --- 组合的空间注意力模块 (SpatialAttention) ---
class SpatialAttention(nn.Module):
    """
    封装了局部路径和全局路径的空间注意力，并通过指定方式融合它们的输出。
    这是在SpatialTemporalCell中用于处理单帧空间信息的核心模块。
    """
    def __init__(self, d_model, num_nodes, output_attention, # 基本参数
                 # 局部路径参数
                 local_st_layers, local_n_heads, local_ffn_dim, local_dropout_rate,
                 local_qkv_bias, local_use_conv_proj, local_conv_kernel_size,
                 # 全局路径参数
                 global_st_layers, global_n_heads, global_ffn_dim, global_dropout_rate,
                 global_qkv_bias, global_use_conv_proj, global_conv_kernel_size,
                 global_use_global_spatial_bias,
                 # 融合参数
                 fusion_type='dynamic_gate',
                 fusion_gate_hidden_dim=None,
                 adj_matrix_for_local=None): # 局部注意力所需的邻接矩阵
        super().__init__()
        self.output_attention = output_attention # 是否输出最终的全局注意力图
        self.fusion_type = fusion_type           # 局部和全局特征的融合方式

        # 创建局部路径的编码器层 (通常是多个EncoderLayer堆叠)
        self.local_path_encoders = nn.ModuleList()
        for _ in range(local_st_layers):
            self.local_path_encoders.append(
                EncoderLayer(d_model=d_model, n_heads=local_n_heads, ffn_dim=local_ffn_dim,
                             dropout_rate=local_dropout_rate, output_attention=False, # 局部路径通常不输出中间注意力图
                             qkv_bias=local_qkv_bias, use_conv_proj=local_use_conv_proj,
                             conv_kernel_size=local_conv_kernel_size, num_nodes=num_nodes,
                             use_global_spatial_bias=False, # 局部路径不使用全局偏置
                             attention_type='local', adj_matrix_for_local=adj_matrix_for_local)
            )

        # 创建全局路径的编码器层
        self.global_path_encoders = nn.ModuleList()
        for i in range(global_st_layers):
            # 只有全局路径的最后一层在需要时才输出注意力图
            current_output_attention = self.output_attention if i == global_st_layers - 1 else False
            self.global_path_encoders.append(
                EncoderLayer(d_model=d_model, n_heads=global_n_heads, ffn_dim=global_ffn_dim,
                             dropout_rate=global_dropout_rate, output_attention=current_output_attention,
                             qkv_bias=global_qkv_bias, use_conv_proj=global_use_conv_proj,
                             conv_kernel_size=global_conv_kernel_size, num_nodes=num_nodes,
                             use_global_spatial_bias=global_use_global_spatial_bias, # 全局路径可以使用全局偏置
                             attention_type='global', adj_matrix_for_local=None) # 全局路径不使用局部邻接矩阵
            )

        # 根据融合类型创建融合模块
        if self.fusion_type == 'dynamic_gate':
            # 动态门控融合，需要对局部和全局特征进行池化（通常在节点维度上取平均）
            # DynamicFusionGate 的输入维度是 d_model (因为池化后的特征维度是d_model)
            self.fusion_gate_module = DynamicFusionGate(feature_dim=d_model, gate_hidden_dim=fusion_gate_hidden_dim, num_gates=1)
        elif self.fusion_type == 'simple_sum_learnable_weights':
            self.alpha_fusion = nn.Parameter(torch.tensor(0.5)) # 可学习的融合权重
        elif self.fusion_type == 'concat_linear':
            self.fusion_projection = nn.Linear(d_model * 2, d_model) # 拼接后通过线性层降维
        else: # 其他融合方式，例如简单的平均
            pass

    def forward(self, x): # x: (B, N, D)
        # 1. 局部路径处理
        x_l = x # 局部路径的输入
        for encoder in self.local_path_encoders:
            x_l, _ = encoder(x_l) # (B,N,D)

        # 2. 全局路径处理
        x_g = x # 全局路径的输入
        final_global_attn_weights = None # 用于存储最后一层全局注意力的权重图
        for i, encoder in enumerate(self.global_path_encoders):
            x_g, attn_weights = encoder(x_g) # (B,N,D)
            if i == len(self.global_path_encoders) - 1 and self.output_attention: # 如果是最后一层且需要输出注意力
                final_global_attn_weights = attn_weights

        # 3. 融合局部和全局路径的输出
        if self.fusion_type == 'dynamic_gate':
            # 对节点维度进行平均池化，得到 (B,D) 的特征用于计算门控值
            pooled_l = x_l.mean(dim=1) # (B,D)
            pooled_g = x_g.mean(dim=1) # (B,D)
            gate_value_g = self.fusion_gate_module(pooled_l, pooled_g) # (B,1) -> 全局特征的权重
            # 将门控值扩展回 (B,1,1) 以便在节点和特征维度上广播
            gate_value_g_expanded = gate_value_g.unsqueeze(-1) # (B,1,1)
            # 动态加权融合: (1-gate)*local_feat + gate*global_feat
            fused_output = (1.0 - gate_value_g_expanded) * x_l + gate_value_g_expanded * x_g
        elif self.fusion_type == 'simple_sum_learnable_weights':
            fused_output = self.alpha_fusion * x_l + (1.0 - self.alpha_fusion) * x_g
        elif self.fusion_type == 'concat_linear':
            fused_output = self.fusion_projection(torch.cat((x_l, x_g), dim=-1))
        else: # 默认使用简单平均融合
            fused_output = (x_l + x_g) / 2.0

        return fused_output, final_global_attn_weights # 返回融合后的特征和可选的注意力图


# --- 时间处理模块 (Provided_TA, MultiScaleTemporalModeling, TemporalTransformerBlock) ---
# 这些模块的定义与你之前提供的保持一致，这里不再重复，以节省篇幅。
# 确保它们的 __init__ 和 forward 方法中的输入输出维度与 SDT_BiGRU_Classifier 中的调用匹配。
# 例如，它们的 input_dim 参数应该接收 self.gru_output_dim。

# (假设 Provided_TA, MultiScaleTemporalModeling, TemporalTransformerBlock 的定义在这里)


class MultiScaleTemporalModeling(nn.Module):
    def __init__(self, input_dim, output_dim, short_term_kernels, long_term_kernels, long_term_dilations, conv_out_channels_ratio=0.5, fusion_hidden_dim_ratio=1.0, dropout_rate=0.1):
        super(MultiScaleTemporalModeling, self).__init__()
        self.short_term_convs = nn.ModuleList()
        short_out_c = max(1, int(input_dim * conv_out_channels_ratio))
        for ks in short_term_kernels: self.short_term_convs.append(nn.Sequential(nn.Conv1d(input_dim,short_out_c,ks,padding=(ks-1)//2,bias=False),nn.BatchNorm1d(short_out_c),nn.ReLU(),nn.Dropout(dropout_rate)))
        self.long_term_convs = nn.ModuleList()
        long_out_c = max(1, int(input_dim * conv_out_channels_ratio))
        for i, ks in enumerate(long_term_kernels):
            dil = long_term_dilations[i]; pad = (ks-1)*dil//2
            self.long_term_convs.append(nn.Sequential(nn.Conv1d(input_dim,long_out_c,ks,padding=pad,dilation=dil,bias=False),nn.BatchNorm1d(long_out_c),nn.ReLU(),nn.Dropout(dropout_rate)))
        total_conv_out_channels = short_out_c*len(short_term_kernels) + long_out_c*len(long_term_kernels)
        fusion_in_dim = input_dim + total_conv_out_channels
        fusion_hidden_d = max(1, int(fusion_in_dim * fusion_hidden_dim_ratio))
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_in_dim,fusion_hidden_d),nn.ReLU(),nn.LayerNorm(fusion_hidden_d),nn.Dropout(dropout_rate),nn.Linear(fusion_hidden_d,output_dim))
        model_logger.info(f"MultiScaleTemporalModeling: {len(short_term_kernels)+len(long_term_kernels)} conv branches, total_conv_out_c: {total_conv_out_channels}, fusion_mlp_in: {fusion_in_dim}, fusion_mlp_out: {output_dim}")
    def forward(self, x_btd, mask=None):
        x_bdt = x_btd.permute(0,2,1)
        branch_outputs = [x_btd]
        for conv in self.short_term_convs: branch_outputs.append(conv(x_bdt).permute(0,2,1))
        for conv in self.long_term_convs: branch_outputs.append(conv(x_bdt).permute(0,2,1))
        fused = self.fusion_mlp(torch.cat(branch_outputs, dim=-1))
        if mask is not None: fused = fused * mask.unsqueeze(-1).float()
        return fused

class TemporalTransformerBlock(nn.Module):
    """标准的时间 Transformer 编码器层，用于捕捉时间序列中的长距离依赖。"""
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(d_model) # 第一个LayerNorm (Pre-LN中的一部分)
        # 多头自注意力，batch_first=True 表示输入形状为 (B, T, D)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True)
        # DropPath 用于残差连接，类似于Dropout，但作用于整个路径
        self.drop_path1 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model) # 第二个LayerNorm
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate) # 前馈网络
        self.drop_path2 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()

    def forward(self, x, key_padding_mask=None): # x: (B, T, D_model)
        # key_padding_mask: (B, T), True表示对应位置是padding，注意力计算时会忽略
        shortcut1 = x
        x_norm1 = self.norm1(x)
        # Q, K, V 都来自 x_norm1
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1, key_padding_mask=key_padding_mask)
        x_after_attn = shortcut1 + self.drop_path1(attn_output) # 残差连接 + DropPath

        shortcut2 = x_after_attn
        x_norm2 = self.norm2(x_after_attn)
        ffn_output = self.ffn(x_norm2)
        x_after_ffn = shortcut2 + self.drop_path2(ffn_output) # 残差连接 + DropPath
        return x_after_ffn

# --- 主模型: 双向空间时间GRU分类器 (SDT_BiGRU_Classifier) ---
class SDT_BiGRU_Classifier(nn.Module):
    def __init__(self, model_cfg: dict):
        """
        解耦的时空分类器模型初始化。
        处理流程: 输入嵌入 -> 逐帧空间特征提取 (SpatialAttention) -> 空间池化 ->
                   时间位置编码 -> 主要时间序列建模 (GRU或Transformer) ->
                   可选的后续时间模块 -> 分类头。
        这个版本已完全移除 Provided_TA (FSTA-inspired) 模块。

        参数:
            model_cfg (dict): 包含所有模型配置的字典。
        """
        super().__init__()
        # model_logger.info(f"模型配置 (model_cfg): \n{yaml.dump(model_cfg, indent=2, sort_keys=False)}") # 可选：打印完整配置

        # --- 1. 基础参数 ---
        self.num_input_dim: int = model_cfg['num_input_dim']         # 输入骨骼数据的原始特征维度
        self.num_nodes: int = model_cfg['num_nodes']                 # 骨骼节点数
        self.num_classes: int = model_cfg['num_classes']             # 分类任务的类别数
        self.max_seq_len: int = model_cfg.get('max_seq_len', 100)    # 输入序列的最大长度

        # --- 2. 输入嵌入层 ---
        self.embedding_dim: int = model_cfg.get('embedding_dim', 128) # 空间处理和时间模型交互的特征维度
        self.input_embedding = nn.Linear(self.num_input_dim, self.embedding_dim)
        self.input_ln = nn.LayerNorm(self.embedding_dim)
        self.input_dropout = nn.Dropout(model_cfg.get('input_dropout_rate', 0.1))

        # --- 3. 逐帧空间特征提取器 (使用 SpatialAttention) ---
        adj_matrix = self._create_local_adj_matrix( # 创建局部注意力路径所需的邻接矩阵
            self.num_nodes, k_hop=model_cfg.get('local_adj_k_hop', 1)
        )
        spatial_attention_cfg = { # SpatialAttention 的配置参数
            'd_model': self.embedding_dim,
            'num_nodes': self.num_nodes,
            'output_attention': model_cfg.get('output_spatial_attention', False),
            'local_st_layers': model_cfg.get('sa_local_st_layers', 1),
            'local_n_heads': model_cfg.get('sa_local_n_heads', 4),
            'local_ffn_dim': model_cfg.get('sa_local_ffn_dim', self.embedding_dim * 2),
            'local_dropout_rate': model_cfg.get('sa_local_dropout_rate', 0.1),
            'local_qkv_bias': model_cfg.get('sa_local_qkv_bias', False),
            'local_use_conv_proj': model_cfg.get('sa_local_use_conv_proj', False), # 默认禁用卷积投影
            'local_conv_kernel_size': model_cfg.get('sa_local_conv_kernel_size', 3),
            'global_st_layers': model_cfg.get('sa_global_st_layers', 1),
            'global_n_heads': model_cfg.get('sa_global_n_heads', 4),
            'global_ffn_dim': model_cfg.get('sa_global_ffn_dim', self.embedding_dim * 2),
            'global_dropout_rate': model_cfg.get('sa_global_dropout_rate', 0.1),
            'global_qkv_bias': model_cfg.get('sa_global_qkv_bias', False),
            'global_use_conv_proj': model_cfg.get('sa_global_use_conv_proj', False), # 默认禁用卷积投影
            'global_conv_kernel_size': model_cfg.get('sa_global_conv_kernel_size', 3),
            'global_use_global_spatial_bias': model_cfg.get('sa_global_use_global_spatial_bias', True),
            'fusion_type': model_cfg.get('spatial_fusion_type', 'dynamic_gate'),
            'fusion_gate_hidden_dim': model_cfg.get('spatial_fusion_gate_hidden_dim', None),
            'adj_matrix_for_local': adj_matrix
        }
        self.spatial_feature_extractor = SpatialAttention(**spatial_attention_cfg)
        self.d_spatial_pooled: int = self.embedding_dim # 空间池化后的维度与embedding_dim相同
        model_logger.info(f"    局部路径卷积投影: {'启用' if spatial_attention_cfg['local_use_conv_proj'] else '禁用'}")
        model_logger.info(f"    全局路径卷积投影: {'启用' if spatial_attention_cfg['global_use_conv_proj'] else '禁用'}")

        # --- 4. 时间位置编码 ---
        self.use_time_pos_enc: bool = model_cfg.get('use_time_pos_enc', True)
        if self.use_time_pos_enc:
            self.time_pos_encoder = PositionalEmbedding(self.d_spatial_pooled, max_len=self.max_seq_len)
        else:
            self.time_pos_encoder = None # 明确设为None

        # --- 5. 主要时间序列建模模块 (nn.GRU 或 nn.TransformerEncoder) ---
        self.temporal_model_type: str = model_cfg.get('temporal_model_type', 'gru').lower()
        self.temporal_hidden_dim: int = model_cfg.get('temporal_hidden_dim', self.d_spatial_pooled)
        self.num_temporal_main_layers: int = model_cfg.get('num_temporal_main_layers', 2)
        temporal_main_dropout: float = model_cfg.get('temporal_main_dropout', 0.1)

        if self.temporal_model_type == 'gru':
            self.bidirectional_time_gru: bool = model_cfg.get('bidirectional_time_gru', True)
            self.temporal_encoder = nn.GRU(
                input_size=self.d_spatial_pooled,
                hidden_size=self.temporal_hidden_dim,
                num_layers=self.num_temporal_main_layers,
                bidirectional=self.bidirectional_time_gru,
                batch_first=True,
                dropout=temporal_main_dropout if self.num_temporal_main_layers > 1 else 0.0
            )
            self.d_temporal_final_out: int = self.temporal_hidden_dim * 2 if self.bidirectional_time_gru else self.temporal_hidden_dim
            model_logger.info(f"  主要时间模型: GRU (双向={self.bidirectional_time_gru}, 层数={self.num_temporal_main_layers}, 隐藏单元={self.temporal_hidden_dim})")
        elif self.temporal_model_type == 'transformer':
            tf_n_heads: int = model_cfg.get('temporal_main_n_heads', 4)
            tf_ffn_dim: int = model_cfg.get('temporal_main_ffn_dim', self.d_spatial_pooled * 4)
            if self.d_spatial_pooled % tf_n_heads != 0:
                raise ValueError(f"时间Transformer的d_model ({self.d_spatial_pooled}) 必须能被头数 ({tf_n_heads}) 整除。")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_spatial_pooled, nhead=tf_n_heads, dim_feedforward=tf_ffn_dim,
                dropout=temporal_main_dropout, activation=model_cfg.get('temporal_main_activation', 'relu'), batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_temporal_main_layers)
            self.d_temporal_final_out: int = self.d_spatial_pooled
            model_logger.info(f"  主要时间模型: TransformerEncoder (层数={self.num_temporal_main_layers}, 头数={tf_n_heads}, d_model={self.d_spatial_pooled})")
        else:
            raise ValueError(f"不支持的时间模型类型: '{self.temporal_model_type}'。")
        model_logger.info(f"  主要时间模型输出维度: {self.d_temporal_final_out}")

        # --- 6. 可选的、在主要时间模型之后的附加时间处理模块 ---
        current_dim_for_downstream: int = self.d_temporal_final_out

        # MultiScaleTemporalModeling 模块
        self.multiscale_temporal_processor = None
        self.use_multiscale_temporal_after_main: bool = model_cfg.get('use_multiscale_temporal_after_main', False)
        if self.use_multiscale_temporal_after_main:
            mst_args = model_cfg.get('multiscale_temporal_args')
            if mst_args:
                self.multiscale_temporal_processor = MultiScaleTemporalModeling(
                    input_dim=current_dim_for_downstream, output_dim=current_dim_for_downstream, **mst_args)
                model_logger.info(f"  可选后续模块: MultiScaleTemporalModeling 已启用 (输入维度: {current_dim_for_downstream})。")
            else:
                model_logger.warning("  配置中启用了 use_multiscale_temporal_after_main 但缺少 multiscale_temporal_args，模块将禁用。")
                self.use_multiscale_temporal_after_main = False

        # 后续的 TemporalTransformerBlock 模块
        self.temporal_attn_blocks_after_main = None
        self.num_temporal_layers_after_main: int = 0
        self.use_temporal_attn_after_main_specific: bool = model_cfg.get('use_temporal_attn_after_main_specific', False)
        if self.use_temporal_attn_after_main_specific:
            self.num_temporal_layers_after_main = model_cfg.get('num_temporal_layers_after_main', 0)
            if self.num_temporal_layers_after_main > 0:
                n_h: int = model_cfg.get('temporal_after_main_n_heads', 4)
                ffn_d: int = model_cfg.get('temporal_after_main_ffn_dim', current_dim_for_downstream * 4)
                drop_r: float = model_cfg.get('temporal_after_main_dropout', 0.1)
                if current_dim_for_downstream % n_h != 0:
                    model_logger.warning(f"后续时间Transformer维度 ({current_dim_for_downstream}) 不能被头数 ({n_h}) 整除。")
                self.temporal_attn_blocks_after_main = nn.ModuleList([
                    TemporalTransformerBlock(current_dim_for_downstream, n_h, ffn_d, drop_r)
                    for _ in range(self.num_temporal_layers_after_main)])
                model_logger.info(f"  可选后续模块: TemporalTransformerBlock ({self.num_temporal_layers_after_main} 层) 已启用 (输入维度: {current_dim_for_downstream})。")
            else:
                self.use_temporal_attn_after_main_specific = False

        # --- 7. 分类头 ---
        self.classifier_input_dim_final: int = current_dim_for_downstream
        self.classifier_hidden_dim: int = model_cfg.get('classifier_hidden_dim', 0)
        self.classifier_dropout_rate: float = model_cfg.get('classifier_dropout', 0.3)
        classifier_layers = []
        if self.classifier_hidden_dim > 0:
            classifier_layers.extend([
                nn.Linear(self.classifier_input_dim_final, self.classifier_hidden_dim), nn.ReLU(),
                nn.Dropout(self.classifier_dropout_rate),
                nn.Linear(self.classifier_hidden_dim, self.num_classes)])
        else:
            classifier_layers.extend([
                nn.Dropout(self.classifier_dropout_rate),
                nn.Linear(self.classifier_input_dim_final, self.num_classes)])
        self.classifier = nn.Sequential(*classifier_layers)
        model_logger.info(f"  分类器: 输入维度={self.classifier_input_dim_final}, 隐藏层维度={self.classifier_hidden_dim}, Dropout={self.classifier_dropout_rate}")
        self.apply(self._init_weights) # 应用权重初始化

    def _create_local_adj_matrix(self, num_nodes: int, k_hop: int = 1) -> torch.Tensor:
            """
            创建用于局部空间注意力的邻接矩阵。
            这个矩阵是布尔型的，并且最终会包含自环和基于k-hop的连接。
            """
            adj_1_hop_physical = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
            neighbor_link_for_logging = [] # 用于日志或调试，实际填充adj_1_hop_physical

            if num_nodes == 20: # UCLA 数据集
                model_logger.info(f"为 UCLA ({num_nodes}个节点) 创建1-hop邻接关系。")
                neighbor_link_for_logging = [(1,2),(2,3),(4,3),(5,3),(6,5),(7,6),(8,7),(9,3),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19)]
                for i, j in neighbor_link_for_logging:
                    adj_1_hop_physical[i-1, j-1] = True
                    adj_1_hop_physical[j-1, i-1] = True
            elif num_nodes == 25: # NTU 数据集
                model_logger.info(f"为 NTU ({num_nodes}个节点) 创建1-hop邻接关系。")
                neighbor_link_for_logging = [(1,2),(2,21),(3,21),(4,3),(5,21),(6,5),(7,6),(8,7),(9,21),(10,9),(11,10),(12,11),(13,1),(14,13),(15,14),(16,15),(17,1),(18,17),(19,18),(20,19),(22,23),(23,8),(24,25),(25,12)]
                for i, j in neighbor_link_for_logging:
                    adj_1_hop_physical[i-1, j-1] = True
                    adj_1_hop_physical[j-1, i-1] = True
            elif num_nodes == 22: # SHREC'17 数据集
                model_logger.info(f"为 SHREC'17 ({num_nodes}个节点) 创建1-hop邻接关系。")
                # 你提供的 SHREC'17 图结构 (1-based index)
                inward_ori_index = [
                    (1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9), (11, 2),
                    (12, 11), (13, 12), (14, 13), (15, 2), (16, 15), (17, 16), (18, 17), (19, 2),
                    (20, 19), (21, 20), (22, 21)
                    # 移除了 (2,2) 自环，因为 torch.eye() 会处理所有节点的自环
                ]
                for i, j in inward_ori_index:
                    # 确保节点索引在有效范围内 (1 到 num_nodes)
                    if 1 <= i <= num_nodes and 1 <= j <= num_nodes:
                        adj_1_hop_physical[i-1, j-1] = True # 转换为0-based索引
                        adj_1_hop_physical[j-1, i-1] = True # 无向图
                    else:
                        model_logger.warning(f"SHREC'17邻接定义中出现无效节点索引: ({i}, {j})，节点数: {num_nodes}")
            else:
                model_logger.warning(f"未知的节点数 ({num_nodes}) 用于预定义的1-hop邻接关系。将只使用自环进行k-hop扩展（如果k_hop>0）。")
                # adj_1_hop_physical 将保持为全零，这意味着如果没有自环，k-hop扩展也不会产生连接。
            # 始终包含自环 (对角线为True)
            # 注意：如果 adj_1_hop_physical 是在 GPU 上创建的，torch.eye 也应该在同一设备上
            # 但通常邻接矩阵在CPU上创建，然后由模型模块的 register_buffer 移动到GPU
            final_adj = torch.eye(num_nodes, dtype=torch.bool)
            if k_hop > 0:
                # current_reach 代表从当前 final_adj 出发，通过 adj_1_hop_physical 能到达的新节点
                # k_hop = 1 表示只使用直接连接 (adj_1_hop_physical) 和自环
                # k_hop = 2 表示使用1-hop的邻居的邻居，依此类推
                # 将1-hop物理连接合并到final_adj中 (已经包含自环)
                final_adj = final_adj | adj_1_hop_physical

                # 如果 k_hop > 1，才需要进一步扩展
                # current_adj_power_k 代表 A^k (布尔矩阵乘法)
                # A_k_plus_1 = A_k @ A_1
                current_adj_power_k = adj_1_hop_physical.clone() # A^1
                for _ in range(k_hop - 1): # 如果 k_hop=1, 此循环不执行
                    # 计算 (当前已达范围) @ (1-hop连接)
                    # (A | A^2 | ... | A^i) @ A_1hop  -> 扩展到 A^(i+1)hop
                    # 或者更直接：A^(k+1) = A^k @ A_1hop
                    next_reach = torch.matmul(current_adj_power_k.float(), adj_1_hop_physical.float()) > 0
                    final_adj = final_adj | next_reach # 合并新可达的节点
                    current_adj_power_k = next_reach # 更新当前已达的最高跳数连接
            
            # model_logger.info(f"  为局部注意力创建了基于 {k_hop}-hop (包含自环和直接连接) 的邻接矩阵 (节点数={num_nodes})。")
            return final_adj

    def _init_weights(self, module: nn.Module):
        """自定义权重初始化方法。"""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine: # 检查是否有可学习的仿射参数
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention): # 初始化多头注意力层的权重
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'weight') and module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)
            # 偏置通常初始化为0
            if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0.)
            if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
                nn.init.constant_(module.out_proj.bias, 0.)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状 (B, T_in, N, C_in)。
            mask (Optional[torch.Tensor]): 可选的掩码张量，形状 (B, T_in)，
                                           True表示有效帧，False表示padding帧。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - logits (torch.Tensor): 分类得分，形状 (B, num_classes)。
                - output_spatial_attns (Optional[torch.Tensor]): 空间注意力权重图（如果启用），
                                                              形状 (B, T_in, NumHeads, N, N)。
        """
        B, T_in, N, C_in = x.shape
        # model_logger.debug(f"输入形状: {x.shape}, Mask形状: {mask.shape if mask is not None else 'None'}")

        # 1. 输入嵌入
        x_flat = rearrange(x, 'b t n c -> (b t n) c')
        x_embedded_flat = self.input_embedding(x_flat)
        x_embedded = rearrange(x_embedded_flat, '(b t n) d -> b t n d', b=B, t=T_in, n=N)
        x_normed = self.input_ln(x_embedded)
        x_processed_input = self.input_dropout(x_normed) # (B, T, N, D_embed)

        # 2. 逐帧空间特征提取
        x_spatial_in = rearrange(x_processed_input, 'b t n d -> (b t) n d')
        x_spatial_features_frames, spatial_attn_weights_frames = self.spatial_feature_extractor(x_spatial_in)
        x_spatial_features_sequence = rearrange(x_spatial_features_frames, '(b t) n d -> b t n d', b=B, t=T_in)

        # 3. 空间池化 (例如，对节点维度N取平均)
        x_temporal_input = x_spatial_features_sequence.mean(dim=2) # (B, T, D_embed)

        # 4. 时间位置编码 (可选)
        if self.use_time_pos_enc and self.time_pos_encoder is not None:
            time_pe = self.time_pos_encoder(x_temporal_input) # 输入 (B, T, D_embed)
            x_temporal_input = x_temporal_input + time_pe

        # 5. 主要时间序列建模
        # Transformer期望的key_padding_mask: (B, T), True表示对应位置是padding (需要被mask掉)
        temporal_key_padding_mask = (mask == False) if mask is not None else None

        if mask is not None: # 在输入到时间模型前，将padding部分特征置零
            x_temporal_input = x_temporal_input * mask.unsqueeze(-1).float()

        if self.temporal_model_type == 'gru':
            if hasattr(self.temporal_encoder, 'flatten_parameters'): # 解决DataParallel的潜在问题
                self.temporal_encoder.flatten_parameters()
            processed_temporal_sequence, _ = self.temporal_encoder(x_temporal_input) # (B, T, D_temporal_final_out)
        elif self.temporal_model_type == 'transformer':
            processed_temporal_sequence = self.temporal_encoder(
                x_temporal_input, src_key_padding_mask=temporal_key_padding_mask
            ) # (B, T, D_temporal_final_out)
        else:
            # 这种情况不应该发生，因为__init__中已经检查过了
            model_logger.error(f"前向传播中遇到未知的temporal_model_type: {self.temporal_model_type}")
            processed_temporal_sequence = x_temporal_input


        # 6. 可选的后续附加时间处理模块
        # 注意：Provided_TA 相关的逻辑已被完全移除

        if self.use_multiscale_temporal_after_main and self.multiscale_temporal_processor:
            processed_temporal_sequence = self.multiscale_temporal_processor(processed_temporal_sequence, mask=mask)

        if self.use_temporal_attn_after_main_specific and self.temporal_attn_blocks_after_main:
            temp_out_after_main = processed_temporal_sequence
            if mask is not None: # 再次确保padding部分为0，以正确处理LayerNorm等
                 temp_out_after_main = temp_out_after_main * mask.unsqueeze(-1).float()
            for block in self.temporal_attn_blocks_after_main:
                temp_out_after_main = block(temp_out_after_main, key_padding_mask=temporal_key_padding_mask)
            processed_temporal_sequence = temp_out_after_main

        # 7. 最终序列表示聚合
        if mask is not None:
            # 对有效时间步进行平均池化
            masked_output_for_agg = processed_temporal_sequence * mask.unsqueeze(-1).float()
            summed_output = masked_output_for_agg.sum(dim=1) # 沿时间维度求和
            valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1), 避免除以零
            final_representation = summed_output / valid_lengths
        else:
            # 如果没有mask，则对所有时间步进行全局平均池化
            final_representation = processed_temporal_sequence.mean(dim=1)

        # 8. 分类头
        logits = self.classifier(final_representation)

        # 处理空间注意力图的返回 (如果需要)
        output_spatial_attns = None
        # 检查 self.spatial_feature_extractor 是否真的有 output_attention 属性并为True
        if hasattr(self.spatial_feature_extractor, 'output_attention') and \
           self.spatial_feature_extractor.output_attention and \
           spatial_attn_weights_frames is not None:
            try:
                output_spatial_attns = rearrange(spatial_attn_weights_frames, '(b t) h n1 n2 -> b t h n1 n2', b=B, t=T_in)
            except Exception as e:
                model_logger.warning(f"无法重塑空间注意力权重图: {e}")

        return logits, output_spatial_attns