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
import numpy as np
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


# 尝试导入 timm 的 DropPath 和 trunc_normal_，如果失败则使用内置替代
try:
    from timm.layers import trunc_normal_, DropPath, Mlp
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
    新增：支持动态邻接偏置 (类似Q的概念)。
    """
    def __init__(self, d_model, n_heads, dropout, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20,
                 use_global_spatial_bias=False, attention_type='global', adj_matrix_for_local=None,
                 # 新增参数用于动态邻接偏置 (Q-like bias)
                 use_dynamic_adj_bias: bool = False,
                 d_q_intermediate: int = None, # 动态偏置计算的中间维度
                 q_dropout_rate: float = 0.0): # 动态偏置计算中的dropout
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
            self.resid_norm_q = nn.LayerNorm(d_model)
            self.resid_norm_k = nn.LayerNorm(d_model)
            self.resid_norm_v = nn.LayerNorm(d_model)
        else: # 使用线性层进行QKV投影
            self.query_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.key_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.value_projection = nn.Linear(d_model, d_model, bias=qkv_bias)

        # 相对位置偏置
        if num_nodes > 0:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * num_nodes - 1), n_heads))
            trunc_normal_(self.relative_position_bias_table, std=.02)
            coords_n = torch.arange(num_nodes)
            relative_coords_n = coords_n[:, None] - coords_n[None, :] + num_nodes - 1
            self.register_buffer("relative_position_index", relative_coords_n.long(), persistent=False)
        else:
            self.relative_position_bias_table = None
            self.register_buffer("relative_position_index", None, persistent=False)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 全局可学习的空间偏置
        if self.use_global_spatial_bias and self.attention_type == 'global' and num_nodes > 0:
            self.global_spatial_bias = nn.Parameter(torch.zeros(n_heads, num_nodes, num_nodes))
            self.alpha_global_bias = nn.Parameter(torch.tensor(1.0))
            trunc_normal_(self.global_spatial_bias, std=.02)
        else:
            self.global_spatial_bias = None
            self.alpha_global_bias = None

        # 局部连接偏置
        if self.attention_type == 'local' and num_nodes > 0: # 即使是局部注意力，也可能有一个静态的、可学习的偏置
            self.local_connection_bias = nn.Parameter(torch.zeros(self.n_heads, self.num_nodes, self.num_nodes))
            # trunc_normal_(self.local_connection_bias, std=.02) # 可以选择初始化方式
        else:
            self.local_connection_bias = None

        # --- 新增：动态邻接偏置 (Q-like bias) 的初始化 ---
        self.use_dynamic_adj_bias = use_dynamic_adj_bias
        if self.use_dynamic_adj_bias:
            self.d_q_intermediate = d_q_intermediate
            if self.d_q_intermediate is None or self.d_q_intermediate <= 0:
                self.d_q_intermediate = max(16, d_model // 4) # 默认中间维度
                # model_logger.info(f"AttentionLayer: d_q_intermediate for dynamic bias not provided or invalid, set to {self.d_q_intermediate}")
            
            self.q_feat_projection = nn.Linear(d_model, self.d_q_intermediate)
            self.q_pairwise_bias_projection = nn.Linear(self.d_q_intermediate, self.n_heads) # 输出每个头的偏置
            self.q_activation = nn.Tanh() # 激活函数，使得偏置值有正有负
            if q_dropout_rate > 0:
                self.q_dropout = nn.Dropout(q_dropout_rate)
            # model_logger.info(f"AttentionLayer: Dynamic Adjacency Bias enabled (intermediate_dim={self.d_q_intermediate}, dropout={q_dropout_rate}).")
        # --- 结束新增 ---

    def _get_relative_positional_bias(self) -> torch.Tensor:
        if self.relative_position_bias_table is None or self.relative_position_index is None:
            return 0.0
        idx = self.relative_position_index.view(-1).to(self.relative_position_bias_table.device)
        relative_position_bias = self.relative_position_bias_table[idx]
        relative_position_bias = relative_position_bias.view(self.num_nodes, self.num_nodes, self.n_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x): # x 输入形状: (B, N, D)
        B, N, D = x.shape; H = self.n_heads
        if N != self.num_nodes:
            raise ValueError(f"输入节点维度 {N} 与 AttentionLayer 初始化的 num_nodes ({self.num_nodes}) 不符")

        if self.use_conv_proj:
            x_permuted = x.permute(0, 2, 1).contiguous()
            q_conv = self.query_projection(x_permuted)
            k_conv = self.key_projection(x_permuted)
            v_conv = self.value_projection(x_permuted)
            q_normed = self.resid_norm_q(q_conv.permute(0, 2, 1).contiguous())
            k_normed = self.resid_norm_k(k_conv.permute(0, 2, 1).contiguous())
            v_normed = self.resid_norm_v(v_conv.permute(0, 2, 1).contiguous())
            queries_proj = x + q_normed
            keys_proj = x + k_normed
            values_proj = x + v_normed
        else:
             queries_proj = self.query_projection(x)
             keys_proj = self.key_projection(x)
             values_proj = self.value_projection(x)

        queries = queries_proj.view(B, N, H, self.d_keys).permute(0, 2, 1, 3)
        keys = keys_proj.view(B, N, H, self.d_keys).permute(0, 2, 1, 3)
        values = values_proj.view(B, N, H, self.d_values).permute(0, 2, 1, 3)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_keys)

        if self.relative_position_bias_table is not None:
            scores = scores + self._get_relative_positional_bias()

        if self.attention_type == 'global' and self.global_spatial_bias is not None:
            scores = scores + self.global_spatial_bias.unsqueeze(0) * self.alpha_global_bias

        if self.attention_type == 'local' and self.local_connection_bias is not None:
            scores = scores + self.local_connection_bias.unsqueeze(0) # 添加静态的局部连接偏置

        # --- 新增：计算并添加动态邻接偏置 ---
        if self.use_dynamic_adj_bias:
            # x 的形状是 (B, N, D)
            q_node_features = self.q_feat_projection(x)  # (B, N, d_q_intermediate)

            # 计算成对特征差异
            q_feat_i = q_node_features.unsqueeze(2)  # (B, N, 1, d_q_intermediate)
            q_feat_j = q_node_features.unsqueeze(1)  # (B, 1, N, d_q_intermediate)
            # 广播相减得到 (B, N, N, d_q_intermediate)
            pairwise_q_relations = q_feat_i - q_feat_j 
            
            pairwise_q_relations_activated = self.q_activation(pairwise_q_relations)

            if hasattr(self, 'q_dropout'): # 检查 q_dropout 是否已定义 (基于 q_dropout_rate > 0)
                pairwise_q_relations_activated = self.q_dropout(pairwise_q_relations_activated)

            # 将成对关系投影到每个头的偏置
            # 输入 (B, N, N, d_q_intermediate), 输出 (B, N, N, n_heads)
            dynamic_adj_bias_unpermuted = self.q_pairwise_bias_projection(pairwise_q_relations_activated)
            
            # 转换维度以匹配 scores (B, n_heads, N, N)
            dynamic_adj_bias = dynamic_adj_bias_unpermuted.permute(0, 3, 1, 2) # (B, n_heads, N, N)
            
            scores = scores + dynamic_adj_bias # 将学习到的动态偏置加到注意力分数上
        # --- 结束新增 ---

        if self.attention_type == 'local' and self.adj_matrix_for_local is not None:
            local_mask = self.adj_matrix_for_local.logical_not().unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(local_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        weighted_values = torch.matmul(attn_weights_dropped, values)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        output = self.out_projection(weighted_values)

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

        # --- 0. 图的初始化 ---
        graph_class_str = model_cfg.get('graph_class') # 例如: "graph.ucla.Graph"
        graph_args_cfg = model_cfg.get('graph_args', {})   # 例如: {"labeling_mode": "spatial"}

        if not graph_class_str:
            raise ValueError("模型配置 'model_cfg' 中必须包含 'graph_class' (图类的导入路径)。")

        try:
            GraphClass = import_class(graph_class_str) # 使用您指定的 import_class
            model_logger.info(f"成功导入图类: {graph_class_str}")
        except ImportError as e:
            model_logger.error(f"无法导入图类 '{graph_class_str}': {e}")
            raise
        except Exception as e:
            model_logger.error(f"导入或获取图类 '{graph_class_str}' 时发生未知错误: {e}")
            raise

        self.graph_instance = GraphClass(**graph_args_cfg)
        model_logger.info(f"图实例 '{graph_class_str}' 创建成功，参数: {graph_args_cfg}")

        # 从图实例中获取邻接矩阵 A。
        # 假设 self.graph_instance.A 返回的是一个 (K, N, N) 或 (N, N) 的 NumPy 数组或 PyTorch 张量。
        raw_A_from_graph = self.graph_instance.A
        if not isinstance(raw_A_from_graph, (np.ndarray, torch.Tensor)):
            raise TypeError(f"图实例返回的 A 必须是 NumPy 数组或 PyTorch 张量，得到 {type(raw_A_from_graph)}")
        # 将 NumPy 数组转换为 PyTorch 张量
        if isinstance(raw_A_from_graph, np.ndarray):
            raw_A_from_graph_tensor = torch.from_numpy(raw_A_from_graph).float()
        else:
            raw_A_from_graph_tensor = raw_A_from_graph.float()
        if raw_A_from_graph_tensor.ndim == 3:
            if raw_A_from_graph_tensor.shape[0] >= 3: # 至少包含 I, In, Out
                # 假设索引1是In, 索引2是Out。这些索引需要与您的 graph.tools.get_spatial_graph 对应
                adj_in = raw_A_from_graph_tensor[1]
                adj_out = raw_A_from_graph_tensor[2]
                # 合并内向和外向连接作为基础物理连接，并确保是布尔型
                # (A > 0) 将非零值转为True，零值转为False
                physical_adj_tensor_1_hop = (adj_in + adj_out) > 0
                model_logger.info(f"  从 (K,N,N) 图邻接矩阵中提取1-hop物理连接 (合并策略索引1和2)。")
            elif raw_A_from_graph_tensor.shape[0] == 1: # 只有一个策略，假设它就是物理连接
                physical_adj_tensor_1_hop = raw_A_from_graph_tensor[0] > 0
                model_logger.info(f"  使用 (1,N,N) 图邻接矩阵的第一个策略作为1-hop物理连接。")
            else:
                raise ValueError(f"图邻接矩阵 A 的形状 {raw_A_from_graph_tensor.shape} 不支持自动提取1-hop物理连接。")
        elif raw_A_from_graph_tensor.ndim == 2: # (N, N)
            physical_adj_tensor_1_hop = raw_A_from_graph_tensor > 0
            model_logger.info(f"  使用 (N,N) 图邻接矩阵作为1-hop物理连接。")
        else:
            raise ValueError(f"图邻接矩阵 A 的维度 ({raw_A_from_graph_tensor.ndim}) 不支持。期望2或3维。")


        # --- 1. 基础参数 ---
        self.num_input_dim: int = model_cfg['num_input_dim']
        self.num_nodes: int = physical_adj_tensor_1_hop.shape[0] # 从图推断节点数
        if 'num_nodes' in model_cfg and model_cfg['num_nodes'] != self.num_nodes:
            model_logger.warning(f"模型配置中的 num_nodes ({model_cfg['num_nodes']}) 与从图推断的节点数 ({self.num_nodes}) 不符。将使用图推断的值。")
        self.num_classes: int = model_cfg['num_classes']
        self.max_seq_len: int = model_cfg.get('max_seq_len', 100)

        # --- 2. 输入嵌入层 ---
        self.embedding_dim: int = model_cfg.get('embedding_dim', 128) # 空间处理和时间模型交互的特征维度
        self.input_embedding = nn.Linear(self.num_input_dim, self.embedding_dim)
        self.input_ln = nn.LayerNorm(self.embedding_dim)
        self.input_dropout = nn.Dropout(model_cfg.get('input_dropout_rate', 0.1))

        # --- 3. 逐帧空间特征提取器 (使用 SpatialAttention) ---
        k_hop_for_spatial_attn = model_cfg.get('local_adj_k_hop', 1)
        # physical_adj_tensor_1_hop 是从Graph类获取的，代表纯粹的1-hop物理连接（可能不含自环）
        adj_for_spatial_attention = self._post_process_adj_for_attention(
            physical_adj_tensor_1_hop, # 这是(N,N)的布尔张量
            k_hop_for_spatial_attn
        )
        # adj_for_spatial_attention 现在是一个 (N,N) 的布尔张量，包含自环和k-hop扩展

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
            'adj_matrix_for_local': adj_for_spatial_attention
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

    def _post_process_adj_for_attention(self, base_adj_physical_1_hop: torch.Tensor, k_hop: int) -> torch.Tensor:
        """
        对基础的1-hop物理邻接矩阵进行后处理，添加自环并进行k-hop扩展。
        Args:
            base_adj_physical_1_hop (torch.Tensor): (N,N) 布尔张量，代表1-hop物理连接。
                                                它应该不包含自环，或者即使包含，也会被这里的自环逻辑覆盖。
            k_hop (int): k-hop扩展的跳数。
        Returns:
            torch.Tensor: (N,N) 布尔张量，处理后的最终邻接矩阵，用于空间注意力。
        """
        num_nodes = base_adj_physical_1_hop.shape[0]
        device = base_adj_physical_1_hop.device # 保持设备一致性

        # 确保 base_adj_physical_1_hop 是对称的 (如果是从 In+Out 合并而来，通常已经是)
        # 并且确保它不包含自环，因为我们会在下一步显式添加
        adj_no_self_loop = base_adj_physical_1_hop.clone()
        if num_nodes > 0: # 只有当有节点时才操作对角线
            adj_no_self_loop.fill_diagonal_(False) # 移除任何可能存在的自环
        adj_no_self_loop = adj_no_self_loop | adj_no_self_loop.t() # 强制对称

        # 1. 初始化 final_adj：如果 k_hop=0，则只有自环；否则，从自环和1-hop开始
        if k_hop == 0:
            final_adj = torch.eye(num_nodes, dtype=torch.bool, device=device)
        else: # k_hop >= 1
            final_adj = torch.eye(num_nodes, dtype=torch.bool, device=device) # 自环
            final_adj = final_adj | adj_no_self_loop # 合并1-hop物理连接

        # 2. 进行 k-hop 扩展 (仅当 k_hop > 1 时)
        #    扩展基于不含自环的1-hop物理连接 (adj_no_self_loop)
        if k_hop > 1:
            # current_power_of_physical_adj 代表 (adj_no_self_loop)^i
            current_power_of_physical_adj = adj_no_self_loop.clone() # 这是物理连接的1次方 (无自环)

            for _hop_level in range(1, k_hop): # k_hop=2时循环1次, _hop_level=1, 计算物理连接的2次方
                                               # k_hop=3时循环2次, _hop_level=1,2, 计算物理连接的2、3次方
                # 计算 (物理连接的i次方) @ (物理连接的1次方) = 物理连接的(i+1)次方
                # 这里的乘法基于不含自环的邻接矩阵，以正确模拟多跳路径
                next_power_reach = torch.matmul(
                    current_power_of_physical_adj.float(),
                    adj_no_self_loop.float() # 始终乘以1-hop物理连接 (无自环)
                ) > 0
                final_adj = final_adj | next_power_reach    # 将新到达的连接合并到最终结果中
                current_power_of_physical_adj = next_power_reach # 更新当前达到的最高次幂
        
        # model_logger.info(f"  为局部注意力创建了基于 {k_hop}-hop 的邻接矩阵 (节点数={num_nodes})。")
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

        -- [新增] 适配新Feeder的维度重排逻辑 --
        新Feeder返回的数据形状是 (B, C, T, V, M)，我们需要将其转换为模型内部期望的 (B, T, V, C)。
        """
        # x 的输入形状: (B, C, T, V, M) - B:批次大小, C:坐标数(3), T:时间帧, V:关节数, M:人数
        # 假设 M=1 的情况，我们先移除最后一个维度。
        if x.dim() == 5:
            # .squeeze(-1) 会移除最后一个维度，如果它是1的话
            x = x.squeeze(-1)  # -> 形状变为 (B, C, T, V)
        
        # 维度重排: (B, C, T, V) -> (B, T, V, C)
        # B: 批次, T: 时间, V: 节点, C: 通道(坐标)
        x = x.permute(0, 2, 3, 1).contiguous()
        # 现在 x 的形状是 (B, T, V, C)，这正是模型后续部分所期望的输入格式。

        # --- 从这里开始，后续的 forward 逻辑与您原始的代码完全相同，无需任何修改 ---

        B, T_in, N, C_in = x.shape
        # model_logger.debug(f"输入形状 (调整后): {x.shape}, Mask形状: {mask.shape if mask is not None else 'None'}")

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
        temporal_key_padding_mask = (mask == False) if mask is not None else None

        if mask is not None:
            x_temporal_input = x_temporal_input * mask.unsqueeze(-1).float()

        if self.temporal_model_type == 'gru':
            if hasattr(self.temporal_encoder, 'flatten_parameters'):
                self.temporal_encoder.flatten_parameters()
            processed_temporal_sequence, _ = self.temporal_encoder(x_temporal_input)
        elif self.temporal_model_type == 'transformer':
            processed_temporal_sequence = self.temporal_encoder(
                x_temporal_input, src_key_padding_mask=temporal_key_padding_mask
            )
        else:
            model_logger.error(f"前向传播中遇到未知的temporal_model_type: {self.temporal_model_type}")
            processed_temporal_sequence = x_temporal_input

        # 6. 可选的后续附加时间处理模块
        if self.use_multiscale_temporal_after_main and self.multiscale_temporal_processor:
            processed_temporal_sequence = self.multiscale_temporal_processor(processed_temporal_sequence, mask=mask)

        if self.use_temporal_attn_after_main_specific and self.temporal_attn_blocks_after_main:
            temp_out_after_main = processed_temporal_sequence
            if mask is not None:
                 temp_out_after_main = temp_out_after_main * mask.unsqueeze(-1).float()
            for block in self.temporal_attn_blocks_after_main:
                temp_out_after_main = block(temp_out_after_main, key_padding_mask=temporal_key_padding_mask)
            processed_temporal_sequence = temp_out_after_main

        # 7. 最终序列表示聚合
        if mask is not None:
            masked_output_for_agg = processed_temporal_sequence * mask.unsqueeze(-1).float()
            summed_output = masked_output_for_agg.sum(dim=1)
            valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            final_representation = summed_output / valid_lengths
        else:
            final_representation = processed_temporal_sequence.mean(dim=1)

        # 8. 分类头
        logits = self.classifier(final_representation)

        # 处理空间注意力图的返回
        output_spatial_attns = None
        if hasattr(self.spatial_feature_extractor, 'output_attention') and \
           self.spatial_feature_extractor.output_attention and \
           spatial_attn_weights_frames is not None:
            try:
                output_spatial_attns = rearrange(spatial_attn_weights_frames, '(b t) h n1 n2 -> b t h n1 n2', b=B, t=T_in)
            except Exception as e:
                model_logger.warning(f"无法重塑空间注意力权重图: {e}")

        return logits, output_spatial_attns