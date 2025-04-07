# 文件名: spatial_attention.py (修改版)

# --- embedding.py 中的类需要能被导入 ---
try:
    # 假设 embedding.py 在同一目录下或 python path 中
    from .embedding import TokenEmbedding, PositionalEmbedding
except ImportError:
     # 或者如果它们在不同目录
     # from model.embedding import TokenEmbedding, PositionalEmbedding
    print("警告: 无法导入 TokenEmbedding 或 PositionalEmbedding。")
    # 临时占位符
    class TokenEmbedding(nn.Module): # ...
    class PositionalEmbedding(nn.Module): # ...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

# --- AttentionLayer, FeedForwardModule, Encoder 保持不变 ---
class AttentionLayer(nn.Module):
    # ... (代码不变) ...

class FeedForwardModule(nn.Module):
    # ... (代码不变) ...

class Encoder(nn.Module):
    # ... (代码不变) ...

# --- 修改 SpatialAttention ---
class SpatialAttention(nn.Module):
    def __init__(self,
                 in_channels, # 输入特征维度 (例如来自上层 GRU 的 num_rnn_units)
                 d_model,     # 模型内部维度 (通常等于 in_channels)
                 num_nodes,
                 n_heads,
                 ffn_dim,
                 st_layers,         # Transformer Encoder 层数
                 dropout_rate,      # 统一使用 dropout_rate
                 output_attention):
        super().__init__()

        # 参数确认: 原始代码用 st_dropout_rate, 但 Encoder 用 dropout。统一用 dropout_rate。
        # 参数确认: Encoder 没有用到 num_nodes 和 st_layers 参数？似乎是冗余的。
        # 参数确认: d_model 通常应等于 in_channels，因为注意力是在节点维度上做的。

        self.in_channels = in_channels
        self.d_model = d_model
        # self.num_nodes = num_nodes # Encoder 内部未使用
        # self.n_heads = n_heads     # Encoder 内部未使用
        # self.ffn_dim = ffn_dim     # Encoder 内部未使用
        self.st_layers = st_layers
        self.dropout_rate = dropout_rate
        self.output_attention = output_attention

        # --- 移除 TokenEmbedding ---
        # 输入 x 应该已经是 d_model 维度了 (由 SDT_GRU_Classifier.input_embedding 或上层 SpatialTemporalCell 处理)
        # self.val_embedding = TokenEmbedding(in_channels, d_model)

        # --- 保留 PositionalEmbedding ---
        # 注意：这里的 PositionalEmbedding 是针对 "节点" 维度的，而不是时间序列维度。
        # 这在手势识别中可能不太直观，因为节点（关节点）的顺序通常是固定的。
        # 如果关节点顺序有意义（例如按某种拓扑结构排列），可以保留。
        # 否则，可以考虑移除节点位置编码，或者使用可学习的节点嵌入。
        # 这里暂时保留，但效果存疑。max_len 应设为 num_nodes。
        self.use_node_pos_emb = True # 可以设为 False 禁用
        if self.use_node_pos_emb:
             self.pos_embedding = PositionalEmbedding(d_model, max_len=num_nodes)

        # --- Transformer Encoder 层 ---
        self.encoders = nn.ModuleList(
            [
                Encoder(d_model=d_model,
                        num_nodes=num_nodes, # Encoder 内部似乎没用
                        n_heads=n_heads,
                        ffn_dim=ffn_dim,
                        st_layers=st_layers, # Encoder 内部似乎没用
                        st_dropout_rate=dropout_rate, # 使用统一的 dropout_rate
                        output_attention=output_attention)
                for _ in range(st_layers) # 使用 st_layers 控制层数
            ]
        )

    # --- 修改 forward 方法 ---
    def forward(self, x):
        # x: 输入特征 (Batch, NumNodes, d_model)
        # 移除 cur_extras 参数

        B, N, D = x.shape
        assert D == self.d_model, f"Input feature dim {D} != d_model {self.d_model}"
        assert N == self.pos_embedding.pe.size(1) if self.use_node_pos_emb else True, "Num nodes mismatch"


        attn_list = []
        src = x # 输入已经是 d_model 维度

        # --- 添加节点位置编码 (可选) ---
        if self.use_node_pos_emb:
            # self.pos_embedding(x) 返回 (1, NumNodes, d_model)
            # 需要广播到 Batch 维度
            node_pe = self.pos_embedding(x) # (1, N, D)
            src = src + node_pe # 直接相加

        # --- 通过 Transformer Encoder 层 ---
        for encoder in self.encoders:
            src, attn = encoder(src) # Encoder 输入 (B, N, D)
            if self.output_attention and attn is not None:
                attn_list.append(attn)

        # --- 组合注意力图 ---
        final_attn = None
        if self.output_attention and attn_list:
            # attn_list 是长度为 st_layers 的列表, 每个元素是 (B, H, N, N) 或 None
            # 过滤掉 None，如果存在的话
            valid_attns = [a for a in attn_list if a is not None]
            if valid_attns:
                 # 堆叠所有层的注意力 (B, st_layers, H, N, N)
                 final_attn = torch.stack(valid_attns, dim=1)

        # 返回处理后的节点表示和注意力图
        return src, final_attn