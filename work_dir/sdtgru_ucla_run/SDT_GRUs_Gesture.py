# 文件名: model/SDT_GRUs_Gesture.py (或 SDT_GRUs_Gesture_Runnable_CN.py)
# 包含所有必要的模型组件定义，用于骨骼动作识别分类
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
import inspect # 用于动态导入检查
import yaml

print(f"PyTorch 版本: {torch.__version__}") # 打印 PyTorch 版本

# --- 辅助函数 ---
def zoneout(prev_h, next_h, rate, training=True):
    """应用 Zoneout。
       以概率 rate 随机将 next_h 的某些单元替换为 prev_h 的对应单元。
    """
    if training and rate > 0.0 and rate < 1.0: # 只有在训练且 rate 有效时应用
        mask = torch.bernoulli(torch.full_like(prev_h, rate)).bool()
        next_h = torch.where(mask, prev_h, next_h)
    elif not training and rate > 0.0 and rate < 1.0: # 推理时进行混合
        next_h = rate * prev_h + (1.0 - rate) * next_h
    # 如果 rate 为 0 或 1，或者不在训练模式，则不执行操作或按原样返回 next_h
    return next_h

# --- 位置编码 ---
class PositionalEmbedding(nn.Module):
    """标准的 Sinusoidal Positional Embedding"""
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: 输入张量，需要其第二维的大小 (SeqLen)
        # 返回: (1, SeqLen, d_model)
        return self.pe[:, :x.size(1)]

# --- 空间注意力组件 ---
class AttentionLayer(nn.Module):
    """多头自注意力层"""
    def __init__(self, d_model, n_heads, dropout, output_attention=False): # output_attention 默认 False
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads
        self.output_attention = output_attention

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, NumNodes, d_model)
        B, N, D = x.shape
        H = self.n_heads

        # 线性投影并调整形状: (B, N, D) -> (B, H, N, D_k/D_v)
        queries = self.query_projection(x).view(B, N, H, self.d_keys).transpose(1, 2)
        keys = self.key_projection(x).view(B, N, H, self.d_keys).transpose(1, 2)
        values = self.value_projection(x).view(B, N, H, self.d_values).transpose(1, 2)

        # 计算注意力分数
        scale = 1. / math.sqrt(self.d_keys)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale # (B, H, N, N)

        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # 计算加权值
        weighted_values = torch.matmul(attn_weights_dropped, values) # (B, H, N, D_v)

        # 合并头并输出投影
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(B, N, -1) # (B, N, D)
        output = self.out_projection(weighted_values)

        if self.output_attention:
            return output, attn_weights.detach() # 返回原始权重用于分析
        else:
            return output, None


class FeedForwardModule(nn.Module):
    """前馈网络模块 (Position-wise Feed-Forward)"""
    def __init__(self, d_model, ffn_dim, activation=F.relu, dropout=0.1): # 默认激活函数 ReLU
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.linear2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class EncoderLayer(nn.Module):
    """标准的 Transformer 编码器层，作用于节点维度"""
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.relu, output_attention=False):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, n_heads, dropout_rate, output_attention)
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate) # 用于 Add & Norm 之后的 dropout

    def forward(self, x):
        # x: (Batch, NumNodes, d_model)
        # 1. 自注意力 + Add & Norm
        attn_output, attn_weights = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output)) # Post-LN 或 Pre-LN? 这里是 Post-LN

        # 2. 前馈网络 + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x, attn_weights


class SpatialAttention(nn.Module):
    """空间注意力模块 (基于 Transformer Encoder)"""
    def __init__(self, d_model, num_nodes, n_heads, ffn_dim, st_layers, dropout_rate, output_attention):
        super().__init__()
        self.d_model = d_model
        self.st_layers = st_layers # 内部 EncoderLayer 的数量
        self.output_attention = output_attention

        # --- 移除内部嵌入层 ---

        # --- 可选的节点位置编码 (默认禁用) ---
        self.use_node_pos_emb = False
        if self.use_node_pos_emb:
             self.pos_embedding = PositionalEmbedding(d_model, max_len=num_nodes)

        # 创建多个 EncoderLayer
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim,
                         dropout_rate=dropout_rate, output_attention=output_attention)
            for _ in range(st_layers)
        ])
        # 最终的 LayerNorm (可选)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: 输入 (Batch, NumNodes, d_model)
        B, N, D = x.shape
        if D != self.d_model:
             raise ValueError(f"SpatialAttention 输入维度错误: 期望 {self.d_model}, 得到 {D}")

        # 可选：添加节点位置编码
        if self.use_node_pos_emb and hasattr(self, 'pos_embedding'):
            node_pe = self.pos_embedding(x) # (1, N, D)
            x = x + node_pe

        attn_list = []
        # 通过所有 EncoderLayer
        for encoder in self.encoders:
            x, attn = encoder(x)
            if self.output_attention and attn is not None:
                attn_list.append(attn)

        x = self.final_norm(x) # 应用最终归一化

        # 组合注意力图
        final_attn = None
        if self.output_attention and attn_list:
            valid_attns = [a for a in attn_list if a is not None]
            if valid_attns:
                 final_attn = torch.stack(valid_attns, dim=1) # (B, st_layers, H, N, N)

        return x, final_attn


# --- 时空单元 (SpatialTemporalCell) ---
class SpatialTemporalCell(nn.Module):
    """结合空间注意力和 GRU 更新的核心单元"""
    def __init__(self,
                 in_channels,      # 此单元的输入特征维度 (通常 == num_rnn_units)
                 out_channels,     # GRU 隐藏状态维度 (通常 == num_rnn_units)
                 num_nodes,
                 n_heads,
                 ffn_dim,
                 st_layers,        # SpatialAttention *内部* 的层数
                 st_dropout_rate,
                 output_attention):
        super().__init__()
        if in_channels != out_channels:
             # 可以放宽这个限制，但需要调整 GRU 投影的维度
             print(f"警告: SpatialTemporalCell 的 in_channels ({in_channels}) 与 out_channels ({out_channels}) 不同。确保 GRU 投影正确处理。")
             # raise ValueError("SpatialTemporalCell 当前假设 in_channels == out_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels # 也是 SpatialAttention 的 d_model
        self.num_nodes = num_nodes
        self.output_attention = output_attention

        # 空间注意力模块
        self.spatial_attn_i = SpatialAttention(d_model=out_channels, num_nodes=num_nodes, n_heads=n_heads,
                                               ffn_dim=ffn_dim, st_layers=st_layers, dropout_rate=st_dropout_rate,
                                               output_attention=output_attention)
        self.spatial_attn_h = SpatialAttention(d_model=out_channels, num_nodes=num_nodes, n_heads=n_heads,
                                               ffn_dim=ffn_dim, st_layers=st_layers, dropout_rate=st_dropout_rate,
                                               output_attention=output_attention)

        # GRU 投影层 (输入和隐藏状态都已经是 out_channels 维度)
        self.gru_projection_i = nn.Linear(out_channels, out_channels * 3)
        self.gru_projection_h = nn.Linear(out_channels, out_channels * 3)

        # GRU 偏置项
        self.bias_ir = Parameter(torch.Tensor(out_channels)); self.bias_ii = Parameter(torch.Tensor(out_channels)); self.bias_in = Parameter(torch.Tensor(out_channels))
        self.bias_hr = Parameter(torch.Tensor(out_channels)); self.bias_hi = Parameter(torch.Tensor(out_channels)); self.bias_hn = Parameter(torch.Tensor(out_channels))

        # 层归一化 (作用于隐藏状态维度)
        self.ln = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for name, weight in self.named_parameters():
            if weight.dim() > 1: init.xavier_uniform_(weight)
            elif "bias" in name: init.zeros_(weight)
        # 确保 LayerNorm 参数初始化 (通常是 1 和 0)
        if hasattr(self.ln, 'weight') and self.ln.weight is not None: init.ones_(self.ln.weight)
        if hasattr(self.ln, 'bias') and self.ln.bias is not None: init.zeros_(self.ln.bias)

    def forward(self, x, prev_hidden=None):
        # x: (Batch, NumNodes, InChannels)
        # prev_hidden: (Batch, NumNodes, OutChannels)
        B, N, C_in = x.shape
        # --- 如果允许 in_channels != out_channels，这里需要处理 ---
        # if C_in != self.in_channels: ...
        if C_in != self.out_channels: # 当前假设它们相等
            raise ValueError(f"SpatialTemporalCell 输入维度 ({C_in}) 与期望 ({self.out_channels}) 不符。")

        if prev_hidden is None:
            prev_hidden = torch.zeros(B, N, self.out_channels, dtype=x.dtype, device=x.device)
        elif prev_hidden.shape != (B, N, self.out_channels):
             raise ValueError(f"prev_hidden 形状错误: 期望 {(B, N, self.out_channels)}, 得到 {prev_hidden.shape}")


        # --- 空间注意力 ---
        input_sp, input_attn = self.spatial_attn_i(x)           # x 已经是 out_channels 维度
        hidden_sp, hidden_attn = self.spatial_attn_h(prev_hidden)

        # --- GRU 更新 ---
        input_r, input_i, input_n = self.gru_projection_i(input_sp).chunk(3, dim=-1)
        hidden_r, hidden_i, hidden_n = self.gru_projection_h(hidden_sp).chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_r + self.bias_ir + hidden_r + self.bias_hr)
        update_gate = torch.sigmoid(input_i + self.bias_ii + hidden_i + self.bias_hi)
        new_gate = torch.tanh(input_n + self.bias_in + reset_gate * (hidden_n + self.bias_hn))
        next_hidden = (1.0 - update_gate) * new_gate + update_gate * prev_hidden

        # --- 层归一化 ---
        next_hidden_normalized = self.ln(next_hidden)
        output = next_hidden_normalized # 输出是归一化后的状态

        # --- 组合注意力图 ---
        final_attn = None
        if self.output_attention:
             if input_attn is not None and hidden_attn is not None:
                 final_attn = torch.stack([input_attn, hidden_attn], dim=1) # (B, 2, st_layers, H, N, N)
             elif input_attn is not None: final_attn = input_attn.unsqueeze(1)
             elif hidden_attn is not None: final_attn = hidden_attn.unsqueeze(1)

        return output, next_hidden_normalized, final_attn


# --- 主要的分类器模型 ---
class SDT_GRU_Classifier(nn.Module):
    """ 时空 GRU 分类器，用于骨骼序列分类 """
    def __init__(self, model_cfg): # 直接接收配置字典
        super().__init__()
        print("初始化 SDT_GRU_Classifier，配置如下:")
        # 使用 .get 提供默认值，增加鲁棒性
        print(yaml.dump(model_cfg, default_flow_style=None))

        # --- 从 model_cfg 获取参数 ---
        self.num_input_dim = model_cfg['num_input_dim']
        self.num_nodes = model_cfg['num_nodes']
        self.num_classes = model_cfg['num_classes']
        self.num_rnn_layers = model_cfg['num_rnn_layers']
        self.num_rnn_units = model_cfg['num_rnn_units'] # d_model
        self.n_heads = model_cfg['n_heads']
        self.ffn_dim = model_cfg['ffn_dim']
        self.st_layers = model_cfg['st_layers'] # SpatialAttention 内部层数

        self.st_dropout_rate = model_cfg.get('st_dropout_rate', 0.1)
        self.rnn_dropout_rate = model_cfg.get('rnn_dropout_rate', 0.1)
        self.zoneout_rate = model_cfg.get('zoneout_rate', 0.0)
        self.classifier_dropout = model_cfg.get('classifier_dropout', 0.5)

        self.output_attention = model_cfg.get('output_attention', False)
        self.use_gap = model_cfg.get('use_gap', True)
        self.classifier_hidden_dim = model_cfg.get('classifier_hidden_dim', 0)

        # --- 输入嵌入层 ---
        self.input_embedding = nn.Linear(self.num_input_dim, self.num_rnn_units)
        self.input_ln = nn.LayerNorm(self.num_rnn_units)
        self.input_dropout = nn.Dropout(self.st_dropout_rate)

        # --- 核心编码器单元 ---
        self.encoder_cells = nn.ModuleList()
        for i in range(self.num_rnn_layers):
            # 所有层的输入输出维度都是 num_rnn_units
            self.encoder_cells.append(
                SpatialTemporalCell(
                    in_channels=self.num_rnn_units,
                    out_channels=self.num_rnn_units,
                    num_nodes=self.num_nodes,
                    n_heads=self.n_heads,
                    ffn_dim=self.ffn_dim,
                    st_layers=self.st_layers,
                    st_dropout_rate=self.st_dropout_rate,
                    output_attention=self.output_attention
                )
            )

        # --- 分类头 ---
        self.final_dropout = nn.Dropout(self.classifier_dropout)
        in_features_classifier = self.num_rnn_units * (1 if self.use_gap else self.num_nodes)

        classifier_layers = []
        if self.classifier_hidden_dim > 0:
             classifier_layers.append(nn.Linear(in_features_classifier, self.classifier_hidden_dim))
             classifier_layers.append(nn.ReLU())
             classifier_layers.append(nn.Dropout(self.classifier_dropout)) # 可以再加一层 Dropout
             classifier_layers.append(nn.Linear(self.classifier_hidden_dim, self.num_classes))
        else:
             classifier_layers.append(nn.Linear(in_features_classifier, self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self.global_step = 0 # 可用于外部追踪

    def forward(self, x, mask=None):
        """
        x: (Batch, SeqLen, NumNodes, InputDim)
        mask: (Optional) (Batch, SeqLen), True 表示有效帧
        """
        B, T, N, C = x.shape
        assert N == self.num_nodes, f"输入节点数 {N} != 配置 {self.num_nodes}"
        assert C == self.num_input_dim, f"输入维度 {C} != 配置 {self.num_input_dim}"

        # 1. 初始嵌入
        x_flat = x.reshape(B * T * N, C)
        x_embedded_flat = self.input_embedding(x_flat)
        x_embedded = x_embedded_flat.reshape(B, T, N, self.num_rnn_units)
        x_processed = self.input_ln(x_embedded)
        x_processed = self.input_dropout(x_processed)
        x_processed = x_processed.permute(1, 0, 2, 3) # (T, B, N, D)

        # 2. 通过 RNN 层
        hidden_states = [None] * self.num_rnn_layers
        all_outputs = []
        all_attentions_list = [] # 存储每个时间步的注意力列表

        current_input_seq = x_processed
        for t in range(T):
            input_for_t = current_input_seq[t] # (B, N, D)
            next_layer_input = input_for_t
            attns_t = [] # 当前时间步所有层的注意力 (每层可能有多个)

            for i, rnn_cell in enumerate(self.encoder_cells):
                prev_hidden = hidden_states[i]
                output, next_hidden, attn = rnn_cell(x=next_layer_input, prev_hidden=prev_hidden)
                if self.zoneout_rate > 0.0 and prev_hidden is not None:
                    next_hidden = zoneout(prev_hidden, next_hidden, self.zoneout_rate, self.training)
                hidden_states[i] = next_hidden
                next_layer_input = output
                if i < self.num_rnn_layers - 1 and self.rnn_dropout_rate > 0.0:
                    next_layer_input = F.dropout(next_layer_input, p=self.rnn_dropout_rate, training=self.training)
                if self.output_attention and attn is not None:
                    attns_t.append(attn) # attn shape (B, 2, st_layers, H, N, N) or None

            all_outputs.append(next_layer_input) # 存储最后一层输出
            if self.output_attention and attns_t:
                all_attentions_list.append(attns_t) # 添加当前时间步的注意力列表

        # 3. 选择最终表示
        stacked_outputs = torch.stack(all_outputs, dim=0) # (T, B, N, D)
        if mask is not None:
            valid_lengths = mask.sum(dim=1)
            last_valid_idx = (valid_lengths - 1).long().clamp(min=0)
            batch_indices = torch.arange(B, device=x.device)
            final_encoder_output = stacked_outputs[last_valid_idx, batch_indices, :, :] # (B, N, D)
        else:
            final_encoder_output = stacked_outputs[-1] # (B, N, D)

        # 4. 分类头
        if self.use_gap: final_representation = final_encoder_output.mean(dim=1) # (B, D)
        else: final_representation = final_encoder_output.reshape(B, -1) # (B, N*D)
        final_representation = self.final_dropout(final_representation)
        logits = self.classifier(final_representation) # (B, num_classes)

        # 5. 组合注意力图 (如果需要)
        final_attns = None
        if self.output_attention and all_attentions_list:
             # 这是一个复杂的结构 (T, List[Tensor(B, 2, st_layers, H, N, N)])
             # 如何聚合需要具体设计，这里暂时不聚合
             # 可以考虑只返回最后一个时间步的:
             # last_t_attns = all_attentions_list[-1] # List of attentions for last step
             # if last_t_attns: final_attns = torch.stack(last_t_attns, dim=1) # (B, num_rnn_layers, 2, st_layers, H, N, N)
             pass # 保持为 None

        if self.training: self.global_step += 1
        return logits, final_attns


# --- 示例用法 (用于测试代码结构) ---
if __name__ == "__main__":
    print("测试 SDT_GRU_Classifier 结构...")
    # --- 示例配置 (适配 NW-UCLA) ---
    cfg_example = {
        'num_input_dim': 3, 'num_nodes': 20, 'num_classes': 10,
        'num_rnn_layers': 2, 'num_rnn_units': 128, 'n_heads': 8,
        'ffn_dim': 256, 'st_layers': 2, 'st_dropout_rate': 0.1,
        'rnn_dropout_rate': 0.1, 'classifier_dropout': 0.5, 'zoneout_rate': 0.0,
        'output_attention': False, 'use_gap': True, 'classifier_hidden_dim': 64
    }
    model = SDT_GRU_Classifier(cfg_example)
    print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {param_count:,}")

    # --- 虚拟输入 ---
    B, T, N, C = 4, 52, 20, 3
    dummy_input = torch.randn(B, T, N, C)
    dummy_mask = torch.ones(B, T, dtype=torch.bool)
    dummy_mask[0, -5:] = False # 示例 padding
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"掩码形状: {dummy_mask.shape}")

    # --- 前向传播测试 ---
    model.train()
    logits, attns = model(dummy_input, mask=dummy_mask) # 测试带掩码
    print(f"输出 logits 形状 (train): {logits.shape}") # 期望: (B, num_classes)
    model.eval()
    with torch.no_grad(): logits_eval, _ = model(dummy_input, mask=None) # 测试不带掩码
    print(f"输出 logits 形状 (eval): {logits_eval.shape}") # 期望: (B, num_classes)