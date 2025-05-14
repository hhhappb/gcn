# -*- coding: utf-8 -*-
# 文件名: model/SDT_GRUs_Gesture.py (v12.5 - Fully Implemented Global Spatial Bias and Zoneout, EncoderLayer uses nn.Dropout)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
import inspect # 保留 inspect，尽管在此版本中可能未直接使用，但 SDT_GRU_Classifier 的 __init__ 中有打印 model_cfg 的逻辑可能间接用到
import yaml
import logging

# 尝试导入 timm 的 DropPath 和 trunc_normal_，如果失败则使用内置替代
try:
    from timm.models.layers import trunc_normal_, DropPath
except ImportError:
    print("无法导入 timm.models.layers 中的 DropPath 或 trunc_normal_。")
    print("请确保已安装 timm 库: pip install timm")
    # 提供一个简单的 DropPath 替代实现（如果 timm 未安装）
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output
    # 提供一个简单的 trunc_normal_ 替代实现
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(min=a, max=b)
    print("警告: 无法从 timm 导入 DropPath/trunc_normal_，将使用内置简化版本（如果代码中仍有DropPath的调用）。")


logger = logging.getLogger(__name__)

print(f"PyTorch 版本: {torch.__version__}")

# --- 位置编码 (PositionalEmbedding) ---
class PositionalEmbedding(nn.Module):
    """标准的 Sinusoidal Positional Embedding (用于时间维度)"""
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term_exponent = torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        div_term = torch.exp(div_term_exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
             pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        time_dim_idx = 1
        return self.pe[:, :x.size(time_dim_idx)]

# --- 空间注意力层 (AttentionLayer - 实现全局空间偏置) ---
class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20,
                 use_global_spatial_bias=False): # <<<--- 接收全局偏置参数
        super().__init__()
        if d_model % n_heads != 0: raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model; self.n_heads = n_heads; self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads; self.output_attention = output_attention
        self.use_conv_proj = use_conv_proj; self.conv_kernel_size = conv_kernel_size
        self.num_nodes = num_nodes
        self.use_global_spatial_bias = use_global_spatial_bias # 保存参数

        if use_conv_proj:
            padding = conv_kernel_size // 2
            self.query_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.key_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.value_projection = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=padding, bias=qkv_bias)
            self.resid_norm_q = nn.LayerNorm(d_model); self.resid_norm_k = nn.LayerNorm(d_model); self.resid_norm_v = nn.LayerNorm(d_model)
        else:
            self.query_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.key_projection = nn.Linear(d_model, d_model, bias=qkv_bias)
            self.value_projection = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * num_nodes - 1), n_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)
        coords_n = torch.arange(num_nodes); relative_coords_n = coords_n[:, None] - coords_n[None, :]
        relative_coords_n += num_nodes - 1
        self.register_buffer("relative_position_index", relative_coords_n, persistent=False)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if self.use_global_spatial_bias:
            self.global_spatial_bias = nn.Parameter(torch.zeros(n_heads, num_nodes, num_nodes))
            self.alpha_global_bias = nn.Parameter(torch.tensor(1.0)) # 初始化为1，也可以尝试0或可学习
            # 可选：对全局偏置进行初始化，例如
            # trunc_normal_(self.global_spatial_bias, std=.02)

    def _get_relative_positional_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_nodes, self.num_nodes, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x):
        B, N, D = x.shape; H = self.n_heads
        if N != self.num_nodes: raise ValueError(f"输入节点维度 {N} != 预期的 {self.num_nodes}")
        if self.use_conv_proj:
            x_permuted = x.permute(0, 2, 1).contiguous(); q_conv = self.query_projection(x_permuted); k_conv = self.key_projection(x_permuted); v_conv = self.value_projection(x_permuted); q_conv_norm = self.resid_norm_q(q_conv.permute(0, 2, 1)).permute(0, 2, 1); k_conv_norm = self.resid_norm_k(k_conv.permute(0, 2, 1)).permute(0, 2, 1); v_conv_norm = self.resid_norm_v(v_conv.permute(0, 2, 1)).permute(0, 2, 1); queries_proj = x + q_conv_norm.permute(0, 2, 1).contiguous(); keys_proj = x + k_conv_norm.permute(0, 2, 1).contiguous(); values_proj = x + v_conv_norm.permute(0, 2, 1).contiguous()
        else:
             queries_proj = self.query_projection(x); keys_proj = self.key_projection(x); values_proj = self.value_projection(x)
        queries = queries_proj.view(B, N, H, self.d_keys).transpose(1, 2); keys = keys_proj.view(B, N, H, self.d_keys).transpose(1, 2); values = values_proj.view(B, N, H, self.d_values).transpose(1, 2)
        scale = 1. / math.sqrt(self.d_keys); scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        relative_position_bias = self._get_relative_positional_bias(); scores = scores + relative_position_bias
        if self.use_global_spatial_bias: # <<<--- 应用全局空间偏置
            scores = scores + self.global_spatial_bias.unsqueeze(0) * self.alpha_global_bias
        attn_weights = torch.softmax(scores, dim=-1); attn_weights_dropped = self.dropout(attn_weights); weighted_values = torch.matmul(attn_weights_dropped, values); weighted_values = weighted_values.transpose(1, 2).contiguous().view(B, N, -1); output = self.out_projection(weighted_values)
        return output, attn_weights.detach() if self.output_attention else None

# --- 前馈网络模块 (FeedForwardModule) ---
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, ffn_dim, activation=F.gelu, dropout=0.1):
        super().__init__(); self.linear1 = nn.Linear(d_model, ffn_dim); self.dropout = nn.Dropout(dropout); self.activation = activation; self.linear2 = nn.Linear(ffn_dim, d_model)
    def forward(self, x): x = self.linear1(x); x = self.activation(x); x = self.dropout(x); x = self.linear2(x); return x

# --- 空间注意力编码器层 (EncoderLayer - 使用 nn.Dropout, 传递全局偏置参数) ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.gelu, output_attention=False,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3, num_nodes=20,
                 use_global_spatial_bias=False): # <<<--- 接收全局偏置参数
        super().__init__()
        self.self_attn = AttentionLayer(d_model, n_heads, dropout_rate, output_attention,
                                      qkv_bias=qkv_bias, use_conv_proj=use_conv_proj,
                                      conv_kernel_size=conv_kernel_size, num_nodes=num_nodes,
                                      use_global_spatial_bias=use_global_spatial_bias) # 传递给 AttentionLayer
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout_residual1 = nn.Dropout(dropout_rate)
        self.dropout_residual2 = nn.Dropout(dropout_rate)
    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x); x_residual_after_attn = x + self.dropout_residual1(attn_output); x_norm1 = self.norm1(x_residual_after_attn); ffn_output = self.ffn(x_norm1); x_residual_after_ffn = x_norm1 + self.dropout_residual2(ffn_output); x_norm2 = self.norm2(x_residual_after_ffn)
        return x_norm2, attn_weights

# --- 空间注意力模块 (SpatialAttention - 传递全局偏置参数) ---
class SpatialAttention(nn.Module):
    def __init__(self, d_model, num_nodes, n_heads, ffn_dim, st_layers, dropout_rate, output_attention,
                 qkv_bias=False, use_conv_proj=True, conv_kernel_size=3,
                 use_global_spatial_bias=False): # <<<--- 接收全局偏置参数
        super().__init__()
        self.d_model = d_model; self.st_layers = st_layers; self.output_attention = output_attention; self.use_node_pos_emb = False
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim,
                         dropout_rate=dropout_rate, output_attention=output_attention,
                         qkv_bias=qkv_bias, use_conv_proj=use_conv_proj,
                         conv_kernel_size=conv_kernel_size, num_nodes=num_nodes,
                         use_global_spatial_bias=use_global_spatial_bias) # 传递给 EncoderLayer
            for _ in range(st_layers)])
    def forward(self, x):
        attn_list = [];
        for encoder in self.encoders: x, attn = encoder(x);
        if self.output_attention and attn is not None: attn_list.append(attn)
        final_attn = attn_list[-1] if self.output_attention and attn_list else None; return x, final_attn

# --- 时空 GRU 单元 (SpatialTemporalCell - 加入 Zoneout, 传递全局偏置参数) ---
class SpatialTemporalCell(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, n_heads, ffn_dim, st_layers, st_dropout_rate,
                 output_attention, qkv_bias=False, use_conv_proj=True, conv_kernel_size=3,
                 use_global_spatial_bias=False, zoneout_rate=0.0): # <<<--- 新增 zoneout_rate 和 use_global_spatial_bias
        super().__init__()
        if in_channels != out_channels: raise ValueError("SpatialTemporalCell 的 in_channels 必须等于 out_channels")
        self.in_channels = in_channels; self.out_channels = out_channels; self.num_nodes = num_nodes; self.output_attention = output_attention
        self.zoneout_rate = zoneout_rate

        self.spatial_attn_i = SpatialAttention(d_model=out_channels, num_nodes=num_nodes, n_heads=n_heads, ffn_dim=ffn_dim, st_layers=st_layers, dropout_rate=st_dropout_rate, output_attention=output_attention, qkv_bias=qkv_bias, use_conv_proj=use_conv_proj, conv_kernel_size=conv_kernel_size, use_global_spatial_bias=use_global_spatial_bias)
        self.spatial_attn_h = SpatialAttention(d_model=out_channels, num_nodes=num_nodes, n_heads=n_heads, ffn_dim=ffn_dim, st_layers=st_layers, dropout_rate=st_dropout_rate, output_attention=output_attention, qkv_bias=qkv_bias, use_conv_proj=use_conv_proj, conv_kernel_size=conv_kernel_size, use_global_spatial_bias=use_global_spatial_bias)
        gru_input_dim = out_channels; self.gru_projection_i = nn.Linear(gru_input_dim, out_channels * 3); self.gru_projection_h = nn.Linear(gru_input_dim, out_channels * 3)
        self.bias_ir = Parameter(torch.Tensor(out_channels)); self.bias_ii = Parameter(torch.Tensor(out_channels)); self.bias_in = Parameter(torch.Tensor(out_channels)); self.bias_hr = Parameter(torch.Tensor(out_channels)); self.bias_hi = Parameter(torch.Tensor(out_channels)); self.bias_hn = Parameter(torch.Tensor(out_channels))
        self.ln = nn.LayerNorm(out_channels); self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        for weight in [self.bias_ir, self.bias_ii, self.bias_in, self.bias_hr, self.bias_hi, self.bias_hn]:
            if weight is not None: init.uniform_(weight, -stdv, stdv)
    def forward(self, x, prev_hidden=None):
        B, N, C = x.shape
        if C != self.out_channels: raise ValueError(f"输入通道 {C} != 预期 {self.out_channels}")
        if prev_hidden is None: prev_hidden = torch.zeros_like(x)
        elif prev_hidden.shape != x.shape: raise ValueError(f"隐藏状态形状 {prev_hidden.shape} != 输入形状 {x.shape}")
        input_sp_attn, input_attn_map = self.spatial_attn_i(x); hidden_sp_attn, hidden_attn_map = self.spatial_attn_h(prev_hidden)
        input_fused = input_sp_attn; hidden_fused = hidden_sp_attn
        input_r, input_i, input_n = self.gru_projection_i(input_fused).chunk(3, dim=-1); hidden_r, hidden_i, hidden_n = self.gru_projection_h(hidden_fused).chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_r + self.bias_ir + hidden_r + self.bias_hr); update_gate = torch.sigmoid(input_i + self.bias_ii + hidden_i + self.bias_hi); new_gate = torch.tanh(input_n + self.bias_in + reset_gate * (hidden_n + self.bias_hn))
        calculated_next_hidden = (1.0 - update_gate) * new_gate + update_gate * prev_hidden
        if self.training and self.zoneout_rate > 0.0:
            zoneout_mask = (torch.rand_like(prev_hidden) < self.zoneout_rate).float()
            final_next_hidden = zoneout_mask * prev_hidden + (1.0 - zoneout_mask) * calculated_next_hidden
        else:
            final_next_hidden = calculated_next_hidden
        next_hidden_normalized = self.ln(final_next_hidden); output = next_hidden_normalized
        final_attn = input_attn_map if self.output_attention and input_attn_map is not None else (hidden_attn_map if self.output_attention else None)
        return output, final_next_hidden, final_attn

# --- 时间 Transformer 块 (TemporalTransformerBlock - 内部残差用DropPath) ---
class TemporalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.gelu, norm_layer=nn.LayerNorm):
        super().__init__(); self.norm1 = norm_layer(d_model); self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True); self.drop_path1 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity(); self.norm2 = norm_layer(d_model); self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate); self.drop_path2 = DropPath(dropout_rate) if dropout_rate > 0. else nn.Identity()
    def forward(self, x):
        shortcut1 = x; x_norm1 = self.norm1(x); attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1); x_after_attn = shortcut1 + self.drop_path1(attn_output); shortcut2 = x_after_attn; x_norm2 = self.norm2(x_after_attn); ffn_output = self.ffn(x_norm2); x_after_ffn = shortcut2 + self.drop_path2(ffn_output)
        return x_after_ffn

# --- 主模型 (SDT_GRU_Classifier) ---
class SDT_GRU_Classifier(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        print("初始化 SDT_GRU_Classifier (v12.5 - Zoneout, 全局空间偏置, EncoderLayer 使用 nn.Dropout)...")
        try:
            print(yaml.dump(model_cfg, default_flow_style=None, sort_keys=False))
        except Exception:
            print(model_cfg)

        self.num_input_dim = model_cfg['num_input_dim']; self.num_nodes = model_cfg['num_nodes']; self.num_classes = model_cfg['num_classes']; self.num_rnn_layers = model_cfg['num_rnn_layers']; self.num_rnn_units = model_cfg['num_rnn_units']; self.n_heads = model_cfg['n_heads']; self.ffn_dim = model_cfg['ffn_dim']; self.st_layers = model_cfg['st_layers']; self.st_dropout_rate = model_cfg.get('st_dropout_rate', 0.1); self.rnn_dropout_rate = model_cfg.get('rnn_dropout_rate', 0.1); self.classifier_dropout = model_cfg.get('classifier_dropout', 0.5); self.output_attention = model_cfg.get('output_attention', False); self.use_gap = model_cfg.get('use_gap', True); self.classifier_hidden_dim = model_cfg.get('classifier_hidden_dim', 0); self.qkv_bias = model_cfg.get('qkv_bias', False); self.use_conv_proj = model_cfg.get('use_conv_proj', True); self.conv_kernel_size = model_cfg.get('conv_kernel_size', 3);
        self.use_temporal_attn = model_cfg.get('use_temporal_attn', False); self.num_temporal_layers = model_cfg.get('num_temporal_layers', 1); self.temporal_n_heads = model_cfg.get('temporal_n_heads', 8); self.temporal_ffn_dim = model_cfg.get('temporal_ffn_dim', 256); self.temporal_dropout_rate = model_cfg.get('temporal_dropout_rate', 0.1);
        self.use_global_spatial_bias = model_cfg.get('use_global_spatial_bias', False)
        self.zoneout_rate = model_cfg.get('zoneout_rate', 0.0)
        if self.use_global_spatial_bias: print("启用 DSTA-Net 风格的全局空间偏置。")
        if self.zoneout_rate > 0.0: print(f"启用 Zoneout，比率: {self.zoneout_rate}")

        self.input_embedding = nn.Linear(self.num_input_dim, self.num_rnn_units); self.input_ln = nn.LayerNorm(self.num_rnn_units); self.input_dropout = nn.Dropout(self.st_dropout_rate); self.pos_encoder = PositionalEmbedding(self.num_rnn_units, max_len=model_cfg.get('max_seq_len', 200));
        print(f"输入维度: {self.num_input_dim}, RNN/特征维度: {self.num_rnn_units}"); print(f"时间位置编码器 (max_len={model_cfg.get('max_seq_len', 200)}, d_model={self.num_rnn_units})")

        self.encoder_cells = nn.ModuleList()
        for i in range(self.num_rnn_layers):
            self.encoder_cells.append(
                SpatialTemporalCell(
                    in_channels=self.num_rnn_units, out_channels=self.num_rnn_units,
                    num_nodes=self.num_nodes, n_heads=self.n_heads, ffn_dim=self.ffn_dim,
                    st_layers=self.st_layers, st_dropout_rate=self.st_dropout_rate,
                    output_attention=self.output_attention, qkv_bias=self.qkv_bias,
                    use_conv_proj=self.use_conv_proj, conv_kernel_size=self.conv_kernel_size,
                    use_global_spatial_bias=self.use_global_spatial_bias, # <<<--- 传递
                    zoneout_rate=self.zoneout_rate # <<<--- 传递
                )
            )
        print(f"已创建 {self.num_rnn_layers} 个 SpatialTemporalCell (Zoneout: {self.zoneout_rate > 0.0}, 全局空间偏置: {self.use_global_spatial_bias})")
        self.temporal_attn_blocks = None;
        if self.use_temporal_attn:
            if self.num_rnn_units % self.temporal_n_heads != 0: raise ValueError(f"Temporal Attention: num_rnn_units ({self.num_rnn_units}) 必须能被 temporal_n_heads ({self.temporal_n_heads}) 整除");
            print(f"启用 Temporal Attention ({self.num_temporal_layers} 层, {self.temporal_n_heads} 头, FFN dim {self.temporal_ffn_dim})");
            self.temporal_attn_blocks = nn.ModuleList([TemporalTransformerBlock(d_model=self.num_rnn_units, n_heads=self.temporal_n_heads, ffn_dim=self.temporal_ffn_dim, dropout_rate=self.temporal_dropout_rate) for _ in range(self.num_temporal_layers)]);
        else: print("Temporal Attention 未启用。");
        self.final_dropout = nn.Dropout(self.classifier_dropout); in_features_classifier = self.num_rnn_units; classifier_layers = [];
        if self.classifier_hidden_dim > 0: classifier_layers.append(nn.Linear(in_features_classifier, self.classifier_hidden_dim)); classifier_layers.append(nn.ReLU()); classifier_layers.append(nn.Dropout(self.classifier_dropout)); classifier_layers.append(nn.Linear(self.classifier_hidden_dim, self.num_classes));
        else: classifier_layers.append(nn.Linear(in_features_classifier, self.num_classes));
        self.classifier = nn.Sequential(*classifier_layers); print(f"分类器输入维度: {in_features_classifier}");
        self.global_step = 0; print("应用权重初始化..."); self.apply(self._init_weights); print("权重初始化完成。");

    def _init_weights(self, module):
            """递归地初始化模型权重"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if hasattr(module, 'bias') and module.bias is not None: # 确保有 bias 属性且不为 None
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine: # LayerNorm 的权重和偏置由 elementwise_affine 控制
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)): # 可以用元组一次判断多种类型
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None: # 确保有 bias 属性且不为 None
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'weight') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0.)
                if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0.)

    def forward(self, x, mask=None):
        B, T_in, N, C_in = x.shape;
        if N != self.num_nodes: raise ValueError(f"Input nodes {N} != expected {self.num_nodes}");
        if C_in != self.num_input_dim: raise ValueError(f"Input dim {C_in} != expected {self.num_input_dim}");
        x_flat = x.view(-1, C_in); x_embedded_flat = self.input_embedding(x_flat); x_embedded = x_embedded_flat.view(B, T_in, N, self.num_rnn_units); x_processed = self.input_ln(x_embedded); x_processed = self.input_dropout(x_processed); T_current = x_processed.size(1); time_pe = self.pos_encoder(x_processed); time_pe_expanded = time_pe.unsqueeze(2).expand(-1, -1, N, -1); x_processed = x_processed + time_pe_expanded; x_processed = x_processed.permute(1, 0, 2, 3).contiguous(); T_proc = x_processed.size(0);
        hidden_states = [None] * self.num_rnn_layers; all_outputs = []; all_attns = [] if self.output_attention else None; current_input_seq = x_processed;
        for t in range(T_proc):
            input_t_layer = current_input_seq[t]; layer_attns = [] if self.output_attention else None;
            for i, rnn_cell_module in enumerate(self.encoder_cells):
                prev_hidden_state = hidden_states[i]; output_cell, next_hidden_cell, attn_cell = rnn_cell_module(input_t_layer, prev_hidden_state); hidden_states[i] = next_hidden_cell; input_t_layer = output_cell;
                if i < self.num_rnn_layers - 1 and self.rnn_dropout_rate > 0.0: input_t_layer = F.dropout(input_t_layer, p=self.rnn_dropout_rate, training=self.training);
                if self.output_attention and attn_cell is not None: layer_attns.append(attn_cell);
            all_outputs.append(input_t_layer);
            if self.output_attention and layer_attns: all_attns.append(torch.stack(layer_attns, dim=1));
        stacked_outputs = torch.stack(all_outputs, dim=0);
        final_encoder_output_gru = None;
        if mask is not None:
            if mask.shape[1] != T_proc:
                if mask.shape[1] > T_proc: mask = mask[:, :T_proc];
                else: mask = None;
                if mask is None: logger.warning(f"Mask length mismatch with sequence length ({T_proc}). Ignoring mask.");
            if mask is not None:
                valid_lengths = mask.sum(dim=1);
                if (valid_lengths == 0).any(): logger.warning("Some samples have mask length 0. Using first time step output for these."); last_valid_indices = torch.zeros_like(valid_lengths); non_zero_mask = valid_lengths > 0; last_valid_indices[non_zero_mask] = (valid_lengths[non_zero_mask] - 1).clamp(min=0).long();
                else: last_valid_indices = (valid_lengths - 1).clamp(min=0).long();
                batch_indices = torch.arange(B, device=x.device);
                try: final_encoder_output_gru = stacked_outputs[last_valid_indices, batch_indices, :, :];
                except IndexError as e: logger.error(f"IndexError accessing stacked_outputs. Shapes: stacked={stacked_outputs.shape}, indices={last_valid_indices.shape}. Error: {e}"); final_encoder_output_gru = stacked_outputs[-1];
            else: final_encoder_output_gru = stacked_outputs[-1];
        else: final_encoder_output_gru = stacked_outputs[-1];
        final_representation = None;
        if self.use_temporal_attn and self.temporal_attn_blocks is not None:
            gru_output_btnd = stacked_outputs.permute(1, 0, 2, 3);
            if mask is not None: mask_expanded_btn1 = mask.unsqueeze(-1).float(); temporal_input_unmasked = gru_output_btnd.mean(dim=2); temporal_input = temporal_input_unmasked * mask_expanded_btn1;
            else: temporal_input = gru_output_btnd.mean(dim=2);
            temporal_output = temporal_input;
            for temp_block in self.temporal_attn_blocks: temporal_output = temp_block(temporal_output);
            if mask is not None: mask_t = mask.unsqueeze(-1).float(); masked_temporal_output = temporal_output * mask_t; time_step_count = mask_t.sum(dim=1, keepdim=True).clamp(min=1.0); final_representation = masked_temporal_output.sum(dim=1) / time_step_count.squeeze(-1);
            else: final_representation = temporal_output.mean(dim=1);
        else:
            if self.use_gap: final_representation = final_encoder_output_gru.mean(dim=1);
            else: logger.warning("use_temporal_attn=False and use_gap=False. Forcing GAP before classifier."); final_representation = final_encoder_output_gru.mean(dim=1);
        final_representation = self.final_dropout(final_representation); logits = self.classifier(final_representation);
        final_attns = None;
        if self.output_attention and all_attns is not None and all_attns:
            try: final_attns = torch.stack(all_attns, dim=1);
            except Exception as e: logger.warning(f"Failed to stack spatial attention maps: {e}"); final_attns = None;
        if self.training:
            if hasattr(self, 'global_step'): self.global_step += 1;
            else: self.global_step = 1;
        return logits, final_attns
