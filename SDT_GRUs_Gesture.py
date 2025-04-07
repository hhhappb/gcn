# 文件名: SDT_GRUs_Gesture_Runnable_CN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
# import random # 对于分类模型的逻辑通常不需要
# import yaml # 配置加载通常在模型文件外部进行

print(f"PyTorch 版本: {torch.__version__}")

# --- 辅助函数 ---
def zoneout(prev_h, next_h, rate, training=True):
    """应用 Zoneout。
       注意: 原始论文的实现可能略有不同。
       此版本遵循描述：随机复制上一个隐藏状态单元。
    """
    if training and rate > 0.0:
        # 创建一个掩码，其中 1 表示保留上一个隐藏状态
        # 使用 torch.bernoulli 根据 rate 生成 0/1 掩码
        mask = torch.bernoulli(torch.full_like(prev_h, rate)).bool()
        # 应用掩码：mask 为 True 时保留 prev_h，为 False 时保留 next_h
        next_h = torch.where(mask, prev_h, next_h)
    elif not training and rate > 0.0:
         # 在推理（非训练）期间，混合上一个和下一个隐藏状态
         next_h = rate * prev_h + (1 - rate) * next_h
    # 如果 rate 为 0 或不在训练模式，next_h 隐式保持不变
    return next_h


# --- 位置编码 (来自 embedding.py) ---
# 注意: 这个 PE 最初是为 Transformer 中的 *时间* 维度设计的。
# 在 SpatialAttention 中，它被应用于 *节点* 维度，这可能不太直观。
# 我们保留它，但它在节点维度（SpatialAttention内部）的应用对于手势数据值得商榷。
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        # 在对数空间中一次性计算位置编码。
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False # 位置编码不参与梯度更新

        position = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)
        # 计算除数项 (div_term)，用于不同维度的频率缩放
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 使用 sin 计算偶数索引的位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 使用 cos 计算奇数索引的位置编码
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # 增加 batch 维度 -> Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe) # 注册为 buffer，模型保存时会保存，但不参与训练

    def forward(self, x):
        # x 的形状: (Batch, SeqLen, d_model) 或类似，其中维度 1 是序列/长度维度
        # 返回适用于 x 长度的位置编码: (1, SeqLen, d_model)
        return self.pe[:, :x.size(1)]


# --- 空间注意力组件 (来自 spatial_attention.py) ---
class AttentionLayer(nn.Module):
    """多头自注意力层"""
    def __init__(self, d_model, n_heads, dropout, output_attention):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads   # 每个头的 key 维度
        self.d_values = d_model // n_heads # 每个头的 value 维度
        self.output_attention = output_attention # 是否输出注意力权重图

        # 线性投影层
        self.query_projection = nn.Linear(d_model, d_model) # 简化：直接投影到 d_model
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        # 输出投影层，将多头结果合并
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: 输入特征 (Batch, NumNodes, d_model) - 这里的 NumNodes 相当于注意力机制中的序列长度
        B, N, _ = x.shape # Batch 大小, 节点数 (序列长度), 特征维度
        H = self.n_heads   # 头数

        # 投影并重塑以适应多头注意力
        # (B, N, D) -> (B, N, H, D_k) -> (B, H, N, D_k)
        queries = self.query_projection(x).view(B, N, H, self.d_keys).transpose(1, 2)
        keys = self.key_projection(x).view(B, N, H, self.d_keys).transpose(1, 2)
        values = self.value_projection(x).view(B, N, H, self.d_values).transpose(1, 2)

        # 计算注意力分数: (B, H, N, D_k) x (B, H, D_k, N) -> (B, H, N, N)
        scale = 1. / math.sqrt(self.d_keys) # 缩放因子
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        # 应用 softmax 获得注意力权重，并应用 dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # 计算加权后的 values: (B, H, N, N) x (B, H, N, D_v) -> (B, H, N, D_v)
        weighted_values = torch.matmul(attn_weights_dropped, values)

        # 合并多头结果并进行最终投影
        # (B, H, N, D_v) -> (B, N, H, D_v) -> (B, N, H*D_v=D)
        # contiguous() 保证内存连续，然后 view
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_projection(weighted_values) # (B, N, D)

        if self.output_attention:
            # 返回原始注意力权重（dropout前）用于分析
            return output, attn_weights.detach() # detach() 防止梯度回传
        else:
            return output, None


class FeedForwardModule(nn.Module):
    """前馈网络模块 (通常在自注意力之后)"""
    def __init__(self, d_model, ffn_dim, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)      # 第一个线性层，扩展维度
        self.dropout = nn.Dropout(dropout)
        self.activation = activation                   # 激活函数 (例如 ReLU 或 GeLU)
        self.linear2 = nn.Linear(ffn_dim, d_model)      # 第二个线性层，恢复维度

    def forward(self, x):
        # x: (Batch, NumNodes, d_model)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class EncoderLayer(nn.Module): # 重命名自 Encoder，更清晰
    """标准的 Transformer 编码器层，适用于空间维度"""
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate, activation=F.relu, output_attention=False):
        super().__init__()
        # 多头自注意力子层
        self.self_attn = AttentionLayer(d_model, n_heads, dropout_rate, output_attention)
        # 前馈网络子层
        self.ffn = FeedForwardModule(d_model, ffn_dim, activation, dropout_rate)
        # 层归一化 (Layer Normalization)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (Batch, NumNodes, d_model)

        # 1. 多头自注意力 + 残差连接 & 层归一化
        attn_output, attn_weights = self.self_attn(x)
        x = x + self.dropout(attn_output) # 残差连接
        x = self.norm1(x)                 # 层归一化

        # 2. 前馈网络 + 残差连接 & 层归一化
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # 残差连接
        x = self.norm2(x)                 # 层归一化

        return x, attn_weights # 返回处理后的表示和注意力权重


class SpatialAttention(nn.Module):
    """ 空间注意力模块，使用 Transformer 编码器层。
        修改后：不再执行初始嵌入，期望输入已经是 d_model 维度。
    """
    def __init__(self,
                 # in_channels, # 不再需要，假设输入是 d_model
                 d_model,     # 模型内部维度
                 num_nodes,   # 如果使用节点 PE 则需要
                 n_heads,
                 ffn_dim,
                 st_layers,      # Transformer Encoder 层数
                 dropout_rate,
                 output_attention):
        super().__init__()
        self.d_model = d_model
        self.st_layers = st_layers
        self.output_attention = output_attention

        # --- 移除了 TokenEmbedding ---

        # --- 可选: 节点的 Positional Embedding ---
        # 注意: 将 sinusoidal PE 应用于节点可能不是最优选择。
        # 如果节点顺序/ID 有意义，可以考虑使用可学习的嵌入 nn.Embedding(num_nodes, d_model)。
        self.use_node_pos_emb = False # 默认禁用，因为对于关节点通常不需要
        if self.use_node_pos_emb:
             self.pos_embedding = PositionalEmbedding(d_model, max_len=num_nodes)

        # Transformer Encoder 层列表
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model,
                             n_heads=n_heads,
                             ffn_dim=ffn_dim,
                             dropout_rate=dropout_rate,
                             output_attention=output_attention)
                for _ in range(st_layers) # 使用 st_layers 控制层数
            ]
        )
        # 在所有编码器层之后的最终层归一化 (可选但常见)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: 输入特征 (Batch, NumNodes, d_model)
        B, N, D = x.shape
        assert D == self.d_model, f"SpatialAttention 期望输入维度 {self.d_model}, 实际为 {D}"

        if self.use_node_pos_emb:
            # 输入 x 仅用于获取长度 N -> 返回 (1, N, D)
            node_pe = self.pos_embedding(x)
            x = x + node_pe # 直接相加

        attn_list = []
        # 依次通过 Transformer Encoder 层
        for encoder in self.encoders:
            x, attn = encoder(x) # Encoder 输入 (B, N, D)
            if self.output_attention and attn is not None:
                attn_list.append(attn) # attn shape (B, H, N, N)

        x = self.final_norm(x) # 应用最终归一化

        # --- 组合注意力图 ---
        final_attn = None
        if self.output_attention and attn_list:
            # attn_list 是长度为 st_layers 的列表, 每个元素是 (B, H, N, N) 或 None
            # 过滤掉 None (如果 EncoderLayer 可能返回 None)
            valid_attns = [a for a in attn_list if a is not None]
            if valid_attns:
                 # 堆叠所有层的注意力 (B, st_layers, H, N, N)
                 final_attn = torch.stack(valid_attns, dim=1)

        # 返回处理后的节点表示和注意力图
        return x, final_attn


# --- 时空单元 (SpatialTemporalCell) ---
class SpatialTemporalCell(nn.Module):
    """ 核心单元，结合了空间注意力和 GRU 更新 """
    def __init__(self,
                 in_channels,      # 此单元的输入特征维度 (例如 num_rnn_units)
                 out_channels,     # GRU 隐藏状态的维度 (== num_rnn_units)
                 num_nodes,
                 n_heads,
                 ffn_dim,
                 st_layers,        # SpatialAttention *内部* 的层数
                 st_dropout_rate,
                 output_attention):
        super().__init__()
        # 为简单起见，当前假设输入和输出通道数相同
        assert in_channels == out_channels, "SpatialTemporalCell 目前假设 in_channels == out_channels"
        self.in_channels = in_channels
        self.out_channels = out_channels # == SpatialAttention 的 d_model
        self.num_nodes = num_nodes
        self.output_attention = output_attention

        # 对当前输入 'x' (已处理的输入) 进行空间注意力计算
        self.spatial_attn_i = SpatialAttention(d_model=out_channels, # 在隐藏维度上操作
                                               num_nodes=num_nodes,
                                               n_heads=n_heads,
                                               ffn_dim=ffn_dim,
                                               st_layers=st_layers,
                                               dropout_rate=st_dropout_rate,
                                               output_attention=output_attention)
        # 对上一个隐藏状态 'h' 进行空间注意力计算
        self.spatial_attn_h = SpatialAttention(d_model=out_channels,
                                               num_nodes=num_nodes,
                                               n_heads=n_heads,
                                               ffn_dim=ffn_dim,
                                               st_layers=st_layers,
                                               dropout_rate=st_dropout_rate,
                                               output_attention=output_attention)

        # GRU 相关的线性投影层
        # 输入是 SpatialAttention 的输出 (out_channels 维度)
        self.gru_projection_i = nn.Linear(out_channels, out_channels * 3) # 对应 r, i, n 门
        self.gru_projection_h = nn.Linear(out_channels, out_channels * 3) # 对应 r, i, n 门

        # GRU 偏置项 (标准 GRU 实现通常包含)
        self.bias_ir = Parameter(torch.Tensor(self.out_channels)) # 输入到重置门
        self.bias_ii = Parameter(torch.Tensor(self.out_channels)) # 输入到更新门
        self.bias_in = Parameter(torch.Tensor(self.out_channels)) # 输入到新门 (候选状态)
        self.bias_hr = Parameter(torch.Tensor(self.out_channels)) # 隐藏状态到重置门
        self.bias_hi = Parameter(torch.Tensor(self.out_channels)) # 隐藏状态到更新门
        self.bias_hn = Parameter(torch.Tensor(self.out_channels)) # 隐藏状态(经过重置门后)到新门

        # GRU 更新后的层归一化
        # 对每个节点的特征维度独立进行归一化
        self.ln = nn.LayerNorm(self.out_channels)

        self.reset_parameters() # 初始化参数

    def reset_parameters(self):
        """初始化模型参数"""
        # 简化初始化
        stdv = 1.0 / math.sqrt(self.out_channels)
        for name, weight in self.named_parameters():
            if weight.dim() > 1: # 初始化权重矩阵 (例如线性层权重)
                 # init.uniform_(weight, -stdv, stdv)
                 init.xavier_uniform_(weight) # 使用 Xavier 初始化可能更常用
            elif "bias" in name: # 初始化偏置项
                 init.zeros_(weight) # 将偏置初始化为 0

        # 确保 LayerNorm 的参数得到初始化 (PyTorch 默认初始化为 1 和 0)
        # if hasattr(self.ln, 'weight') and self.ln.weight is not None:
        #     init.ones_(self.ln.weight)
        # if hasattr(self.ln, 'bias') and self.ln.bias is not None:
        #     init.zeros_(self.ln.bias)


    def forward(self, x, prev_hidden=None):
        # x: 当前时间步的输入 (Batch, NumNodes, InChannels==OutChannels)
        # prev_hidden: 上一个时间步的隐藏状态 (Batch, NumNodes, OutChannels)
        B, N, C = x.shape
        assert C == self.in_channels, f"输入通道数 {C} != 期望的 {self.in_channels}"

        # 如果没有提供上一个隐藏状态，则初始化为零
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, N, self.out_channels, dtype=x.dtype, device=x.device)

        # --- 应用空间注意力 ---
        # input_sp: (B, N, OutChannels), input_attn: (B, st_layers, H, N, N) or None
        input_sp, input_attn = self.spatial_attn_i(x)           # 处理当前输入
        hidden_sp, hidden_attn = self.spatial_attn_h(prev_hidden) # 处理上一个隐藏状态

        # --- GRU 更新计算 ---
        # 线性投影
        # chunk(3, dim=-1) 将最后一个维度分成 3 块，分别对应 r, i, n 门
        input_r, input_i, input_n = self.gru_projection_i(input_sp).chunk(3, dim=-1)
        hidden_r, hidden_i, hidden_n = self.gru_projection_h(hidden_sp).chunk(3, dim=-1)

        # 计算门控（element-wise 逐元素操作）
        reset_gate = torch.sigmoid(input_r + self.bias_ir + hidden_r + self.bias_hr)
        update_gate = torch.sigmoid(input_i + self.bias_ii + hidden_i + self.bias_hi)
        # 注意：重置门作用于计算新候选状态时 *隐藏状态* 的那部分贡献
        new_gate = torch.tanh(input_n + self.bias_in + reset_gate * (hidden_n + self.bias_hn))

        # 计算下一个隐藏状态
        # (1 - 更新门) * 新候选状态 + 更新门 * 上一个隐藏状态
        next_hidden = (1.0 - update_gate) * new_gate + update_gate * prev_hidden # 这里应该用 prev_hidden

        # --- 应用层归一化 ---
        # 对每个节点和每个特征维度进行归一化
        next_hidden_normalized = self.ln(next_hidden)

        # 当前时间步的输出就是归一化后的隐藏状态
        output = next_hidden_normalized

        # --- 组合注意力图 (如果需要输出) ---
        final_attn = None
        if self.output_attention:
             if input_attn is not None and hidden_attn is not None:
                 # 沿新维度堆叠 (例如维度 1)
                 # 形状: (B, 2, st_layers, H, N, N)
                 # 这里的 '2' 代表来自 input 和 hidden 的注意力
                 final_attn = torch.stack([input_attn, hidden_attn], dim=1)
             elif input_attn is not None:
                 # (B, 1, st_layers, H, N, N)
                 final_attn = input_attn.unsqueeze(1)
             elif hidden_attn is not None:
                 # (B, 1, st_layers, H, N, N)
                 final_attn = hidden_attn.unsqueeze(1)

        # 返回当前步输出，最终隐藏状态（用于下一步），以及注意力（可选）
        # 注意：返回归一化后的状态作为隐藏状态传递给下一步
        return output, next_hidden_normalized, final_attn


# --- 主要的分类器模型 ---
class SDT_GRU_Classifier(nn.Module):
    """ 时空 GRU 分类器，用于序列分类任务 """
    def __init__(self, model_cfg):
        super().__init__()
        print("初始化 SDT_GRU_Classifier，配置如下:")
        print(model_cfg)

        # --- 核心参数 ---
        self.num_input_dim = model_cfg['num_input_dim'] # 输入特征维度 (例如 x, y, z 为 3)
        self.num_nodes = model_cfg['num_nodes']         # 骨骼节点数量
        self.num_classes = model_cfg['num_classes']     # 手势类别数量

        # --- 模型维度和层数 ---
        self.num_rnn_layers = model_cfg['num_rnn_layers']   # 堆叠的 SpatialTemporalCell 层数
        self.num_rnn_units = model_cfg['num_rnn_units']   # GRU 隐藏维度 (也即 d_model)
        self.n_heads = model_cfg['n_heads']               # 注意力头数
        self.ffn_dim = model_cfg['ffn_dim']               # SpatialAttention 中 FFN 的隐藏维度
        self.st_layers = model_cfg['st_layers']           # SpatialAttention *内部* 的 EncoderLayer 层数

        # --- Dropout 和 Zoneout 比率 ---
        # .get 提供默认值，增加配置灵活性
        self.st_dropout_rate = model_cfg.get('st_dropout_rate', 0.1) # SpatialAttention/FFN 中的 Dropout
        self.rnn_dropout_rate = model_cfg.get('rnn_dropout_rate', 0.1)# RNN 层之间的 Dropout
        self.zoneout_rate = model_cfg.get('zoneout_rate', 0.0)      # GRU 隐藏状态的 Zoneout

        # --- 选项 ---
        self.output_attention = model_cfg.get('output_attention', False) # 是否输出注意力图
        self.use_gap = model_cfg.get('use_gap', True) # 分类前是否使用全局平均池化 (GAP)

        # --- 输入嵌入层 ---
        # 将原始输入特征（如 3D 坐标）投影到模型的隐藏维度
        self.input_embedding = nn.Linear(self.num_input_dim, self.num_rnn_units)
        # 可选：嵌入后添加层归一化
        self.input_ln = nn.LayerNorm(self.num_rnn_units)
        # 嵌入后应用 Dropout
        self.input_dropout = nn.Dropout(self.st_dropout_rate)

        # --- 核心编码器单元 (SpatialTemporalCell 列表) ---
        self.encoder_cells = nn.ModuleList()
        # 所有层都在 num_rnn_units 维度上操作
        for i in range(self.num_rnn_layers):
            self.encoder_cells.append(
                SpatialTemporalCell(
                    in_channels=self.num_rnn_units,    # 输入维度
                    out_channels=self.num_rnn_units,   # 输出维度
                    num_nodes=self.num_nodes,
                    n_heads=self.n_heads,
                    ffn_dim=self.ffn_dim,
                    st_layers=self.st_layers,
                    st_dropout_rate=self.st_dropout_rate,
                    output_attention=self.output_attention
                )
            )

        # --- 分类头 ---
        # 分类器之前的 Dropout
        self.classifier_dropout = nn.Dropout(model_cfg.get('classifier_dropout', 0.5))
        # 根据是否使用 GAP 确定分类器的输入特征数
        in_features_classifier = self.num_rnn_units * (1 if self.use_gap else self.num_nodes)

        # 构建分类器（可以是多层）
        classifier_layers = []
        # 可选的分类器隐藏层维度，为 0 表示直接线性映射
        hidden_classifier_dim = model_cfg.get('classifier_hidden_dim', 0)
        if hidden_classifier_dim > 0:
             classifier_layers.append(nn.Linear(in_features_classifier, hidden_classifier_dim))
             classifier_layers.append(nn.ReLU())
             # 可在分类器内部再加 Dropout
             classifier_layers.append(nn.Dropout(model_cfg.get('classifier_dropout', 0.5)))
             classifier_layers.append(nn.Linear(hidden_classifier_dim, self.num_classes))
        else:
             # 直接从输入特征映射到类别数
             classifier_layers.append(nn.Linear(in_features_classifier, self.num_classes))

        self.classifier = nn.Sequential(*classifier_layers) # 将层组合成 Sequential 模块

        self.global_step = 0 # 用于可能的外部用途（例如日志记录步数）

    def forward(self, x, mask=None):
        """
        前向传播函数
        x: 输入序列 (Batch, SeqLen, NumNodes, InputDim)
        mask: (可选)布尔掩码 (Batch, SeqLen)，指示有效帧 (True=有效)
        """
        B, T, N, C = x.shape # 获取输入维度
        # 断言检查输入维度是否符合预期
        assert N == self.num_nodes, f"输入节点数不匹配: 收到 {N}, 期望 {self.num_nodes}"
        assert C == self.num_input_dim, f"输入维度不匹配: 收到 {C}, 期望 {self.num_input_dim}"

        # --- 1. 初始嵌入与准备 ---
        # 为了高效计算，先将时序和节点维度合并，再进行线性嵌入
        # (B, T, N, C) -> (B*T*N, C)
        x_flat = x.reshape(-1, C)
        # (B*T*N, C) -> (B*T*N, D) D=num_rnn_units
        x_embedded_flat = self.input_embedding(x_flat)
        # 恢复原始形状 (除了最后一个维度变为 D)
        # (B*T*N, D) -> (B, T, N, D)
        x_embedded = x_embedded_flat.reshape(B, T, N, self.num_rnn_units)

        # 应用可选的层归一化和 Dropout
        x_processed = self.input_ln(x_embedded)
        x_processed = self.input_dropout(x_processed)

        # 调整维度顺序以适应 RNN/逐时间步处理: (B, T, N, D) -> (T, B, N, D)
        x_processed = x_processed.permute(1, 0, 2, 3)

        # --- 2. 通过编码器层 (SpatialTemporalCell) 处理 ---
        # 初始化隐藏状态列表，每个层一个
        hidden_states = [None] * self.num_rnn_layers
        # 存储最后一个 RNN 层在每个时间步的输出
        all_outputs = []
        # 存储所有注意力图 (如果 output_attention=True)
        all_attentions = []

        current_input_seq = x_processed # Shape (T, B, N, D)

        # 逐时间步处理
        for t in range(T):
            input_for_t = current_input_seq[t] # 获取当前时间步的输入, Shape (B, N, D)
            next_layer_input = input_for_t # 作为第一层的输入

            layer_attns_t = [] # 存储当前时间步所有层的注意力图

            # 逐层处理
            for i, rnn_cell in enumerate(self.encoder_cells):
                prev_hidden = hidden_states[i] # 获取上一时间步该层的隐藏状态
                # output, next_hidden shapes: (B, N, D) D=num_rnn_units
                # attn shape: (B, 2, st_layers, H, N, N) or None
                output, next_hidden, attn = rnn_cell(x=next_layer_input, prev_hidden=prev_hidden)

                # --- 应用 Zoneout ---
                if self.zoneout_rate > 0.0 and prev_hidden is not None:
                    next_hidden = zoneout(prev_hidden, next_hidden, self.zoneout_rate, self.training)

                hidden_states[i] = next_hidden     # 更新隐藏状态，用于下一时间步
                next_layer_input = output          # 当前层的输出作为下一层的输入

                # --- 在 RNN 层之间应用 Dropout (除了最后一层之后) ---
                if i < self.num_rnn_layers - 1 and self.rnn_dropout_rate > 0.0:
                    next_layer_input = F.dropout(next_layer_input, p=self.rnn_dropout_rate, training=self.training)

                # 收集注意力图
                if self.output_attention and attn is not None:
                    layer_attns_t.append(attn) # 存储当前层在 t 时刻的注意力图

            # 存储当前时间步 *最后一层* 的输出
            all_outputs.append(next_layer_input)

            # 存储当前时间步所有层的注意力图列表
            if self.output_attention and layer_attns_t:
                 all_attentions.append(layer_attns_t) # 列表的列表

        # --- 3. 选择最终的序列表示 ---
        # 将所有时间步的最后一层输出堆叠起来: (T, B, N, D)
        stacked_outputs = torch.stack(all_outputs, dim=0)

        # 如果提供了掩码 (mask)，则根据掩码找到每个样本最后一个有效帧的输出
        if mask is not None:
            # mask 形状: (B, T)，计算每个样本的有效长度
            valid_lengths = mask.sum(dim=1) # Shape (B,)
            # 计算最后一个有效帧的索引 (长度减 1)
            last_valid_idx = valid_lengths - 1 # Shape (B,)
            # 确保索引非负 (对于空序列，索引为 -1，需要 clamp 到 0)
            last_valid_idx = last_valid_idx.long().clamp(min=0)

            # 使用高级索引从 stacked_outputs 中提取对应帧的输出
            # 需要正确构造索引，使其能作用于 stacked_outputs (T, B, N, D)
            # 方法一: 使用 gather
            # idx_gather = last_valid_idx.view(1, B, 1, 1).expand(1, B, N, self.num_rnn_units)
            # final_encoder_output = torch.gather(stacked_outputs, 0, idx_gather).squeeze(0) # Shape (B, N, D)
            # 方法二: 使用高级索引 (可能更直观)
            batch_indices = torch.arange(B, device=x.device) # [0, 1, ..., B-1]
            final_encoder_output = stacked_outputs[last_valid_idx, batch_indices, :, :] # Shape (B, N, D)

        else:
            # 如果没有掩码，简单地取最后一个时间步的输出
            final_encoder_output = stacked_outputs[-1] # Shape (B, N, D)

        # --- 4. 通过分类头进行分类 ---
        # 可选：在节点维度上进行全局平均池化 (GAP)
        if self.use_gap:
            # (B, N, D) -> (B, D)
            final_representation = final_encoder_output.mean(dim=1)
        else:
            # 将节点和特征维度展平
            # (B, N, D) -> (B, N * D)
            final_representation = final_encoder_output.reshape(B, -1)

        # 在分类器前应用 Dropout
        final_representation = self.classifier_dropout(final_representation)

        # 通过分类器获得 logits (类别得分)
        logits = self.classifier(final_representation) # Shape (B, num_classes)

        # --- 5. 准备注意力输出 (如果需要) ---
        # 如何聚合或返回注意力图需要仔细设计
        final_attns = None
        if self.output_attention and all_attentions:
            # 示例：仅返回最后一个有效时间步的注意力图列表
            # (需要配合 mask 找到 last_valid_idx)
            # last_t_attns = all_attentions[last_valid_idx] # 这需要更复杂的索引
            # 简化：返回最后一个时间步的注意力列表
            # last_t_attns_list = all_attentions[-1] # 列表，元素为 (B, 2, st_layers, H, N, N)
            # final_attns = torch.stack(last_t_attns_list, dim=1) # (B, num_layers, 2, st_layers, H, N, N)
             pass # 占位符 - 根据你的需求决定如何返回/使用注意力

        # 更新全局步数（如果在训练中）
        if self.training:
            self.global_step += 1

        # 返回 logits 和可选的注意力图
        return logits, final_attns


# --- 示例用法 (用于测试代码结构) ---
if __name__ == "__main__":
    print("测试 SDT_GRU_Classifier 结构...")

    # --- 示例配置 (需要根据具体数据集调整，例如 NTU RGB+D 60) ---
    cfg_example = {
        'num_input_dim': 3,      # 输入维度 (例如 x, y, z 坐标)
        'num_nodes': 25,         # 节点数 (例如 NTU 数据集有 25 个关节点)
        'num_classes': 60,       # 类别数 (例如 NTU 数据集有 60 个动作类别)
        'num_rnn_layers': 2,     # 堆叠的 SpatialTemporalCell 层数
        'num_rnn_units': 128,    # 隐藏状态维度 (d_model)
        'n_heads': 8,            # 注意力头数
        'ffn_dim': 256,          # SpatialAttention 中 FFN 的隐藏维度
        'st_layers': 2,          # SpatialAttention *内部* 的 EncoderLayer 层数
        'st_dropout_rate': 0.1,  # SpatialAttention 中的 Dropout
        'rnn_dropout_rate': 0.1, # RNN 层之间的 Dropout
        'classifier_dropout': 0.5,# 分类器 Dropout
        'zoneout_rate': 0.0,     # Zoneout 比率 (0 表示禁用)
        'output_attention': False,# 是否输出注意力图
        'use_gap': True,         # 是否在分类前使用 GAP
        'classifier_hidden_dim': 64 # 分类器中的隐藏层维度 (0 表示无隐藏层)
    }

    # --- 实例化模型 ---
    model = SDT_GRU_Classifier(cfg_example)
    print(model) # 打印模型结构
    # 计算并打印模型参数量
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {param_count:,}")

    # --- 创建虚拟输入数据 ---
    batch_size = 4       # 批次大小
    seq_len = 50       # 示例序列长度 (帧数)
    num_nodes = cfg_example['num_nodes']
    input_dim = cfg_example['num_input_dim']
    # 输入形状: (Batch, SeqLen, NumNodes, InputDim)
    dummy_input = torch.randn(batch_size, seq_len, num_nodes, input_dim)

    # --- 可选：创建虚拟掩码 (Mask) ---
    # 假设某些样本最后几帧是填充的
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # dummy_mask[0, -10:] = False # 第一个样本最后 10 帧是填充
    # dummy_mask[1, -5:] = False  # 第二个样本最后 5 帧是填充

    print(f"\n输入形状: {dummy_input.shape}")
    print(f"掩码形状: {dummy_mask.shape}")

    # --- 执行前向传播 ---
    model.train() # 设置为训练模式以启用 Dropout/Zoneout
    # 测试带掩码的情况
    # logits, attns = model(dummy_input, mask=dummy_mask)
    # 测试不带掩码的情况
    logits, attns = model(dummy_input, mask=None)

    print(f"输出 logits 形状: {logits.shape}") # 期望: (Batch, num_classes)
    if attns is not None:
        # 注意力图的形状取决于返回逻辑
        print(f"输出 attention 形状: {attns.shape}")
    else:
        print("输出 attention: None (符合默认设置)")

    # --- 测试评估模式 ---
    model.eval() # 设置为评估模式
    with torch.no_grad(): # 关闭梯度计算
         logits_eval, _ = model(dummy_input, mask=None)
    print(f"输出 logits 形状 (评估模式): {logits_eval.shape}")