# 文件名: model/sgt_net.py(8.6修改)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
import logging
from einops import rearrange
from torch.autograd import Variable
# 设置日志
model_logger = logging.getLogger("Model")
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from .precomputed_hop_matrices import get_precomputed_hop_matrix

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class PositionalEmbedding(nn.Module):
    """
    创建一个可学习的、统一的时空位置编码
    """
    def __init__(self, d_model, max_len_t=100, num_nodes=25):
        super().__init__()
        # 创建一个 nn.Parameter，这就是模型将要学习的位置信息
        # 形状为 [1, C, T_max, V_max]，1是为了方便广播
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_len_t, num_nodes))
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, x):
        """
        输入 x: [B, C, T, V]
        输出: x + PE
        """
        B, C, T, V = x.shape
        # 通过切片操作，让位置编码的尺寸与当前输入的时空尺寸完全匹配
        embedding_to_add = self.pos_embedding[:, :, :T, :V]
        return x + embedding_to_add


class AttentionLayer(nn.Module):
    """
    全局注意力层 - 使用基于跳数距离的相对位置编码 (RPE)
    """
    def __init__(self, d_model, n_heads, dropout_rate=0.0, hop_matrix=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_rate = dropout_rate
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # --- 全局注意力：初始化相对位置编码 (RPE) ---
        self.rpe = None

        if hop_matrix is not None:
            # 全局层：初始化相对位置编码 (RPE)
            self.register_buffer('hop_matrix', hop_matrix)
            max_hop = hop_matrix.max().item()
            # 为每个头学习一个从“距离”到“偏置值”的映射
            self.rpe = nn.Parameter(torch.randn(n_heads, max_hop + 1))
        
    def forward(self, x):
        B, V, D = x.shape
        q = self.q_proj(x).view(B, V, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, V, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, V, self.n_heads, self.head_dim).transpose(1, 2)

        # --- 构建相对位置编码偏置 ---
        # 1. 初始化偏置为 None，表示默认无偏置
        attn_bias = None

        # 2. 准备RPE偏置
        if self.rpe is not None:
            # self.rpe: [H, max_hop+1], self.hop_matrix: [V, V] -> rpe_bias: [H, V, V]
            rpe_bias = self.rpe[:, self.hop_matrix]
            # 直接赋值，并为 batch 维度 unsqueeze -> [1, H, V, V]
            attn_bias = rpe_bias.unsqueeze(0)

        # 4. 【核心】调用 PyTorch 2.0+ 的原生 scaled_dot_product_attention
        #    它会自动处理 attn_bias 为 None 的情况（即不添加任何偏置）
        #    并在硬件和数据支持时，自动启用 FlashAttention 或其他内存优化核
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,  # 将我们的浮点偏置作为 mask 传入
            dropout_p=self.dropout_rate if self.training else 0.0
        )
        
        # 5. 后续处理保持不变
        out = out.transpose(1, 2).reshape(B, V, D)
        out = self.out_proj(out)
        return out, None


class EncoderLayer(nn.Module):
    """单层编码器"""
    
    def __init__(self, d_model, n_heads, dropout_rate=0.0, drop_path_rate=0.0, hop_matrix=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AttentionLayer(
            d_model=d_model, 
            n_heads=n_heads, 
            dropout_rate=dropout_rate,
            hop_matrix=hop_matrix
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        # Pre-LN架构
        shortcut1 = x
        x_norm1 = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm1)
        x = shortcut1 + self.drop_path1(attn_output)
        return x, attn_weights

class SpatialCrossAttentionFusionGate(nn.Module):
    def __init__(self, feature_dim, n_heads=8, dropout_rate=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(feature_dim, feature_dim * 3, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, feat1, feat2):
        B, C, T, V = feat1.shape
        
        # 直接在4D tensor上操作，减少reshape
        # 重新排列为 [B, T, V, C] 便于后续处理
        x1 = feat1.permute(0, 2, 3, 1)  # [B, T, V, C]
        x2 = feat2.permute(0, 2, 3, 1)  # [B, T, V, C]
        
        # 批量计算QKV
        qkv1 = self.qkv_proj(x1)  # [B, T, V, 3C]
        qkv2 = self.qkv_proj(x2)  # [B, T, V, 3C]
        
        # 重塑为多头格式
        qkv1 = qkv1.reshape(B, T, V, 3, self.n_heads, self.head_dim)
        qkv2 = qkv2.reshape(B, T, V, 3, self.n_heads, self.head_dim)
        
        q1, k1, v1 = qkv1.permute(3, 0, 1, 4, 2, 5).unbind(0)  # [B, T, H, V, D]
        q2, k2, v2 = qkv2.permute(3, 0, 1, 4, 2, 5).unbind(0)  # [B, T, H, V, D]
        
        # 使用einsum进行注意力计算 - 更高效
        # 交叉注意力
        attn1 = torch.einsum('bthid,bthjd->bthij', q1, k2) * self.scale
        attn2 = torch.einsum('bthid,bthjd->bthij', q2, k1) * self.scale
        
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)
        
        if self.training:
            attn1 = self.dropout(attn1)
            attn2 = self.dropout(attn2)
        
        # 输出计算
        out1 = torch.einsum('bthij,bthjd->bthid', attn1, v2)
        out2 = torch.einsum('bthij,bthjd->bthid', attn2, v1)
        
        # 融合和后处理
        out1 = out1.reshape(B, T, V, C)
        out2 = out2.reshape(B, T, V, C)
        
        fused = self.out_proj(out1) + self.out_proj(out2)
        fused = self.norm(fused)
        
        # 恢复到原始格式
        return fused.permute(0, 3, 1, 2)  # [B, C, T, V]

# 添加TD-GCN的空间处理模块
class TDGC(nn.Module):
    """TD-GCN的核心空间处理模块"""
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(TDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1, gamma=0.1):
        x1, x3 = self.conv1(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x3.mean(-3).unsqueeze(-1) - x3.mean(-3).unsqueeze(-2))
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)
        x1 = x1 * beta + x5 * gamma
        return x1


class LocalGCN(nn.Module):
    """局部路径：使用GCN处理局部物理连接"""
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(LocalGCN, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(TDGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            # A已经是torch.Tensor，直接转换为Parameter
            self.PA = nn.Parameter(A.float())
        else:
            # A已经是torch.Tensor，直接使用
            self.A = Variable(A.float(), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            # 确保A在正确的设备上
            A = self.A.to(x.device)
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha, self.beta, self.gamma)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class DynamicDualStreamAttention(nn.Module):
    """空间注意力 - 局部GCN + 全局Transformer + 动态融合 """
    def __init__(self, in_channels, out_channels, A=None, hop_matrix=None, global_n_heads=8,
                 fusion_type='dynamic_gate', fusion_gate_n_heads=8,
                 dropout_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        if in_channels != out_channels:
            self.pre_projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.pre_projection = nn.Identity()
        # 1. 局部路径：GCN处理 
        self.local_gcn = LocalGCN(out_channels, out_channels, A, adaptive=True, residual=True)
        
        # 2. 全局路径：Transformer的输入投影 
        self.global_layer = EncoderLayer(
            d_model=out_channels, n_heads=global_n_heads,
            dropout_rate=dropout_rate, drop_path_rate=drop_path_rate, hop_matrix=hop_matrix
        )

        # 5. 融合策略
        self.fusion_type = fusion_type
        if self.fusion_type == 'dynamic_gate':
            self.fusion = SpatialCrossAttentionFusionGate(out_channels, n_heads=fusion_gate_n_heads)
        elif self.fusion_type == 'add':
            self.fusion = lambda x, y: x + y
        elif self.fusion_type == 'concat':
            self.fusion = lambda x, y: torch.cat([x, y], dim=1) 
            self.proj = nn.Conv2d(out_channels * 2, out_channels, 1)  
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.pre_projection(x)
        # 局部路径输出
        local_output = self.local_gcn(x)
        
        # 全局Transformer路径输出
        x_3d = rearrange(x, 'b c t v -> (b t) v c')
        global_output, attn_weights = self.global_layer(x_3d)
        global_output_4d = rearrange(global_output, '(b t) v c -> b c t v', b=B, t=T)
        
        # --- 直接对两路输出进行融合 ---
        if self.fusion_type == 'concat':
            fused_output = self.fusion(local_output, global_output_4d)
            fused_output = self.proj(fused_output) 
        else:
            fused_output = self.fusion(local_output, global_output_4d)
        
        return fused_output, attn_weights

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 dilations=[1,2],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += self.residual(x)
        return out


class SpatioTemporalUnit(nn.Module):
    """简化的时空融合单元 - 参考TD-GCN和HD-GCN的设计"""
    def __init__(self, in_channels, out_channels, A=None, hop_matrix=None,
                 stride=1, residual=True, dropout_rate=0.0, drop_path_rate=0.0, fusion_type='dynamic_gate'):
        super().__init__()
        
        # --- 1. 空间处理：GCN + Transformer双流 ---
        self.spatial_attn = DynamicDualStreamAttention(in_channels=in_channels,out_channels=out_channels,
            A=A, hop_matrix=hop_matrix,global_n_heads=8,dropout_rate=dropout_rate,drop_path_rate=drop_path_rate)
        
        # --- 2. 时间处理：多尺度TCN ---
        self.temporal_conv = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=5, stride=stride,
            dilations=[1, 2], residual=False)
        
        # --- 3. 残差连接 ---
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x  
        # 空间处理
        x_spatial, _ = self.spatial_attn(x)
        # 时间处理
        x_temporal = self.temporal_conv(x_spatial)
        # 残差连接和激活
        y = self.relu(x_temporal + self.residual(residual))
        return y

# === 主模型：Model ===
class Model(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        # 基础参数
        self.in_channels = 3  # 直接定死为3，对应骨架的x,y,z坐标
        self.num_nodes = model_cfg['num_nodes']
        self.num_class = model_cfg['num_class']
        self.num_person = model_cfg.get('num_person', 2)
        self.dropout_rate = model_cfg.get('dropout_rate', 0.1)
        self.drop_path_rate = model_cfg.get('drop_path_rate', 0.1)
        self.data_bn = nn.BatchNorm1d(self.num_person * self.in_channels * self.num_nodes)
        self.base_channels = 64
        
        # 图配置
        graph = model_cfg.get('graph', None)
        graph_args = model_cfg.get('graph_args', {})

        if graph is None:
            raise ValueError("必须提供graph参数，例如 'graph.ntu_rgb_d.Graph'")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            
        A = torch.from_numpy(self.graph.A).float() # (K, V, V) e.g. (3, 25, 25)

        # 准备用于【全局注意力】的跳数距离矩阵 (V, V)
        dataset_name = model_cfg.get('dataset_name', 'ntu_rgb_d')
        hop_matrix = get_precomputed_hop_matrix(dataset_name, device='cpu')
        self.register_buffer('hop_matrix', hop_matrix)
        self.pos_embedding = PositionalEmbedding(d_model=self.base_channels,max_len_t=model_cfg.get('max_frames', 200), num_nodes=self.num_nodes )
        self.early_local_stream = nn.ModuleList()
        self.early_global_stream = nn.ModuleList()
        # L1层：负责从 in_channels -> base_channels 的升维
        # 局部流L1
        self.early_local_stream.append(
            nn.Sequential(
                LocalGCN(self.in_channels, self.base_channels, A, residual=False),
                MultiScale_TemporalConv(self.base_channels, self.base_channels)
            )
        )
        # 全局流L1
        self.early_global_stream.append(
            nn.Sequential(
                nn.Linear(self.in_channels, self.base_channels), # 输入投影升维
                EncoderLayer(self.base_channels, n_heads=8, hop_matrix=self.hop_matrix, 
                                dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate),
                MultiScale_TemporalConv(self.base_channels, self.base_channels)
            )
        )
        
        # L2-L4层: 在 base_channels 维度上进行处理
        for _ in range(3): # 剩下3层
            # 局部流
            self.early_local_stream.append(
                nn.Sequential(
                    LocalGCN(self.base_channels, self.base_channels, A, residual=True),
                    MultiScale_TemporalConv(self.base_channels, self.base_channels)
                )
            )
            # 全局流
            self.early_global_stream.append(
                nn.Sequential(
                    nn.Linear(self.base_channels, self.base_channels), # 维度不变的投影
                    EncoderLayer(self.base_channels, n_heads=8, hop_matrix=self.hop_matrix, 
                                 dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate),
                    MultiScale_TemporalConv(self.base_channels, self.base_channels)
                )
            )
            
        self.fusion_gate = SpatialCrossAttentionFusionGate(self.base_channels, n_heads=8)
        
        # 第2阶段：下采样 + 特征增强 (l5-l7)
        self.l5 = SpatioTemporalUnit(self.base_channels, self.base_channels*2, A=A, hop_matrix=self.hop_matrix, stride=2, dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)
        self.l6 = SpatioTemporalUnit(self.base_channels*2, self.base_channels*2, A=A, hop_matrix=self.hop_matrix, dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)
        self.l7 = SpatioTemporalUnit(self.base_channels*2, self.base_channels*2, A=A, hop_matrix=self.hop_matrix, dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)
        
        # 第3阶段：高级特征提取 (l8-l10)
        self.l8 = SpatioTemporalUnit(self.base_channels*2, self.base_channels*4, A=A, hop_matrix=self.hop_matrix, stride=2, dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)
        self.l9 = SpatioTemporalUnit(self.base_channels*4, self.base_channels*4, A=A, hop_matrix=self.hop_matrix, dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)
        self.l10 = SpatioTemporalUnit(self.base_channels*4, self.base_channels*4, A=A, hop_matrix=self.hop_matrix, dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)
        # 分类头
        self.fc = nn.Linear(self.base_channels*4, self.num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)
        if self.dropout_rate > 0:
            self.drop_out = nn.Dropout(self.dropout_rate)
        else:
            self.drop_out = lambda x: x
        
        # 初始化权重
        self.apply(weights_init)

    def forward(self, x):
        # --- 数据预处理 (不变) ---
        if len(x.shape) == 3:
            B, T, VC = x.shape
            x = x.view(B, T, self.num_nodes, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        elif len(x.shape) == 4:
            B, T, V, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        B, C, T, V, M = x.size()
        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(B, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(B * M, C, T, V)
        
        # --- 阶段1: 早期独立演化 ---
        x_local = x.clone()
        x_global = x.clone()
        
        for i in range(len(self.early_local_stream)):
            # --- 处理局部流 ---
            local_proc = self.early_local_stream[i]
            x_local = local_proc(x_local)
            
            # --- 处理全局流 ---
            global_seq = self.early_global_stream[i]
            glob_proj = global_seq[0]
            glob_enc = global_seq[1]
            glob_tcn = global_seq[2]
            
            B_glob, C_glob, T_glob, V_glob = x_global.shape
            x_global_3d = rearrange(x_global, 'b c t v -> (b t) v c')
            x_global_3d = glob_proj(x_global_3d) # L1层在这里完成升维
            x_global_3d, _ = glob_enc(x_global_3d)
            x_global_4d = rearrange(x_global_3d, '(b t) v c -> b c t v', b=B_glob, t=T_glob)
            x_global = glob_tcn(x_global_4d)
            
            # 在第一层升维后，添加位置编码
            if i == 0:
                x_local = self.pos_embedding(x_local)
                x_global = self.pos_embedding(x_global)

        # --- 阶段2: 融合 ---
        x_fused = self.fusion_gate(x_local, x_global)
        
        # --- 阶段3: 后期融合处理 ---
        x = self.l5(x_fused)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
    
        # 简化的池化和分类
        _, C_final, T, V = x.size()
        x = x.view(B, M, C_final, -1)
        x = x.mean(3).mean(1)  # 直接使用平均池化
        x = self.drop_out(x)
        logits = self.fc(x)   
        return logits 