# 文件名: model/sgt_net8.10.py(8.10)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
from einops import rearrange
from torch.autograd import Variable
from .tem_mixf import Temporal_MixFormer
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

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.0, hop_matrix=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # 融合QKV投影优化
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout_rate
        self.attn_drop = nn.Dropout(dropout_rate)

        # 每头独立的RPE bias
        if hop_matrix is not None:
            self.register_buffer('hop_matrix', hop_matrix)
            max_hop = hop_matrix.max().item()
            # 每个头对每种距离都有独立的标量偏置
            self.rpe_bias = nn.Parameter(torch.zeros(self.n_heads, max_hop + 1))
        else:
            self.rpe_bias = None

    def forward(self, x):
        B, V, D = x.shape

        # 1.融合的QKV投影
        qkv = self.qkv_proj(x)  # [B, V, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # 每个都是 [B, V, D]

        # 2. 转换为多头格式
        q = q.view(B, V, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, V, D_h]
        k = k.view(B, V, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, V, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数
        attn_score = torch.matmul(q, k.transpose(-2, -1))  # [B, H, V, V]

        # 4. 轻量级RPE bias（每头独立）
        if self.rpe_bias is not None:
            # [H, max_hop+1] -> [H, V, V] 通过索引查找
            rpe_bias_lookup = self.rpe_bias[:, self.hop_matrix]  # [H, V, V]
            attn_score = attn_score + rpe_bias_lookup.unsqueeze(0)  # broadcast to [B, H, V, V]

        # 5. 统一缩放并softmax
        attn = (attn_score * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 6. 直接使用学习到的注意力
        out_heads = torch.einsum('bhvw,bhwd->bhvd', attn, v)

        # 7. 恢复形状并输出投影
        out = out_heads.transpose(1, 2).reshape(B, V, D)
        out = self.out_proj(out)
        
        return out, attn


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

class DynamicFusionGate(nn.Module):
    """动态融合门"""
    
    def __init__(self, feature_dim, gate_hidden_dim=None):
        super().__init__()
        if gate_hidden_dim is None:
            gate_hidden_dim = feature_dim // 4
        
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(gate_hidden_dim),
            nn.Linear(gate_hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        final_linear_layer = self.gate_network[-2]
        # 将其偏置初始化为0
        if final_linear_layer.bias is not None:
            nn.init.constant_(final_linear_layer.bias, 0)

    def forward(self, feat1, feat2):
        # feat1, feat2 的形状是 [B, C, T, V]
        
        # 1. 将维度C换到最后
        feat1_p = feat1.permute(0, 2, 3, 1) # -> [B, T, V, C]
        feat2_p = feat2.permute(0, 2, 3, 1) # -> [B, T, V, C]
        
        # 2. 在最后一个维度（C）上拼接
        combined = torch.cat([feat1_p, feat2_p], dim=-1) # -> [B, T, V, 2*C]
        
        # 3. 计算门控值
        gate = self.gate_network(combined) # Linear作用于2*C维度
        
        # 4. 进行融合
        fused = gate * feat1_p + (1 - gate) * feat2_p
        
        # 5. 将维度换回来
        fused_out = fused.permute(0, 3, 1, 2) # -> [B, C, T, V]
        
        return fused_out

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
    """空间注意力 - 【创新点：局部GCN + 全局Transformer + 动态融合】"""
    def __init__(self, d_model,
                 A=None,  # 添加A参数
                 hop_matrix=None,
                 # 局部路径参数（GCN）
                 local_gcn_layers=1,
                 # 全局路径参数（Transformer）
                 global_st_layers=1, global_n_heads=4,
                 # 融合参数
                 fusion_type='dynamic_gate', fusion_gate_hidden_dim=None,
                 dropout_rate=0.0, drop_path_rate=0.0, **kwargs):
        super().__init__()
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.dropout_rate = dropout_rate
        
        # --- 1. 局部路径：GCN处理 (替换原来的局部注意力) ---
        self.local_gcn_layers = nn.ModuleList()
        for i in range(local_gcn_layers):
            # 每个GCN层都需要A参数
            self.local_gcn_layers.append(
                LocalGCN(d_model, d_model, A, adaptive=True, residual=True)
            )
        
        # --- 2. 全局路径：Transformer处理 (保持不变) ---
        self.global_layers = nn.ModuleList()
        for i in range(global_st_layers):
            self.global_layers.append(
                EncoderLayer(
                    d_model=d_model,
                    n_heads=global_n_heads,
                    dropout_rate=dropout_rate,
                    drop_path_rate=drop_path_rate,
                    hop_matrix=hop_matrix
                )
            )
        
        # 融合策略 - 保持动态融合创新点
        if fusion_type == 'dynamic_gate':
            self.fusion = DynamicFusionGate(d_model, fusion_gate_hidden_dim)
        elif fusion_type == 'add':
            self.fusion = lambda x, y: x + y
        elif fusion_type == 'concat':
            self.fusion = lambda x, y: torch.cat([x, y], dim=-1)
            self.proj = nn.Linear(d_model * 2, d_model)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, x):
        # x是4D: [B, C, T, V]
        B, C, T, V = x.shape
        
        # 1. 局部路径处理（GCN）- 直接使用4D输入
        local_output = x
        for layer in self.local_gcn_layers:
            local_output = layer(local_output)
        
        # 2. 全局路径处理（Transformer）- 需要转换为3D
        # 从4D [B, C, T, V] 转换为3D [B*T, V, C]
        x_3d = rearrange(x, 'b c t v -> (b t) v c')
        
        global_output = x_3d
        for layer in self.global_layers:
            global_output, attn_weights = layer(global_output)
        
        # 将Transformer输出转换回4D格式用于融合
        global_output_4d = rearrange(global_output, '(b t) v c -> b c t v', b=B, t=T)

        # 3. 融合
        if self.fusion_type == 'add':
            # 相加融合：仅在 add 分支做连续化，避免 out= 以支持自动求导
            local_tensor = local_output.contiguous()
            global_tensor = global_output_4d.contiguous()
            fused_output = torch.add(local_tensor, global_tensor)
        elif self.fusion_type == 'concat':
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
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
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
    def __init__(self, in_channels, out_channels, A=None, hop_matrix=None,
                 stride=1, spatial_config=None, residual=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # --- 1. 空间处理模块 (类似TCN_GCN_unit中的gcn1) ---
        if spatial_config is None:
            spatial_config = {
                'local_gcn_layers': 1,
                'global_st_layers': 1,
                'fusion_type': 'dynamic_gate',
                'fusion_gate_hidden_dim': None,
                'dropout_rate': 0.0,
            }

        # 自动计算全局路径的头数（局部路径使用GCN，不需要多头注意力）
        spatial_config['global_n_heads'] = 4
        
        self.spatial_attn = DynamicDualStreamAttention(
            d_model=in_channels,
            A=A,  # 添加A参数传递给LocalGCN
            hop_matrix=hop_matrix,  
            **spatial_config
        )
        
        # --- 2. 时间处理模块 (类似TCN_GCN_unit中的tcn1) ---
        # 直接使用MultiScale_TemporalConv的默认参数
        self.temporal_conv = MultiScale_TemporalConv(
            in_channels, in_channels,  # 保持通道数不变
            kernel_size=5,  # 默认值
            stride=stride,
            dilations=[1, 2],  # 默认值
            residual=False  # 在单元内部不使用残差，在单元级别使用
        )
        
        # --- 3. 残差连接处理 (类似TCN_GCN_unit中的residual) ---
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # 需要下采样或通道数变化时的残差连接
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        # --- 4. 通道投影 (如果需要) ---
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.channel_proj = nn.Identity()

    def forward(self, x):
        # 保存输入用于残差连接
        residual = x
        
        # --- 1. 空间处理 ---
        # 直接传递4D数据给spatial_attn，它会内部处理GCN和Transformer的维度需求
        x_spatial, _ = self.spatial_attn(x)
        
        # --- 2. 时间处理 ---
        x_temporal = self.temporal_conv(x_spatial)
        
        # --- 3. 通道投影 ---
        x_out = self.channel_proj(x_temporal)
        
        # --- 4. 残差连接和激活 (类似TCN_GCN_unit的最终处理) ---
        y = self.relu(x_out + self.residual(residual))
        
        return y

# === 主模型：SDT Transformer ===
class Model(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        
        # 基础参数
        self.in_channels = 3  # 直接定死为3，对应骨架的x,y,z坐标
        self.num_nodes = model_cfg['num_nodes']
        self.num_class = model_cfg['num_class']
        self.num_person = model_cfg.get('num_person', 2)
        
        # 🔥【修正3 & 4】在顶层定义正则化参数
        self.dropout_rate = model_cfg.get('dropout_rate', 0.1)
        self.drop_path_rate = model_cfg.get('drop_path_rate', 0.1)
        
        # 添加数据归一化层
        self.data_bn = nn.BatchNorm1d(self.num_person * self.in_channels * self.num_nodes)
        
        # 模型维度 - 直接定死base_channels为64
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
        # 使用预计算的跳数矩阵，根据数据集名称获取
        dataset_name = model_cfg.get('dataset_name', 'ntu_rgb_d')
        hop_matrix = get_precomputed_hop_matrix(dataset_name, device='cpu')
        self.register_buffer('hop_matrix', hop_matrix)
        
        # 输入投影
        self.input_proj = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=1)
        
        # 空间注意力配置
        spatial_config = {
            'local_gcn_layers': model_cfg.get('sa_local_gcn_layers', 1),
            'global_st_layers': model_cfg.get('sa_global_st_layers', 1),
            'fusion_type': model_cfg.get('spatial_fusion_type', 'dynamic_gate'),
            'fusion_gate_hidden_dim': model_cfg.get('spatial_fusion_gate_hidden_dim', None),
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,  
        }
        
        # 直接定义l1到l10层，参考HD-GCN的方式
        # 第1阶段：基础特征提取 (l1-l4, stride=1)
        self.l1 = SpatioTemporalUnit(self.base_channels, self.base_channels,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config,
                                    residual=False)
        self.l2 = SpatioTemporalUnit(self.base_channels, self.base_channels,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config)
        self.l3 = SpatioTemporalUnit(self.base_channels, self.base_channels,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config)
        self.l4 = SpatioTemporalUnit(self.base_channels, self.base_channels,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config)
        
        # 第2阶段：下采样 + 特征增强 (l5-l7)
        self.l5 = SpatioTemporalUnit(self.base_channels, self.base_channels*2,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    stride=2,
                                    spatial_config=spatial_config)
        self.l6 = SpatioTemporalUnit(self.base_channels*2, self.base_channels*2,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config)
        self.l7 = SpatioTemporalUnit(self.base_channels*2, self.base_channels*2,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config)
        
        # 第3阶段：高级特征提取 (l8-l10)
        self.l8 = SpatioTemporalUnit(self.base_channels*2, self.base_channels*4,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    stride=2,
                                    spatial_config=spatial_config)
        self.l9 = SpatioTemporalUnit(self.base_channels*4, self.base_channels*4,
                                    A=A,  # 添加A参数
                                    hop_matrix=self.hop_matrix,
                                    spatial_config=spatial_config)
        self.l10 = SpatioTemporalUnit(self.base_channels*4, self.base_channels*4,
                                     A=A,  # 添加A参数
                                     hop_matrix=self.hop_matrix,
                                     spatial_config=spatial_config)

        # 分类头
        self.fc = nn.Linear(self.base_channels*4, self.num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)
        if self.dropout_rate > 0:
            self.drop_out = nn.Dropout(self.dropout_rate)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        """
        输入:
            x: 骨架序列 (B, C, T, V, M) -        
        返回:
            logits: (B, num_class)
        """
        if len(x.shape) == 3:
            # 输入格式: (B, T, V*C)
            B, T, VC = x.shape
            x = x.view(B, T, self.num_nodes, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        elif len(x.shape) == 4:
            # 输入格式: (B, T, V, C) -> (B, C, T, V, 1)
            B, T, V, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        B, C, T, V, M = x.size()
        
        # 数据归一化（先归一化）
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(B, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(B * M, C, T, V)
        
        # 输入投影（再投影）
        x = self.input_proj(x)
        # 时空融合处理 - 直接调用l1到l10，参考HD-GCN的方式
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
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