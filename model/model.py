# 文件名: model/sdt_transformer_v2.py(8.3)
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
model_logger = logging.getLogger("SDT_Transformer")
from timm.layers.drop import DropPath
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
        
        # 🔥 优化1: 融合QKV投影 - 一次矩阵乘法代替三次
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout_rate
        self.attn_drop = nn.Dropout(dropout_rate)

        # Phase 3 核心：动态上下文RPE（贴近 Hyperformer：零初始化 + 统一缩放 + 外连门控）
        if hop_matrix is not None:
            self.register_buffer('hop_matrix', hop_matrix)
            max_hop = hop_matrix.max().item()
            # RPE 向量零初始化，训练中逐步学习
            self.rpe_embedding = nn.Parameter(torch.zeros(max_hop + 1, self.head_dim))
            # 贴近 Hyperformer，不额外线性变换
            self.rpe_proj = None
            # per-head 恒等外连与门控（outer 初始即恒等，alpha 初始为 0）
            num_nodes = hop_matrix.size(0)
            self.outer = nn.Parameter(torch.stack([torch.eye(num_nodes) for _ in range(self.n_heads)], dim=0),
                                      requires_grad=True)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.rpe_embedding = None
            self.rpe_proj = None
            self.outer = None
            self.alpha = None

    def forward(self, x):
        B, V, D = x.shape

        # 1. 🔥 融合的QKV投影并分割
        qkv = self.qkv_proj(x)  # [B, V, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # 每个都是 [B, V, D]

        # 2. 转换为多头格式
        q = q.view(B, V, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, V, D_h]
        k = k.view(B, V, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, V, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数 a = q @ k^T
        attn_score = torch.matmul(q, k.transpose(-2, -1))  # [B, H, V, V]

        # 4. 计算 RPE 偏置（b），并与 a 相加
        if self.rpe_embedding is not None:
            rpe_vectors = self.rpe_embedding[self.hop_matrix]  # [V, V, D_h]
            # [B, H, V, D_h] × [V, V, D_h] → [B, H, V, V]
            attn_bias = torch.einsum('bhvd,vwd->bhvw', q, rpe_vectors)
            attn_score = attn_score + attn_bias

        # 5. 统一缩放并 softmax
        attn = (attn_score * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 6. 外连恒等与可学习门控（x = (alpha*attn + outer) @ v）
        if self.outer is not None and self.alpha is not None:
            eff_attn = self.alpha * attn + self.outer  # broadcast 到 [B, H, V, V]
        else:
            eff_attn = attn
        out_heads = torch.einsum('bhvw,bhwd->bhvd', eff_attn, v)

        # 7. 恢复形状并输出投影
        out = out_heads.transpose(1, 2).reshape(B, V, D)
        out = self.out_proj(out)
        
        # 注意：SDPA为了性能优化通常不返回attention weights
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


class GCNUnit(nn.Module):
    """仅局部路径的时空单元：LocalGCN + MultiScale_TemporalConv + 残差"""
    def __init__(self, in_channels, out_channels, A, stride=1, kernel_size=5, dilations=[1, 2], residual=True):
        super().__init__()
        self.gcn1 = LocalGCN(in_channels, out_channels, A, adaptive=True, residual=True)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations, residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = residual_conv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class residual_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(residual_conv, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x 

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
            bias=True)

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


class TransUnit(nn.Module):
    """仅全局路径的时空单元：EncoderLayer(MHSA) + MultiScale_TemporalConv + 残差"""
    def __init__(self, in_channels, out_channels, hop_matrix=None, n_heads=4, stride=1,
                 dropout_rate=0.0, drop_path_rate=0.0, residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.encoder = EncoderLayer(d_model=in_channels, n_heads=n_heads,
                                    dropout_rate=dropout_rate, drop_path_rate=drop_path_rate,
                                    hop_matrix=hop_matrix)
        # 通道对齐到 out_channels 再走 TCN
        self.channel_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=5, stride=stride,
                                            dilations=[1, 2], residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = residual_conv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # x: [B,C,T,V] -> [B*T,V,C] -> MHSA -> [B,C,T,V]
        B, C, T, V = x.shape
        x3 = rearrange(x, 'b c t v -> (b t) v c')
        x3, _ = self.encoder(x3)
        x_attn = rearrange(x3, '(b t) v c -> b c t v', b=B, t=T)
        y = self.relu(self.tcn1(self.channel_proj(x_attn)) + self.residual(x))
        return y


# === 主模型：SDT Transformer ===
class ScoreFusionGate(nn.Module):
    """分数级动态融合：根据两路 logits 计算逐类 gate，输出融合后的 logits。
    gate = sigmoid(MLP([logits_gcn, logits_trans]))
    out = gate * logits_gcn + (1 - gate) * logits_trans
    """
    def __init__(self, num_class, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(16, num_class)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_class, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_class),
            nn.Sigmoid()
        )
        # 使初始 gate≈0.5：最后层权重、偏置置零
        last = self.mlp[-2]
        if hasattr(last, 'weight'):
            nn.init.constant_(last.weight, 0)
        if hasattr(last, 'bias') and last.bias is not None:
            nn.init.constant_(last.bias, 0)

    def forward(self, logits_gcn, logits_trans):
        # 允许梯度流经门控与两路 logits
        x = torch.cat([logits_gcn, logits_trans], dim=-1)
        gate = self.mlp(x)
        return gate * logits_gcn + (1 - gate) * logits_trans


class GCNBackbone(nn.Module):
    def __init__(self, in_channels, base_channels, A):
        super().__init__()
        c = base_channels
        # 与 tdgcn.py 一致：在 l1 内完成升维 in_channels -> base_channels
        self.l1 = GCNUnit(in_channels, c, A, residual=False)
        self.l2 = GCNUnit(c, c, A)
        self.l3 = GCNUnit(c, c, A)
        self.l4 = GCNUnit(c, c, A)
        self.l5 = GCNUnit(c, c * 2, A, stride=2)
        self.l6 = GCNUnit(c * 2, c * 2, A)
        self.l7 = GCNUnit(c * 2, c * 2, A)
        self.l8 = GCNUnit(c * 2, c * 4, A, stride=2)
        self.l9 = GCNUnit(c * 4, c * 4, A)
        self.l10 = GCNUnit(c * 4, c * 4, A)

    def forward(self, x):
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
        return x


class TransBackbone(nn.Module):
    def __init__(self, base_channels, hop_matrix, n_heads=4, dropout_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        c = base_channels
        dp = drop_path_rate
        self.l1 = TransUnit(c, c, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp, residual=False)
        self.l2 = TransUnit(c, c, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l3 = TransUnit(c, c, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l4 = TransUnit(c, c, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l5 = TransUnit(c, c * 2, hop_matrix, n_heads=n_heads, stride=2, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l6 = TransUnit(c * 2, c * 2, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l7 = TransUnit(c * 2, c * 2, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l8 = TransUnit(c * 2, c * 4, hop_matrix, n_heads=n_heads, stride=2, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l9 = TransUnit(c * 4, c * 4, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)
        self.l10 = TransUnit(c * 4, c * 4, hop_matrix, n_heads=n_heads, dropout_rate=dropout_rate, drop_path_rate=dp)

    def forward(self, x):
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
        return x


class Model(nn.Module):
    """解耦双流（GCN/Transformer），各自走 TCN，末端分数级动态融合"""
    def __init__(self, model_cfg: dict):
        super().__init__()
        # 基础参数
        self.in_channels = 3
        self.num_nodes = model_cfg['num_nodes']
        self.num_class = model_cfg['num_class']
        self.num_person = model_cfg.get('num_person', 2)
        self.dropout_rate = model_cfg.get('dropout_rate', 0.1)
        self.drop_path_rate = model_cfg.get('drop_path_rate', 0.05)
        self.base_channels = model_cfg.get('base_channels', 64)

        # 数据归一化
        self.data_bn = nn.BatchNorm1d(self.num_person * self.in_channels * self.num_nodes)

        # 图与跳数矩阵
        graph = model_cfg.get('graph', None)
        graph_args = model_cfg.get('graph_args', {})
        if graph is None:
            raise ValueError("必须提供graph参数，例如 'graph.ntu_rgb_d.Graph'")
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = torch.from_numpy(self.graph.A).float()

        dataset_name = model_cfg.get('dataset_name', 'ntu_rgb_d')
        hop_matrix = get_precomputed_hop_matrix(dataset_name, device='cpu')
        self.register_buffer('hop_matrix', hop_matrix)

        # 关节绝对位置编码（仅用于 Transformer 流），形状: [1, V, C]
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_nodes, self.base_channels)
        )

        # 输入投影：仅为 Transformer 提供专属投影；GCN 流按 tdgcn 由 l1 升维
        self.input_proj_trans = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels, kernel_size=1),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(0.1)
        )

        # 两个独立骨干
        self.gcn_backbone = GCNBackbone(self.in_channels, self.base_channels, A)
        heads = max(2, min(8, self.base_channels // 16))
        self.trans_backbone = TransBackbone(self.base_channels, self.hop_matrix, n_heads=heads,
                                            dropout_rate=self.dropout_rate, drop_path_rate=self.drop_path_rate)

        # 两个分类头
        self.fc_gcn = nn.Linear(self.base_channels * 4, self.num_class)
        self.fc_trans = nn.Linear(self.base_channels * 4, self.num_class)

        # 分数级动态融合
        self.score_fusion = ScoreFusionGate(self.num_class)

        # 初始化
        nn.init.normal_(self.fc_gcn.weight, 0, math.sqrt(2. / self.num_class))
        nn.init.normal_(self.fc_trans.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)

        # 分支默认开关（可通过 YAML 的 model_args 直接控制）
        self.use_gcn_default = model_cfg.get('use_gcn', True)
        self.use_transformer_default = model_cfg.get('use_transformer', True)

    def forward(self, x, y=None, use_gcn=None, use_transformer=None):
        # 若未显式传入，则使用模型内的默认开关
        if use_gcn is None:
            use_gcn = self.use_gcn_default
        if use_transformer is None:
            use_transformer = self.use_transformer_default
        # 兼容多种输入
        if len(x.shape) == 3:
            B, T, VC = x.shape
            x = x.view(B, T, self.num_nodes, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        elif len(x.shape) == 4:
            B, T, V, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        B, C, T, V, M = x.size()
        # 归一化
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(B, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(B * M, C, T, V)
        x_norm = x

        # 双流并行（带可选开关）
        # GCN 流
        if use_gcn:
            # 按 tdgcn：不做提前投影，直接送入骨干，由 l1 升维
            x_g = self.gcn_backbone(x_norm)
            _, Cg, Tg, Vg = x_g.size()
            feat_g = x_g.view(B, M, Cg, -1).mean(3).mean(1)
            logits_g = self.fc_gcn(feat_g)
        else:
            logits_g = torch.zeros(B, self.num_class, device=x_norm.device, dtype=x_norm.dtype)

        # Transformer 流（加入关节绝对位置编码）
        if use_transformer:
            # Transformer 独立输入投影 + 绝对位置编码
            x_t_proj = self.input_proj_trans(x_norm)
            x_t_in = x_t_proj.permute(0, 2, 3, 1) + self.pos_embedding.unsqueeze(1)  # [B*,T,V,C]
            x_t_in = x_t_in.permute(0, 3, 1, 2).contiguous()                        # [B*,C,T,V]
            x_t = self.trans_backbone(x_t_in)
            _, Ct, Tt, Vt = x_t.size()
            feat_t = x_t.view(B, M, Ct, -1).mean(3).mean(1)
            logits_t = self.fc_trans(feat_t)
        else:
            logits_t = torch.zeros(B, self.num_class, device=x_norm.device, dtype=x_norm.dtype)

        # 融合逻辑
        if use_gcn and use_transformer:
            logits = self.score_fusion(logits_g, logits_t)
        elif use_gcn:
            logits = logits_g
        elif use_transformer:
            logits = logits_t
        else:
            logits = logits_g
        return logits, y