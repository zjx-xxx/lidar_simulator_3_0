import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- 小组件 -----------

class SEModule(nn.Module):
    """Squeeze-and-Excitation for 1D features: [B,C,L]"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, C, L]
        w = self.pool(x).squeeze(-1)         # [B,C]
        w = self.fc(w).unsqueeze(-1)         # [B,C,1]
        return x * w

class ResidualConvBlock(nn.Module):
    """
    残差卷积块（环形 padding），支持 dilation / SE / 可选归一化
    in/out 通道一致，便于堆叠
    """
    def __init__(self, channels, kernel_size=5, dilation=1,
                 dropout=0.2, use_se=True, norm='group'):
        super().__init__()
        k1 = 3
        p1 = (k1 // 2) * dilation               # 保持长度
        k2 = 5
        p2 = (k2 // 2) * dilation
        Norm = (lambda c: nn.GroupNorm(8, c)) if norm == 'group' else nn.BatchNorm1d

        self.conv1 = nn.Conv1d(channels, channels, k1,
                               padding=p1, dilation=dilation,
                               padding_mode='circular')
        self.norm1 = Norm(channels)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, k2,
                               padding=p2, dilation=dilation,
                               padding_mode='circular')
        self.norm2 = Norm(channels)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.use_se = use_se
        if use_se:
            self.se = SEModule(channels, reduction=8)

    def forward(self, x):
        residual = x
        x = self.conv1(x); x = self.norm1(x); x = self.act1(x); x = self.drop1(x)
        x = self.conv2(x); x = self.norm2(x)
        if self.use_se:
            x = self.se(x)
        x = self.act2(x); x = self.drop2(x)
        return x + residual

class ResidualMLP(nn.Module):
    """用于回归头的残差 MLP，稳定加深"""
    def __init__(self, dim, hidden=64, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y

# ----------- 主网络（加深版） -----------

class RegressionNetwork(nn.Module):
    """
    深化版：多层残差Conv1d（环形） + FiLM + 门控 + 残差MLP回归头
    接口与原先保持一致
    """
    def __init__(
        self,
        use_embedding: bool = True,
        n_road: int = 10,
        n_turn: int = 5,
        d_road: int = 8,
        d_turn: int = 4,
        base_channels: int = 32,
        num_blocks: int = 8,                  # ★ 控制深度
        dilation_schedule = (1,2,4,1,2,4),    # ★ 多尺度
        kernel_size: int = 5,
        stem_pool: bool = True,               # 首层是否下采样
        norm: str = 'group',                  # 'group' 或 'batch'
        dropout: float = 0.2,
        use_se: bool = True,
        out_dim: int = 1,
    ):
        super().__init__()
        self.use_embedding = use_embedding
        C = base_channels
        k = kernel_size
        p = k // 2

        # ----- Stem -----
        self.stem = nn.Sequential(
            nn.Conv1d(1, C, kernel_size=k, padding=p, padding_mode='circular'),
            nn.GroupNorm(8, C) if norm=='group' else nn.BatchNorm1d(C),
            nn.GELU(),
            nn.MaxPool1d(2) if stem_pool else nn.Identity(),  # 可选下采样
        )

        # ----- Residual Conv Blocks -----
        # 通道保持不变，靠 dilation 扩感受野
        blocks = []
        ds = list(dilation_schedule)
        if len(ds) < num_blocks:
            # 不够就循环
            ds = (ds * ((num_blocks + len(ds) - 1)//len(ds)))[:num_blocks]
        for d in ds[:num_blocks]:
            blocks.append(
                ResidualConvBlock(C, kernel_size=k, dilation=d,
                                  dropout=dropout, use_se=use_se, norm=norm)
            )
        self.blocks = nn.Sequential(*blocks)

        # ----- Global pooling + 线性投影 -----
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(C, 128)    # 更大的全局表征
        self.proj_act = nn.GELU()

        # ----- 条件向量处理（与原版一致） -----
        if use_embedding:
            self.emb_road = nn.Embedding(n_road, d_road)
            self.emb_turn = nn.Embedding(n_turn, d_turn)
            cond_dim = d_road + d_turn
        else:
            cond_dim = 2

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.film_gamma = nn.Linear(128, 128)
        self.film_beta  = nn.Linear(128, 128)
        self.gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        # ----- 回归头：残差 MLP 堆叠 -----
        self.head_pre = nn.Linear(128, 128)
        self.head_blocks = nn.Sequential(
            ResidualMLP(128, hidden=128, dropout=dropout),
            ResidualMLP(128, hidden=128, dropout=dropout),
        )
        self.head_out = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_lidar, road_type, turn_direction):
        if x_lidar.dim() == 2:
            x_lidar = x_lidar.unsqueeze(1)  # [B,1,L]

        # Backbone
        x = self.stem(x_lidar)               # [B,C,L']
        x = self.blocks(x)                   # 深层残差
        x = self.global_pool(x).squeeze(-1)  # [B,C]
        x = self.proj_act(self.proj(x))      # [B,128]
        lidar_feat = x

        # 条件向量
        if self.use_embedding:
            road = road_type.long().view(-1)
            turn = turn_direction.long().view(-1)
            cond_raw = torch.cat([self.emb_road(road), self.emb_turn(turn)], dim=1)
        else:
            road = road_type.float().view(-1, 1)
            turn = turn_direction.float().view(-1, 1)
            cond_raw = torch.cat([road, turn], dim=1)
        cond = self.cond_mlp(cond_raw)       # [B,128]

        # FiLM + 门控
        gamma = self.film_gamma(cond)
        beta  = self.film_beta(cond)
        mod = gamma * lidar_feat + beta
        gate_w = self.gate(cond)             # [B,1]
        fused = gate_w * mod + (1 - gate_w) * lidar_feat

        # 回归头（更深）
        y = self.head_pre(fused)
        y = self.head_blocks(y)
        out = self.head_out(y)               # [B,out_dim]
        if out.shape[1] == 1:
            out = out.squeeze(1)
        return out