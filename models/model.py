import torch
import torch.nn as nn
import torch.nn.functional as F

# 基本卷积块：Conv1d -> BN -> ReLU
class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, groups=1, p_mode='circular'):
        super().__init__()
        pad = ((k - 1) // 2) * d  # same 感受野
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=pad, dilation=d, groups=groups,
                              bias=False, padding_mode=p_mode)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 深度可分离卷积块：DW + PW
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, p_mode='circular'):
        super().__init__()
        self.dw = CBR(in_ch, in_ch, k=k, s=s, d=d, groups=in_ch, p_mode=p_mode)
        self.pw = CBR(in_ch, out_ch, k=1, s=1, d=1, p_mode=p_mode)
    def forward(self, x):
        return self.pw(self.dw(x))

# 轻量 SE 注意力（可选）
class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(ch, ch // r, kernel_size=1)
        self.fc2 = nn.Conv1d(ch // r, ch, kernel_size=1)
    def forward(self, x):
        # GAP: [B, C, L] -> [B, C, 1]
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

# 残差块（带可分离卷积 + 可选 SE）
class ResBlock(nn.Module):
    def __init__(self, ch, k=3, d=1, use_se=True, p_mode='circular'):
        super().__init__()
        self.conv1 = DepthwiseSeparable(ch, ch, k=k, s=1, d=d, p_mode=p_mode)
        self.conv2 = DepthwiseSeparable(ch, ch, k=k, s=1, d=d, p_mode=p_mode)
        self.se = SE1D(ch) if use_se else nn.Identity()
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        out = self.bn(out)
        out = self.act(out + x)
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=4, in_len=360, base_ch=32, p_mode='circular'):
        """
        num_classes: 类别数
        in_len: 输入长度（默认 360）
        base_ch: 基础通道数
        p_mode: 'circular' 以匹配 360° 环形数据；如不支持可改 'zeros'
        """
        super().__init__()
        # Stem：多尺度感受野（7,5,3），提升初期特征提取能力
        self.stem = nn.Sequential(
            CBR(1, base_ch, k=7, p_mode=p_mode),
            CBR(base_ch, base_ch, k=5, p_mode=p_mode),
            CBR(base_ch, base_ch, k=3, p_mode=p_mode),
        )

        # Stage 1: 下采样到 L/2
        self.down1 = nn.Sequential(
            DepthwiseSeparable(base_ch, base_ch*2, k=5, s=2, p_mode=p_mode),  # stride=2
            ResBlock(base_ch*2, k=3, d=1, use_se=True, p_mode=p_mode),
        )

        # Stage 2: 下采样到 L/4
        self.down2 = nn.Sequential(
            DepthwiseSeparable(base_ch*2, base_ch*4, k=5, s=2, p_mode=p_mode),
            ResBlock(base_ch*4, k=3, d=1, use_se=True, p_mode=p_mode),
            ResBlock(base_ch*4, k=3, d=2, use_se=True, p_mode=p_mode),  # 空洞卷积扩大感受野
        )

        # 头部：GAP + 小型分类器
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                 # [B, C, L] -> [B, C, 1]
            nn.Flatten(),                             # [B, C]
            nn.Dropout(p=0.2),
            nn.Linear(base_ch*4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

        # 简单的权重初始化
        self.apply(self._init_weights)

        # 记录输入长度（可用于检查）
        self.in_len = in_len

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 360]
        x = x.unsqueeze(1)                 # [B, 1, L]
        x = self.stem(x)                   # [B, C, L]
        x = self.down1(x)                  # [B, 2C, L/2]
        x = self.down2(x)                  # [B, 4C, L/4]
        x = self.head(x)                   # [B, num_classes]
        return x
