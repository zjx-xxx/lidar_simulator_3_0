# model_reg.py
import torch
import torch.nn as nn


class RegressionNetwork(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        ff: int = 256,
        num_layers: int = 4,
        max_len: int = 360,
        dropout: float = 0.1,
        max_angle: float = 30.0,   # 输出角度上限（度）
    ):
        super().__init__()
        self.max_len = max_len
        self.max_angle = max_angle

        # 标量距离 -> d_model
        self.input_proj = nn.Linear(1, d_model)

        # 可学习位置编码（索引 0..L-1），注意：LiDAR 是环形，后续可替换为 sin/cos 编码
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 条件信息嵌入（与你原数据保持一致的类别大小/维度）
        self.road_emb = nn.Embedding(4, 4)      # 道路类型
        self.towards_emb = nn.Embedding(3, 4)   # 转向方向（数据集中已有的先验/脚本标签）

        # 回归头
        self.fc = nn.Sequential(
            nn.Linear(d_model + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, road_type: torch.Tensor, turn_direction: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L=360]  距离序列（建议预处理：裁剪上限并归一化）
        road_type: [B]   0..3
        turn_direction: [B]  0..2
        return:
            out_deg: [B]  回归角度（度），范围 [-max_angle, max_angle]
        """
        x = x.float()
        B, L = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len={self.max_len}")

        # 投影 + 位置
        h = self.input_proj(x.unsqueeze(-1))                  # [B, L, d_model]
        pos = self.pos_emb(torch.arange(L, device=x.device))  # [L, d_model]
        h = h + pos.unsqueeze(0)

        # 编码
        h = self.encoder(h)                                   # [B, L, d_model]
        h = self.norm(h)
        h = h.mean(dim=1)                                     # 简单池化；可换注意力池化/CLS

        # 条件拼接
        road_vec = self.road_emb(road_type)                   # [B, 4]
        towards_vec = self.towards_emb(turn_direction)        # [B, 4]
        h = torch.cat([h, road_vec, towards_vec], dim=1)      # [B, d_model+8]

        raw = self.fc(h).squeeze(1)                           # 未约束实数
        out_deg = self.max_angle * torch.tanh(raw)            # 限幅到 [-30°, 30°]
        return out_deg
