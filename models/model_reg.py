import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Small modules ----------------

class SEModule(nn.Module):
    """Squeeze-and-Excitation for 1D features: [B, C, L]"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        w = self.pool(x).squeeze(-1)      # [B, C]
        w = self.fc(w).unsqueeze(-1)      # [B, C, 1]
        return x * w


def make_norm_1d(channels: int, norm: str = "group", gn_max_groups: int = 8) -> nn.Module:
    """
    Recommended normalization for small/medium batch sizes:
    - GroupNorm with an adaptive group count.
    """
    if norm == "batch":
        return nn.BatchNorm1d(channels)

    # group norm
    g = min(gn_max_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, channels)


class ResidualConvBlock(nn.Module):
    """
    Residual Conv1D block with circular padding.
    Two conv layers, each supports dilation. Optional SE.
    in/out channels are the same.
    """
    def __init__(
        self,
        channels: int,
        k1: int = 7,
        k2: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_se: bool = True,
        norm: str = "group",
    ):
        super().__init__()
        assert k1 % 2 == 1 and k2 % 2 == 1, "Use odd kernel sizes to keep symmetric padding."

        p1 = ((k1 - 1) // 2) * dilation
        p2 = ((k2 - 1) // 2) * dilation

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=k1,
            padding=p1, dilation=dilation, padding_mode="circular"
        )
        self.norm1 = make_norm_1d(channels, norm=norm)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=k2,
            padding=p2, dilation=dilation, padding_mode="circular"
        )
        self.norm2 = make_norm_1d(channels, norm=norm)

        self.use_se = use_se
        if use_se:
            self.se = SEModule(channels, reduction=8)

        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.norm2(y)

        if self.use_se:
            y = self.se(y)

        y = self.act2(y)
        y = self.drop2(y)
        return y + residual


class ResidualMLP(nn.Module):
    """Residual MLP block for the regression head."""
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y


# ---------------- Main network ----------------

class RegressionNetwork(nn.Module):
    """
    Recommended version:
    - Stem conv uses k=7 (circular) for radar consistency
    - Default stem_pool=False (easier to control RF in degrees)
    - Residual blocks use (k1=7, k2=3) with dilation schedule
    - Radar-aware gate: gate([cond, lidar_feat])
    - FiLM identity init: gamma = 1 + Linear(cond), beta = Linear(cond) init to 0
    - aux_tail branch: dropout (train only) + aux_scale limiter
    """

    def __init__(
        self,
        use_embedding: bool = True,
        n_road: int = 4,
        n_turn: int = 3,
        d_road: int = 4,
        d_turn: int = 4,
        base_channels: int = 32,

        # blocks / receptive field control
        num_blocks: int = 5,
        dilation_schedule=(1, 2, 4, 8, 6),
        block_k1: int = 7,
        block_k2: int = 3,

        # stem
        stem_k: int = 7,
        stem_pool: bool = False,  # recommended: False for RF control

        # regularization / norm
        norm: str = "group",
        dropout: float = 0.2,
        use_se: bool = True,

        # fusion / head
        out_dim: int = 1,
        gate_hidden: int = 64,
        gate_dropout: float = 0.1,
        film_identity: bool = True,

        # aux tail features (NOT road/turn)
        aux_tail: int = 2,          # number of extra continuous features appended to the end of x_lidar
        aux_dropout_p: float = 0.1, # dropout prob applied to aux_tail (train only)
        aux_scale: float = 0.2,     # max contribution strength from aux branch
        print_dilations: bool = False,  # set True if you want to confirm runtime dilations
    ):
        super().__init__()
        self.use_embedding = use_embedding
        self.film_identity = film_identity
        self.aux_tail = aux_tail
        self.aux_scale = aux_scale
        self.print_dilations = print_dilations

        C = base_channels

        # ----- Stem -----
        stem_p = stem_k // 2
        self.stem = nn.Sequential(
            nn.Conv1d(1, C, kernel_size=stem_k, padding=stem_p, padding_mode="circular"),
            make_norm_1d(C, norm=norm),
            nn.GELU(),
            nn.MaxPool1d(2) if stem_pool else nn.Identity(),
        )
        self.stem_pool = stem_pool

        # ----- Residual Conv Blocks with dilation schedule -----
        ds = list(dilation_schedule)
        if len(ds) < num_blocks:
            ds = (ds * ((num_blocks + len(ds) - 1) // len(ds)))[:num_blocks]
        ds = ds[:num_blocks]
        self._dilations_used = ds  # for inspection

        blocks = []
        for d in ds:
            blocks.append(
                ResidualConvBlock(
                    channels=C,
                    k1=block_k1,
                    k2=block_k2,
                    dilation=int(d),
                    dropout=dropout,
                    use_se=use_se,
                    norm=norm,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # ----- Global pooling + projection -----
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(C, 128)
        self.proj_act = nn.GELU()

        # ----- Condition branch (road/turn) -----
        if use_embedding:
            self.emb_road = nn.Embedding(n_road, d_road)
            self.emb_turn = nn.Embedding(n_turn, d_turn)
            cond_in_dim = d_road + d_turn
        else:
            cond_in_dim = 2

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # FiLM
        self.film_gamma = nn.Linear(128, 128)
        self.film_beta  = nn.Linear(128, 128)

        if film_identity:
            # start near identity: gamma ~ 1, beta ~ 0
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        # Radar-aware gate: input = [cond, lidar_feat]
        self.gate = nn.Sequential(
            nn.Linear(128 + 128, gate_hidden),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

        # ----- Aux tail branch (optional) -----
        if aux_tail > 0:
            self.aux_dropout = nn.Dropout(p=aux_dropout_p)
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_tail, 128),
                nn.GELU(),
                nn.Linear(128, 128),
            )
        else:
            self.aux_dropout = None
            self.aux_proj = None

        # ----- Regression head -----
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

    def forward(self, x_lidar: torch.Tensor, road_type: torch.Tensor, turn_direction: torch.Tensor) -> torch.Tensor:
        # x_lidar: [B, L] or [B, 1, L]
        if x_lidar.dim() == 2:
            seq = x_lidar
        elif x_lidar.dim() == 3:
            seq = x_lidar.squeeze(1)
        else:
            raise ValueError("x_lidar must be [B,L] or [B,1,L]")

        # split: first 360 are radar, last aux_tail are extra features (if present)
        L = seq.size(-1)
        aux_raw = None
        if self.aux_tail > 0 and L >= 360 + self.aux_tail:
            aux_raw = seq[:, -self.aux_tail:]   # [B, aux_tail]
            seq = seq[:, :360]                  # [B, 360]
        else:
            seq = seq[:, :360] if L >= 360 else seq  # allow shorter, but your data should be 360

        # ---- Radar backbone ----
        x = seq.unsqueeze(1)     # [B, 1, 360]
        x = self.stem(x)         # [B, C, L'] (L' = 360 if no pool else 180)
        if self.print_dilations and (not hasattr(self, "_printed_once")):
            print("[Model] stem_pool:", self.stem_pool, "| dilations used:", self._dilations_used)
            self._printed_once = True

        x = self.blocks(x)                       # [B, C, L']
        x = self.global_pool(x).squeeze(-1)      # [B, C]
        lidar_feat = self.proj_act(self.proj(x)) # [B, 128]

        # ---- Condition vector ----
        if self.use_embedding:
            road = road_type.long().view(-1)
            turn = turn_direction.long().view(-1)
            cond_raw = torch.cat([self.emb_road(road), self.emb_turn(turn)], dim=1)  # [B, d_road+d_turn]
        else:
            road = road_type.float().view(-1, 1)
            turn = turn_direction.float().view(-1, 1)
            cond_raw = torch.cat([road, turn], dim=1)                                 # [B, 2]

        cond = self.cond_mlp(cond_raw)  # [B, 128]

        # ---- FiLM + radar-aware gate ----
        gamma = self.film_gamma(cond)
        beta  = self.film_beta(cond)
        if self.film_identity:
            gamma = 1.0 + gamma

        mod = gamma * lidar_feat + beta

        gate_in = torch.cat([cond, lidar_feat], dim=1)  # radar-aware
        gate_w = self.gate(gate_in)                     # [B, 1]
        fused = gate_w * mod + (1.0 - gate_w) * lidar_feat

        # ---- Aux tail correction ----
        if (aux_raw is not None) and (self.aux_proj is not None):
            aux = self.aux_dropout(aux_raw)     # train only; disabled in eval()
            aux_feat = self.aux_proj(aux)       # [B, 128]
            fused = fused + self.aux_scale * aux_feat

        # ---- Regression head ----
        y = self.head_pre(fused)
        y = self.head_blocks(y)
        out = self.head_out(y)  # [B, out_dim]
        if out.shape[1] == 1:
            out = out.squeeze(1)
        return out
