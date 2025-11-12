# =========================================================
# train_reg.py
# 适配新版 RegressionNetwork（Conv1d + FiLM + 门控），单输出角度（度）
# 损失：环形误差 + Huber（Smooth L1）
# 评估：MAE/Hit@3° 使用环形误差
# 使用方式：与原脚本基本一致
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_reg import RegressionNetwork  # 你提供的新版

# =========================
# 设备与随机种子
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(2025)


# =========================
# 环形辅助
# =========================
def ang_wrap_deg(delta):
    # 把任意角度差映射到 [-180, 180)
    return (delta + 180.0) % 360.0 - 180.0


# =========================
# Huber（Smooth L1）
# =========================
def huber_loss(x, delta=5.0):
    # x 的单位是“度”，delta 推荐 3~10°
    absx = torch.abs(x)
    quad = 0.5 * x ** 2
    lin = delta * (absx - 0.5 * delta)
    return torch.where(absx <= delta, quad, lin)


# =========================
# 环形 + Huber（带样本权重，与原权重策略兼容）
# =========================
def wrapped_huber_angle_loss(pred_deg, target_deg, base_weight=1.0, angle_weight=10.0, delta=5.0):
    """
    pred_deg: [B] 或 [B,1] 预测角度（度）
    target_deg: [B] 或 [B,1] 目标角度（度）
    """
    if pred_deg.ndim > 1:
        pred_deg = pred_deg.view(-1)
    if target_deg.ndim > 1:
        target_deg = target_deg.view(-1)

    # 环形误差
    ang_err = ang_wrap_deg(pred_deg - target_deg)

    # 样本权重：沿用你原策略（按 |θ| 加权）
    w = base_weight + angle_weight * torch.abs(target_deg) / 30.0

    loss = w * huber_loss(ang_err, delta=delta)
    return loss.mean()


# =========================
# Dataset（兼容离散 embedding 与连续特征两种路径）
# =========================
class LidarRegressionDataset(Dataset):
    def __init__(self, X_main, road_type, turn_direction, y, use_embedding=True):
        self.X_main = torch.tensor(X_main, dtype=torch.float32)
        self.use_embedding = use_embedding

        if use_embedding:
            # 作为类别 id（整数）
            self.road_type = torch.tensor(road_type, dtype=torch.long).view(-1)
            self.turn_direction = torch.tensor(turn_direction, dtype=torch.long).view(-1)
        else:
            # 作为连续特征（浮点）
            self.road_type = torch.tensor(road_type, dtype=torch.float32).view(-1)
            self.turn_direction = torch.tensor(turn_direction, dtype=torch.float32).view(-1)

        self.y = torch.tensor(y, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.X_main)

    def __getitem__(self, idx):
        return (
            self.X_main[idx],
            self.road_type[idx],
            self.turn_direction[idx],
            self.y[idx],
        )


# =========================
# 评估函数（环形 MAE / Hit@3°）
# =========================
@torch.no_grad()
def evaluate(model, dataloader, base_weight=1.0, angle_weight=10.0,
             hit_threshold_deg=3.0, delta=5.0):
    model.eval()
    total_loss, total_mae, total_hit, total_samples = 0.0, 0.0, 0.0, 0

    for x_lidar, road_type, turn_direction, target in dataloader:
        x_lidar = x_lidar.to(device)
        road_type = road_type.to(device)
        turn_direction = turn_direction.to(device)
        target = target.to(device)

        outputs = model(x_lidar, road_type, turn_direction)  # [B]（你的模型里 squeeze 掉了）
        loss = wrapped_huber_angle_loss(outputs, target,
                                        base_weight=base_weight,
                                        angle_weight=angle_weight,
                                        delta=delta)

        # 环形误差 -> MAE / Hit
        if target.ndim > 1:
            target_flat = target.view(-1)
        else:
            target_flat = target
        err = torch.abs(ang_wrap_deg(outputs - target_flat))
        mae = torch.sum(err).item()
        hit = torch.sum((err < hit_threshold_deg).float()).item()

        bs = target_flat.size(0)
        total_loss += loss.item() * bs
        total_mae += mae
        total_hit += hit
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    hit_rate = total_hit / total_samples
    return avg_loss, avg_mae, hit_rate


# =========================
# 训练主循环（接口保持不变）
# =========================
def train(model, train_data, val_data,
          num_epochs=1000, batch_size=64, learning_rate=1e-3,
          early_stop_patience=50,
          base_weight=1.0, angle_weight=10.0, delta=5.0):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_hit3": [], "lr": []}

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        running_loss, seen = 0.0, 0

        for x_lidar, road_type, turn_direction, target in train_loader:
            x_lidar = x_lidar.to(device)
            road_type = road_type.to(device)
            turn_direction = turn_direction.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(x_lidar, road_type, turn_direction)  # [B]
            loss = wrapped_huber_angle_loss(outputs, target,
                                            base_weight=base_weight,
                                            angle_weight=angle_weight,
                                            delta=delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x_lidar.size(0)
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)

        # 验证
        val_loss, val_mae, val_hit3 = evaluate(
            model, val_loader,
            base_weight=base_weight, angle_weight=angle_weight,
            hit_threshold_deg=3.0, delta=delta
        )

        scheduler.step(val_loss)

        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_hit3"].append(val_hit3)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE(deg): {val_mae:.3f} | "
              f"Hit@3°: {val_hit3 * 100:.1f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early Stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience = 0
            os.makedirs('./model', exist_ok=True)
            torch.save(best_state, './model/model_regression_best.pth')
            print(f"✅ Best model updated and saved at epoch {epoch}")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs)")
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model_regression_last.pth')
    print("🎯 Final model saved.")
    return history


# =========================
# 画训练曲线（沿用）
# =========================
def plot_history(history, out_path='./model/training_curves.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Loss vs. Epoch')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_loss.png'));
    plt.close()

    # MAE
    plt.figure()
    plt.plot(epochs, history["val_mae"], label='Val MAE (deg)')
    plt.xlabel('Epoch');
    plt.ylabel('MAE (deg)');
    plt.title('Validation MAE vs. Epoch')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_mae.png'));
    plt.close()

    # Hit@3°
    plt.figure()
    plt.plot(epochs, history["val_hit3"], label='Hit@3°')
    plt.xlabel('Epoch');
    plt.ylabel('Hit Rate');
    plt.title('Hit@3° vs. Epoch')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_hit3.png'));
    plt.close()

    # Learning Rate
    plt.figure()
    plt.plot(epochs, history["lr"], label='Learning Rate')
    plt.xlabel('Epoch');
    plt.ylabel('LR');
    plt.title('Learning Rate vs. Epoch')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_lr.png'));
    plt.close()


# =========================
# 数据读取函数（保持与原版兼容）
# =========================
def load_split(prefix):
    """
    返回：
      X_main: [N, L]，L 默认 360
      road_type: [N,1]  (类别 id 或 连续值)
      turn_direction: [N,1] (类别 id 或 连续值)
      y: [N,1] (角度，单位度)
    """
    X_main = pd.read_csv(f'./mydata/X_{prefix}.csv', header=None).values.astype(np.float32)
    road_type = pd.read_csv(f'./mydata/type/Y_{prefix}.csv', header=None).values
    turn_direction = pd.read_csv(f'./mydata/towards/Y_{prefix}.csv', header=None).values
    y = pd.read_csv(f'./mydata/direction/Y_{prefix}.csv', header=None).values.astype(np.float32)

    if road_type.ndim == 1: road_type = road_type.reshape(-1, 1)
    if turn_direction.ndim == 1: turn_direction = turn_direction.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)

    # 统一标签到 [-180,180)
    y = ((y + 180.0) % 360.0) - 180.0
    return X_main, road_type, turn_direction, y


def _extract_y(sample):
    """
    从样本中鲁棒提取 y：
    - tuple/list: 取最后一个
    - dict: 取 'y' 键或最后一个值
    返回 torch.Tensor (1,) 或标量张量
    """
    if isinstance(sample, dict):
        if 'y' in sample:
            y = sample['y']
        else:
            # 取最后一个键的值
            # 注意：Python 3.7+ 字典有插入序
            y = list(sample.values())[-1]
    elif isinstance(sample, (tuple, list)):
        y = sample[-1]
    else:
        raise TypeError(f"Unsupported sample type for extracting y: {type(sample)}")

    if isinstance(y, torch.Tensor):
        return y.detach().float().reshape(-1)
    else:
        # 可能是 numpy / 标量
        return torch.as_tensor(y, dtype=torch.float32).reshape(-1)


def collect_y_tensor(dataset, batch_size=4096, num_workers=0):
    """
    高效收集整个 dataset 的 y 到一个 1D Tensor。
    不用逐条 __getitem__，而是用 DataLoader 批处理。
    注意：collate_fn 默认即可；我们只从 batch 中抽 y。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    ys = []
    for batch in loader:
        # batch 可能是 tuple/list/dict，与单条一致
        ys.append(_extract_y(batch))
    return torch.cat(ys, dim=0)


def describe_tensor_1d(name, ys: torch.Tensor, bins=10):
    ys = ys[torch.isfinite(ys)]
    if ys.numel() == 0:
        print(f"\n--- {name} Dataset Statistics ---")
        print("Empty or non-finite labels.")
        return
    q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32, device=ys.device)
    qv = torch.quantile(ys, q).cpu().tolist()
    h = torch.histc(ys, bins=bins, min=ys.min(), max=ys.max()).cpu().tolist()
    print(f"\n--- {name} Dataset Statistics ---")
    print(f"Count: {ys.numel()}")
    print(f"Min:   {ys.min().item():.6f}")
    print(f"Max:   {ys.max().item():.6f}")
    print(f"Mean:  {ys.mean().item():.6f}")
    # unbiased=False -> 与 numpy.std(ddof=0) 一致
    print(f"Std:   {ys.std(unbiased=False).item():.6f}")
    print(f"25/50/75% quantiles: [{qv[0]:.6f}, {qv[1]:.6f}, {qv[2]:.6f}]")
    print(f"Histogram ({bins} bins): {h}")


def describe_dataset(name, dataset, bins=10, batch_size=4096, num_workers=0):
    ys = collect_y_tensor(dataset, batch_size=batch_size, num_workers=num_workers)
    describe_tensor_1d(name, ys, bins=bins)


# =========================
# 主入口（与原版相同风格）
# =========================
if __name__ == '__main__':
    X_train, road_train, turn_train, y_train = load_split('train')
    X_val, road_val, turn_val, y_val = load_split('test')

    print('Train shapes:', X_train.shape, road_train.shape, turn_train.shape, y_train.shape)
    print('Val   shapes:', X_val.shape, road_val.shape, turn_val.shape, y_val.shape)

    # 如果 road/turn 是离散类别 id -> 设 True；若为连续特征 -> 设 False
    USE_EMB = True

    train_dataset = LidarRegressionDataset(X_train, road_train, turn_train, y_train, use_embedding=USE_EMB)
    val_dataset = LidarRegressionDataset(X_val, road_val, turn_val, y_val, use_embedding=USE_EMB)

    print(f"Train shapes: {X_train.shape} {y_train.shape} ...")  # 你已有
    print(f"Val   shapes: {X_val.shape} {y_val.shape} ...")

    # 统计 y（自动适配 (x, y) / (x, a, b, y) / dict）
    describe_dataset("Train", train_dataset, bins=10, batch_size=4096, num_workers=0)
    describe_dataset("Val", val_dataset, bins=10, batch_size=4096, num_workers=0)

    # 与你 model_reg.py 的默认参数保持一致（n_road / n_turn 请按你的真实类别数改）
    model = RegressionNetwork(
        use_embedding=USE_EMB,
        n_road=10,  # ★ 按实际类别数调整
        n_turn=5,  # ★ 按实际类别数调整
        # 其余超参保持默认即可
    ).to(device)

    history = train(model, train_dataset, val_dataset,
                    num_epochs=1000,
                    batch_size=64,
                    learning_rate=1e-3,
                    early_stop_patience=100,
                    base_weight=1.0,
                    angle_weight=10.0,
                    delta=5.0)

    plot_history(history, out_path='./model/training_curves.png')
    print('📈 Curves saved to ./model/training_curves_*')

    # # 可选：推理示例
    # with torch.no_grad():
    #     x_lidar, road, turn, gt = val_dataset[0]
    #     pred_deg = model(x_lidar.unsqueeze(0).to(device),
    #                      road.unsqueeze(0).to(device),
    #                      turn.unsqueeze(0).to(device))
    #     pred_deg = RegressionNetwork.vec2angle_deg(pred_deg)  # 规范到 [-180,180)
    #     print("Pred angle (deg):", float(pred_deg.cpu()))
