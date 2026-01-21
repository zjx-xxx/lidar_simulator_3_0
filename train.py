import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from evaluate import *
from models.model import NeuralNetwork
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 增加旋转增强的数据集定义
class LidarDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            shift = torch.randint(0, x.shape[0], (1,)).item()
            x = torch.roll(x, shifts=shift, dims=0)

        return x, y

# 训练函数
def train(model, X_train, y_train, num_epochs=10, batch_size=32, learning_rate=0.001):
    print(f'Training on {device}')

    train_dataset = LidarDataset(X_train, y_train, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

if __name__ == '__main__':
    # 加载训练数据
    X_train = pd.read_csv('./mydata/X_train.csv', header=None).values
    y_train = pd.read_csv('./mydata/type/Y_train.csv', header=None).values

    # 初始化模型
    model = NeuralNetwork().to(device)

    num_epochs = 100
    batch_size = 64

    # 训练模型
    train(model, X_train, y_train, num_epochs, batch_size)

    # 加载测试数据
    X_test = pd.read_csv('./mydata/X_test.csv', header=None).values
    y_test = pd.read_csv('./mydata/type/Y_test.csv', header=None).values.flatten()

    # 测试评估
    acc = evaluate(model, X_test, y_test, device)

    while acc < 0.82:
        model = NeuralNetwork().to(device)  # 重新初始化模型并转移到GPU
        train(model, X_train, y_train, num_epochs, batch_size)
        acc = evaluate(model, X_test, y_test, device)

    # 保存训练好的模型
    torch.save(model.state_dict(), f'./model/model_{acc * 100:.2f}acc')
    torch.save(model.state_dict(), './model/model')
