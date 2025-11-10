# predict.py

import torch
import numpy as np
from models.model import NeuralNetwork

def predict(model, X_new):
    # 转换为torch张量
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        prediction = model(X_new_tensor)
        _, predicted_class = torch.max(prediction, 1)  # 获取预测的类别
        return predicted_class.item()

if __name__ == '__main__':
    # 假设你有新的输入数据 X_new
    X_new = np.random.rand(1, 360)  # 1个样本，360维特征
    print(X_new)

    # 初始化模型
    model = NeuralNetwork()

    # 加载已训练好的模型
    model.load_state_dict(torch.load('./model/model'))

    # 进行预测
    predicted_class = predict(model, X_new)
    print(f'Predicted class: {predicted_class}')
