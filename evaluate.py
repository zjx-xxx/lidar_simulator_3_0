import torch
import pandas as pd
from models.model import NeuralNetwork

def evaluate(model, X_test, y_test, device):
    # 转换为torch张量并移到设备上
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        correct = (predicted == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy


if __name__ == '__main__':
    # 确定设备是GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取X_test 和 Y_test
    X_test = pd.read_csv('./mydata/X_test.csv', header=None).values  # 从CSV文件读取X_test
    y_test = pd.read_csv('./mydata/type/Y_test.csv', header=None).values.flatten()  # 从CSV文件读取Y_test

    # 初始化模型
    model = NeuralNetwork()

    # 将模型移到设备上
    model.to(device)

    # 加载已训练好的模型
    try:
        model.load_state_dict(torch.load('./model/model'))
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit()

    # 评估模型
    evaluate(model, X_test, y_test, device)
