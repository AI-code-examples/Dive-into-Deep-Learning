"""
=================================================
@path   : Dive-into-Deep-Learning -> sec0302.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-23 15:22
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   3.3.线性回归的简洁实现
@小结：
-   使用 torch 可以更加简洁地实现模型
-   在 torch 中
    -   data 模块提供数据集工具
    -   nn 模块定义了神经网络的层和损失函数
==================================================
"""
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

from tools import beep_end


# ----------------------------------------------------------------------
def main():
    num_examples = 1000
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, num_examples)
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    net = define_model()  # 定义模型
    loss = define_loss()  # 定义损失
    trainer = define_trainer(net)  # 定义训练器
    # 训练模型
    train_model(data_iter, features, labels, loss, net, trainer)
    w, b = net[0].weight.data, net[0].bias.data
    print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差：{true_w - b}')
    pass


def train_model(data_iter, features, labels, loss, net, trainer):
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch}, loss {l:f}")


def define_trainer(net):
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    return trainer


def define_loss():
    loss = nn.MSELoss()
    return loss


def define_model():
    net = nn.Sequential(nn.Linear(2, 1))
    return net


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch 的数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
