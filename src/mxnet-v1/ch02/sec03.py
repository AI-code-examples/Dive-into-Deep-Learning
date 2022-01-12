# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   Dive-into-Deep-Learning
@File       :   sec0202.py
@Version    :   v0.1
@Time       :   2020-12-27 9:25
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   《动手学深度学习》
@Desc       :   Sec 2.3 自动求梯度的例子
@理解：
"""
from mxnet import nd, autograd
from tools import beep_end, show_subtitle


# ----------------------------------------------------------------------
def autograd_examples(X):
    # 申请梯度需要的内存空间
    X.attach_grad()
    # 建立需要求梯度的函数 $y=2 X^2$
    with autograd.record():
        y = 2 * nd.dot(X.T, X)
    # 基于函数对 X 求梯度 $dy/dx = 4X$
    y.backward()
    # 函数求得的梯度结果 $4X$
    print("X.grad=", X.grad)
    pass


def train_predict(X):
    # ToDo: 训练模式和预测模式？（Ref：3.13）
    print(autograd.is_training())
    with autograd.record():
        print(autograd.is_training())
    pass


def f1(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c


def f2(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b * b
    else:
        c = b * b * b
    return c


def flow_grad(X):
    """
    定义函数 f1(), f2()，然后对函数求导
    注：观察函数可以发现求导是按步倒推的
    """

    show_subtitle("function gradient")
    a = nd.random.normal(shape=1)
    # a = nd.random.normal(shape=(3, 3))
    print("a=", a)
    a.attach_grad()

    show_subtitle("c")
    with autograd.record():
        c = f1(a)
    print("c=", c)
    c.backward()
    print(a.grad)
    print(c / a)

    show_subtitle("d")
    with autograd.record():
        d = f2(a)
    print("d=", d)
    d.backward()
    print(a.grad)
    print(d / a)
    pass


def main():
    X = nd.arange(4).reshape((4, 1))
    print("X=", X)
    autograd_examples(X)
    train_predict(X)
    flow_grad(X)
    pass


# ----------------------------------------------------------------------
# 小结
# -   MXNet 提供了 autograd 模块来自动化求导过程
# -   MXNet 的 autograd 模块可以对一般的命令式程序进行求导
# -   MXNet 的运行模式包括训练模式和预测模式

if __name__ == '__main__':
    # 运行结束的提醒
    main()
    beep_end()
