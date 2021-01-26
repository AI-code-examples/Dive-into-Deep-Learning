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
@Desc       :   Sec 8.2 异步计算
@小结：
1.  MXNet 包括用户直接用来交互的前端和系统用来执行计算的后端
2.  MXNet 能够通过异步计算提升计算性能
3.  建议使用每个小批量训练或者预测时至少使用一个同步函数，从而短时间内将过多计算任务丢给后端
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
import os
import subprocess
import time
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def main():
    a = nd.ones((1, 2))
    b = nd.ones((1, 2))
    c = a * b + 2
    print(c)

    show_subtitle("Asynchronous Computation")
    with Benchmark("Workloads are queued."):
        x = nd.random.uniform(shape=(2000, 2000))
        y = nd.dot(x, x, ).sum()
        pass
    with Benchmark("Workloads are finished."):
        # 当 y 需要输出时才进行计算，才需要消耗 CPU 的资源，才会有时间差
        print("sum =", y)
        pass

    with Benchmark():
        y = nd.dot(x, x)
        show_subtitle("wait_to_read() 等待计算完成")
        y.wait_to_read()
        pass
    with Benchmark():
        y = nd.dot(x, x)
        z = nd.dot(x, x)
        show_subtitle("waitall() 等待所有计算完成")
        nd.waitall()
        pass
    with Benchmark():
        y = nd.dot(x, x)
        show_subtitle("asnumpy() 不支持异步操作，系统需要等待其完成")
        y.asnumpy()
        pass
    with Benchmark():
        y = nd.dot(x, x)
        show_subtitle("asscalar() 不支持异步操作，系统需要等待其完成")
        y.norm().asscalar()
        pass
    show_subtitle("synchronous vs. asynchronous")
    # ToDo: 同步比异步速度更快？
    # 1000(t1+t2+t3)>t1+1000t2+t3
    with Benchmark("synchronous."):
        for _ in range(1000):
            y = x + 1
            y.wait_to_read()
            pass
        pass
    with Benchmark('asynchronous.'):
        for _ in range(1000):
            y = x + 1
            pass
        nd.waitall()
        pass

    def data_iter():
        start = time.time()
        num_batches, batch_size = 100, 1024
        for i in range(num_batches):
            X = nd.random.normal(shape=(batch_size, 512))
            y = nd.ones((batch_size,))
            yield X, y
            if (i + 1) % 50 == 0:
                print("batch %d, time %f sec" % (i + 1, time.time() - start))
                pass
            pass
        pass

    net = nn.Sequential()
    net.add(nn.Dense(2048, activation='relu'),
            nn.Dense(512, activation='relu'),
            nn.Dense(1))
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})
    loss = gloss.L2Loss()

    def get_mem_linux():
        # os.getpid() 只能在 Linux 或者 MacOS 上运行
        res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
        return int(str(res).split()[15]) / 1e3

    def get_mem():
        return 0

    for X, y in data_iter():
        break
        pass
    loss(y, net(X)).wait_to_read()
    l_sum, mem = 0, get_mem()
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
            pass
        l_sum += l.mean().asscalar()
        l.backward()
        trainer.step(X.shape[0])
        pass
    nd.waitall()
    print("increased memory: %f MB" % (get_mem() - mem))

    mem = get_mem()
    for X, y in data_iter():
        with autograd.record():
            l=loss(y,net(X))
            pass
        l.backward()
        trainer.step(X.shape[0])
        pass
    nd.waitall()
    print("increased memory: %f MB" % (get_mem() - mem))
    pass


class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''
        pass

    def __enter__(self):
        self.start = time.time()
        pass

    def __exit__(self, *args):
        print("%stime: %.4f sec" % (self.prefix, time.time() - self.start))
        pass


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
