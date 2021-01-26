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
@Desc       :   Sec 8.3 自动并行计算
@小结：
1.  MXNet 能够通过自动并行计算提升计算性能
"""
import d2lzh as d2l
import data
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_title, show_figures


# ----------------------------------------------------------------------
def run(x):
    return [nd.dot(x, x) for _ in range(10)]


def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]


def main():
    x_cpu = nd.random.uniform(shape=(2000, 2000))
    x_gpu = nd.random.uniform(shape=(6000, 6000), ctx=mx.gpu(0))
    # ToDo: 为什么预热？
    # 预热开始
    run(x_cpu)
    run(x_gpu)
    nd.waitall()
    # 预热结束

    # parallel_run(x_cpu, x_gpu)
    with d2l.Benchmark("Run on GPU."):
        y = run(x_gpu)
        nd.waitall()
        pass
    with d2l.Benchmark("Then copy to CPU."):
        copy_to_cpu(y)
        nd.waitall()
        pass
    with d2l.Benchmark("Run and copy in parallel."):
        y = run(x_gpu)
        copy_to_cpu(y)
        nd.waitall()
        pass
    # Note: 系统自动实现 计算和通信 的并行操作，时间比单独在 GPU上运算 和 GPU到CPU通信 操作的合计时间短
    pass


def parallel_run(x_cpu, x_gpu):
    with d2l.Benchmark("Run on CPU."):
        run(x_cpu)
        nd.waitall()
        pass
    with d2l.Benchmark("Run on GPU."):
        run(x_gpu)
        nd.waitall()
        pass
    with d2l.Benchmark("Run on both CPU and GPU in parallel."):
        run(x_cpu)
        run(x_gpu)
        nd.waitall()
        pass
    # Note: 系统自动在 CPU 和 GPU 上并行计算，时间比单独在 CPU 和 GPU 上运算的合计时间短


# ----------------------------------------------------------------------


if __name__ == '__main__':
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
