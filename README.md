# Dive-into-Deep-Learning

学习「动手学深度学习」书的代码和笔记

## Environment

使用项目自带的 `environment.yml` 文件创建环境

```powershell
conda create -f environment.yml
```

如果需要安装 GPU 版本

```powershell
pip uninstall mxnet
pip install mxnet-cu101==1.5.0
conda install cudatoolkit=10.1.243
```

注1：`mxnet-cu101` 表示 CUDA 的版本为 10.1
注2：如果不需要大量计算时，建议不使用 GPU 版本，因为 GPU 版本的启动速度比 CPU 版本的启动速度慢。
补：好像 Conda 安装 CUDA 就可以与 OS 安装的 CUDA 一起使用了，而不需要在系统中再安装 CUDA 的 Toolkit
