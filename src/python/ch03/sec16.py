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
@Desc       :   Sec 3.16 Kaggle 比赛：房价预测
@小结：
"""
import d2lzh as d2l
import numpy as np
import pandas as pd

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
from tools import beep_end, show_subtitle, show_figures, get_root_path, show_title


# ----------------------------------------------------------------------
def main():
    show_subtitle("3.16.2 读取数据集")
    # 训练数据集包括：1460个样本，80个特征，1个标签（SalePrice）
    root_path = get_root_path()
    # 测试数据集包括：1459个样本，80个特征，没有标签
    train_data = pd.read_csv(root_path + "data/kaggle_house_pred_train.csv")
    test_data = pd.read_csv(root_path + "data/kaggle_house_pred_test.csv")
    # 训练特征集包括：79个特征（去除 Id 特征，去除 SalePrice 标签）
    train_features = train_data.iloc[:, 1:-1]  # 剔除 -1 是为了将训练数据中的标签摘除
    # 测试特征集包括：79个特征（去除 Id 特征）
    test_features = test_data.iloc[:, 1:]
    all_features = pd.concat((train_features, test_features))
    # output_dataset(test_data, train_data)

    show_subtitle("3.16.3 预处理数据集")
    # 先将特征值标准化，标准化后均值为0，因此将缺失值使用0替换
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 将离散数值转换成指示特征，特征数从 79 个增加到 331 个
    # dummy_na=True 将缺失值也当作合法的特征值并且为其创建指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0]
    train_features = nd.array(all_features[:n_train].values)
    test_features = nd.array(all_features[n_train:].values)
    train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

    show_subtitle("3.16.4 训练模型")
    # train()

    show_subtitle("3.16.5 K 折交叉验证")
    # k_fold()

    show_subtitle("3.16.6 模型选择")
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print("%d-fold validation: avg train rmse %f, avg valid rmse %f" % (k, train_l, valid_l))

    # show_subtitle("3.16.7 训练、预测、提交")
    # train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    pass


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print("train rmse %f" % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    pass

def get_net():
    # 线性回归模型
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    return net


def log_rmse(net, features, labels):
    # 对数均方根误差函数 $\sqrt{1/n \sum_{i=1}^n (\log(y_i)-\log(\hat_{y}_i))^2}$
    # 将小于1的值设成1，使得取对数时数值更加稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 使用 Adam 优化算法，避免学习率过于敏感的 sgd 算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
                pass
            l.backward()
            trainer.step(batch_size)
            pass
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
            pass
        pass
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == 1:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
            pass
        pass
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    # train_l_sum(train loss sum)
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # train_ls(train loss single)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            # d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
            #              range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
            pass
        # -1 表示最后一次训练的损失
        print("fold %d, train rmse %f, valid rmse %f" % (i, train_ls[-1], valid_ls[-1]))
        pass
    return train_l_sum / k, valid_l_sum / k

def output_dataset(test_data, train_data):
    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


# ----------------------------------------------------------------------


if __name__ == '__main__':
    show_title("program begin")
    loss = gloss.L2Loss()
    main()
    # 运行结束的提醒
    beep_end()
    show_figures()
