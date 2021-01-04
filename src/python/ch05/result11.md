# ResNet

## 结论

data_size 变小时，模型会出现欠拟合，说明图片变小后有部分信息丢失。

## 实验数据

### data_size=96

-   epoch 1, loss 0.4178, train acc 0.852, test acc 0.884, time 45.0 sec
-   epoch 2, loss 0.2376, train acc 0.913, test acc 0.913, time 43.9 sec
-   epoch 3, loss 0.1782, train acc 0.934, test acc 0.924, time 44.4 sec
-   epoch 4, loss 0.1350, train acc 0.951, test acc 0.914, time 43.7 sec
-   epoch 5, loss 0.1027, train acc 0.963, test acc 0.925, time 43.9 sec
-   epoch 6, loss 0.0749, train acc 0.973, test acc 0.922, time 44.4 sec
-   epoch 7, loss 0.0553, train acc 0.980, test acc 0.893, time 44.1 sec
-   epoch 8, loss 0.0395, train acc 0.986, test acc 0.924, time 44.2 sec
-   epoch 9, loss 0.0293, train acc 0.990, test acc 0.928, time 44.6 sec
-   epoch 10, loss 0.0193, train acc 0.994, test acc 0.925, time 44.5 sec
-   epoch 11, loss 0.0148, train acc 0.995, test acc 0.926, time 44.4 sec
-   epoch 12, loss 0.0086, train acc 0.997, test acc 0.914, time 44.5 sec
-   epoch 13, loss 0.0078, train acc 0.998, test acc 0.927, time 44.7 sec
-   epoch 14, loss 0.0029, train acc 0.999, test acc 0.933, time 44.1 sec
-   epoch 15, loss 0.0015, train acc 1.000, test acc 0.936, time 44.5 sec

### data_size=224

-   epoch 1, loss 0.4752, train acc 0.831, test acc 0.893, time 238.2 sec
-   epoch 2, loss 0.2497, train acc 0.909, test acc 0.888, time 237.7 sec
-   epoch 3, loss 0.1947, train acc 0.930, test acc 0.918, time 235.8 sec
-   epoch 4, loss 0.1546, train acc 0.944, test acc 0.916, time 241.1 sec
-   epoch 5, loss 0.1185, train acc 0.957, test acc 0.914, time 241.7 sec
-   epoch 6, loss 0.0840, train acc 0.970, test acc 0.926, time 237.3 sec
-   epoch 7, loss 0.0525, train acc 0.982, test acc 0.925, time 239.4 sec
-   epoch 8, loss 0.0344, train acc 0.989, test acc 0.931, time 239.7 sec
-   epoch 9, loss 0.0209, train acc 0.994, test acc 0.927, time 239.1 sec
-   epoch 10, loss 0.0070, train acc 0.999, test acc 0.935, time 239.2 sec
-   epoch 11, loss 0.0022, train acc 1.000, test acc 0.940, time 239.6 sec
-   epoch 12, loss 0.0010, train acc 1.000, test acc 0.940, time 239.5 sec
-   epoch 13, loss 0.0006, train acc 1.000, test acc 0.941, time 240.9 sec
-   epoch 14, loss 0.0004, train acc 1.000, test acc 0.941, time 238.5 sec
-   epoch 15, loss 0.0003, train acc 1.000, test acc 0.941, time 242.4 sec
