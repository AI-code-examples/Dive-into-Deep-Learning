
# GoogLeNet

## 结论

data_size 变小时，模型会出现欠拟合，说明图片变小后有部分信息丢失。

## 实验数据

### data_size=96

-   epoch 1, loss 1.7543, train acc 0.337, test acc 0.669, time 57.1 sec
-   epoch 2, loss 0.6899, train acc 0.737, test acc 0.800, time 50.4 sec
-   epoch 3, loss 0.4705, train acc 0.823, test acc 0.856, time 50.7 sec
-   epoch 4, loss 0.3848, train acc 0.854, test acc 0.850, time 50.8 sec
-   epoch 5, loss 0.3423, train acc 0.871, test acc 0.876, time 50.8 sec
-   epoch 6, loss 0.3136, train acc 0.881, test acc 0.891, time 50.8 sec
-   epoch 7, loss 0.3252, train acc 0.878, test acc 0.897, time 51.1 sec
-   epoch 8, loss 0.2764, train acc 0.895, test acc 0.898, time 51.0 sec
-   epoch 9, loss 0.2597, train acc 0.901, test acc 0.900, time 51.2 sec
-   epoch 10, loss 0.2438, train acc 0.909, test acc 0.905, time 51.4 sec
-   epoch 11, loss 0.2307, train acc 0.913, test acc 0.912, time 51.7 sec
-   epoch 12, loss 0.2176, train acc 0.918, test acc 0.903, time 51.2 sec
-   epoch 13, loss 0.2076, train acc 0.922, test acc 0.913, time 51.0 sec
-   epoch 14, loss 0.1953, train acc 0.927, test acc 0.912, time 50.8 sec
-   epoch 15, loss 0.1894, train acc 0.929, test acc 0.912, time 51.0 sec
-   epoch 16, loss 0.1761, train acc 0.934, test acc 0.916, time 51.0 sec
-   epoch 17, loss 0.1661, train acc 0.936, test acc 0.917, time 50.9 sec
-   epoch 18, loss 0.1575, train acc 0.941, test acc 0.918, time 50.9 sec
-   epoch 19, loss 0.3732, train acc 0.865, test acc 0.587, time 50.4 sec
-   epoch 20, loss 0.9272, train acc 0.646, test acc 0.868, time 50.7 sec
-   epoch 21, loss 0.2879, train acc 0.892, test acc 0.898, time 50.8 sec
-   epoch 22, loss 0.2090, train acc 0.921, test acc 0.912, time 50.8 sec
-   epoch 23, loss 0.1733, train acc 0.934, test acc 0.919, time 50.9 sec
-   epoch 24, loss 0.5370, train acc 0.798, test acc 0.564, time 50.9 sec
-   epoch 25, loss 0.4664, train acc 0.824, test acc 0.890, time 50.8 sec

### data_size=224

-   epoch 1, loss 1.9818, train acc 0.259, test acc 0.503, time 348.7 sec
-   epoch 2, loss 0.9020, train acc 0.642, test acc 0.418, time 341.5 sec
-   epoch 3, loss 0.6288, train acc 0.763, test acc 0.831, time 339.6 sec
-   epoch 4, loss 0.4446, train acc 0.833, test acc 0.848, time 333.0 sec
-   epoch 5, loss 0.3881, train acc 0.855, test acc 0.870, time 338.5 sec
-   epoch 6, loss 0.3450, train acc 0.869, test acc 0.884, time 328.2 sec
-   epoch 7, loss 0.3174, train acc 0.881, test acc 0.882, time 327.2 sec
-   epoch 8, loss 0.2946, train acc 0.890, test acc 0.887, time 326.1 sec
-   epoch 9, loss 0.2717, train acc 0.897, test acc 0.901, time 335.3 sec
-   epoch 10, loss 0.2563, train acc 0.904, test acc 0.905, time 337.0 sec
-   epoch 11, loss 0.2439, train acc 0.908, test acc 0.904, time 335.9 sec
-   epoch 12, loss 0.2299, train acc 0.915, test acc 0.915, time 332.5 sec
-   epoch 13, loss 0.2150, train acc 0.920, test acc 0.899, time 336.5 sec
-   epoch 14, loss 0.2030, train acc 0.924, test acc 0.916, time 339.7 sec
-   epoch 15, loss 0.1921, train acc 0.929, test acc 0.921, time 336.0 sec
-   epoch 16, loss 0.1830, train acc 0.932, test acc 0.923, time 336.0 sec
-   epoch 17, loss 0.7943, train acc 0.720, test acc 0.480, time 336.5 sec
-   epoch 18, loss 0.7100, train acc 0.720, test acc 0.827, time 337.2 sec
-   epoch 19, loss 0.3527, train acc 0.867, test acc 0.888, time 340.4 sec
-   epoch 20, loss 0.2611, train acc 0.902, test acc 0.902, time 338.4 sec
-   epoch 21, loss 0.2260, train acc 0.915, test acc 0.911, time 335.8 sec
-   epoch 22, loss 0.1992, train acc 0.925, test acc 0.918, time 341.6 sec
-   epoch 23, loss 0.1827, train acc 0.932, test acc 0.919, time 341.9 sec
-   epoch 24, loss 0.1684, train acc 0.937, test acc 0.914, time 343.7 sec
-   epoch 25, loss 0.1549, train acc 0.941, test acc 0.921, time 343.2 sec
