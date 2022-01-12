
## 结论

更多卷积层可以避免模型欠拟合，即精度无法进一步提升。

## 实验数据

nin_block 中有二个 1x1 卷积层

-   epoch 1, loss 2.1149, train acc 0.211, test acc 0.338, time 74.2 sec
-   epoch 2, loss 1.1991, train acc 0.555, test acc 0.661, time 71.1 sec
-   epoch 3, loss 0.9088, train acc 0.659, test acc 0.713, time 71.3 sec
-   epoch 4, loss 0.7648, train acc 0.712, test acc 0.831, time 71.3 sec
-   epoch 5, loss 1.4234, train acc 0.502, test acc 0.725, time 71.4 sec
-   epoch 6, loss 0.6449, train acc 0.765, test acc 0.819, time 71.2 sec
-   epoch 7, loss 0.4919, train acc 0.820, test acc 0.840, time 71.3 sec
-   epoch 8, loss 0.4398, train acc 0.837, test acc 0.853, time 71.4 sec
-   epoch 9, loss 0.4007, train acc 0.851, test acc 0.867, time 71.2 sec
-   epoch 10, loss 0.3752, train acc 0.861, test acc 0.876, time 71.3 sec
-   epoch 11, loss 0.3576, train acc 0.868, test acc 0.884, time 71.3 sec
-   epoch 12, loss 0.3503, train acc 0.871, test acc 0.886, time 71.2 sec
-   epoch 13, loss 0.3263, train acc 0.879, test acc 0.891, time 71.3 sec
-   epoch 14, loss 0.3150, train acc 0.884, test acc 0.896, time 71.8 sec
-   epoch 15, loss 0.3036, train acc 0.888, test acc 0.898, time 71.6 sec
-   epoch 16, loss 0.2964, train acc 0.890, test acc 0.895, time 72.1 sec
-   epoch 17, loss 0.2861, train acc 0.894, test acc 0.902, time 72.0 sec
-   epoch 18, loss 0.2805, train acc 0.896, test acc 0.904, time 71.5 sec
-   epoch 19, loss 0.2732, train acc 0.899, test acc 0.909, time 71.3 sec
-   epoch 20, loss 0.2625, train acc 0.902, test acc 0.910, time 71.3 sec
-   epoch 21, loss 0.2588, train acc 0.904, test acc 0.910, time 71.3 sec
-   epoch 22, loss 0.2541, train acc 0.906, test acc 0.905, time 71.4 sec
-   epoch 23, loss 0.2479, train acc 0.909, test acc 0.913, time 71.3 sec
-   epoch 24, loss 0.2439, train acc 0.911, test acc 0.901, time 71.3 sec
-   epoch 25, loss 0.2382, train acc 0.912, test acc 0.915, time 71.2 sec

nin_block 中有一个 1x1 卷积层

-   epoch 1, loss 1.8533, train acc 0.354, test acc 0.625, time 61.3 sec
-   epoch 2, loss 0.7738, train acc 0.717, test acc 0.770, time 58.2 sec
-   epoch 3, loss 0.6299, train acc 0.775, test acc 0.808, time 57.3 sec
-   epoch 4, loss 0.5029, train acc 0.818, test acc 0.839, time 57.7 sec
-   epoch 5, loss 0.4434, train acc 0.839, test acc 0.866, time 58.1 sec
-   epoch 6, loss 0.4131, train acc 0.850, test acc 0.858, time 57.5 sec
-   epoch 7, loss 0.3858, train acc 0.859, test acc 0.878, time 57.5 sec
-   epoch 8, loss 0.3674, train acc 0.866, test acc 0.882, time 57.4 sec
-   epoch 9, loss 0.3513, train acc 0.873, test acc 0.886, time 57.4 sec
-   epoch 10, loss 0.3368, train acc 0.877, test acc 0.884, time 57.7 sec
-   epoch 11, loss 0.3219, train acc 0.882, test acc 0.892, time 57.4 sec
-   epoch 12, loss 0.3118, train acc 0.886, test acc 0.895, time 57.3 sec
-   epoch 13, loss 0.3019, train acc 0.890, test acc 0.896, time 57.3 sec
-   epoch 14, loss 0.2920, train acc 0.893, test acc 0.899, time 57.3 sec
-   epoch 15, loss 0.2856, train acc 0.896, test acc 0.877, time 57.9 sec
-   epoch 16, loss 0.2758, train acc 0.900, test acc 0.902, time 58.6 sec
-   epoch 17, loss 0.2689, train acc 0.900, test acc 0.908, time 58.3 sec
-   epoch 18, loss 0.2631, train acc 0.904, test acc 0.906, time 58.5 sec
-   epoch 19, loss 0.2570, train acc 0.905, test acc 0.910, time 58.2 sec
-   epoch 20, loss 0.2514, train acc 0.908, test acc 0.905, time 58.3 sec
-   epoch 21, loss 0.2470, train acc 0.909, test acc 0.910, time 58.3 sec
-   epoch 22, loss 0.2414, train acc 0.911, test acc 0.910, time 58.0 sec
-   epoch 23, loss 0.2357, train acc 0.913, test acc 0.904, time 58.3 sec
-   epoch 24, loss 0.2301, train acc 0.915, test acc 0.917, time 58.1 sec
-   epoch 25, loss 0.2247, train acc 0.917, test acc 0.905, time 58.2 sec
