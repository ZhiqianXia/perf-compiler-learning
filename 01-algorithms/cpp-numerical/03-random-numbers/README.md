# 第3章 随机数的产生

## 背景
V5 第 3 章包含均匀分布随机数、随机数序列以及正态分布随机数生成。代表旧文件包括 `3RND1.C`、`3RNDS.C`、`3GRN1.C`、`3GRNS.C`。

## 代表算法
这里采用 Box-Muller 变换，从两个独立均匀分布随机数构造两个独立标准正态随机数。

## 背景知识
若 $U_1, U_2 \sim \mathrm{Uniform}(0,1)$，则
$$
Z_1 = \sqrt{-2 \ln U_1} \cos(2\pi U_2), \quad
Z_2 = \sqrt{-2 \ln U_1} \sin(2\pi U_2)
$$
服从标准正态分布。

## 文件
- `box_muller.cpp`: Box-Muller 正态随机数发生器
- `linear_congruential_generator.cpp`: 对应 `3RND1.C` 的线性同余均匀随机数
- `normal_statistics.cpp`: 对正态样本做均值和方差统计检查
