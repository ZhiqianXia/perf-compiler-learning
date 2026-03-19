# 第8章 插值与逼近

## 背景
V5 第 8 章包括 Lagrange 插值、Hermite 插值、样条、连分式逼近和最小二乘拟合。代表旧文件有 `8LGR.C`、`8LG3.C`、`8HMT.C`、`8ATK.C`。

## 代表算法
这里先保留“局部 Lagrange 三点插值”的思路。旧版 `8LG3.C` 就是围绕目标点选最近的 3 个节点，做低阶局部插值，优先追求局部稳定性而不是全局高次多项式。

## 数值背景
- 全局高次插值容易出现 Runge 现象
- 局部低阶插值在工程表格查值中更稳健
- 当节点很多时，二分查找 + 局部模板是常见做法

## 文件
- `local_lagrange.cpp`: 最近邻三点局部 Lagrange 插值
- `neville_interpolation.cpp`: Neville 递推插值表，对应表格型插值实现
- `hermite_interpolation.cpp`: 对应 `8HMT.C` 的 Hermite 插值
- `natural_cubic_spline.cpp`: 对应样条插值主线的自然三次样条
