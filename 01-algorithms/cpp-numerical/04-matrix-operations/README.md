# 第4章 矩阵运算

## 背景
V5 第 4 章覆盖实矩阵与复矩阵运算、求逆、LU、QR、Cholesky 等。代表旧文件包括 `4CHOL.C`、`4LLUU.C`、`4MAQR.C`、`4GINV.C`。

## 代表算法
这里选用 Cholesky 分解，因为它是对称正定矩阵最重要的直接法之一。若 $A$ 对称正定，则可写成
$$
A = L L^T
$$
其中 $L$ 为下三角矩阵。

## 数值意义
- 比通用 LU 更省一半存储与计算
- 适用于最小二乘正规方程、协方差矩阵、核方法
- 前提是矩阵必须对称正定

## 文件
- `cholesky.cpp`: 现代 C++ 版本的 Cholesky 分解与重构校验
- `qr_decomposition.cpp`: 对应 `4MAQR.C` 的改进 Gram-Schmidt QR 分解
- `lu_decomposition.cpp`: 对应 `4LLUU.C` 的 Doolittle LU 分解
- `matrix_inverse.cpp`: 对应 `4GINV.C` 的 Gauss-Jordan 求逆
