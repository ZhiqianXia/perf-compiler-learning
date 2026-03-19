# 第5章 矩阵特征值与特征向量

## 背景
V5 第 5 章包含对称矩阵 Jacobi 变换、Hessenberg 约化、QR 迭代等。代表旧文件包括 `5JCBI.C`、`5HHBG.C`、`5HHQR.C`。

## 代表算法
这里放入对称矩阵的 Jacobi 特征值算法。它不断用平面旋转消去非对角元，最终把矩阵逼近为对角阵，对角线元素就是特征值。

## 为什么这样选
- 教学可读性明显高于完整 Hessenberg + QR 链路
- 能很好解释“相似变换不改变特征值”
- 与第 4 章的正交分解内容自然衔接

## 文件
- `jacobi_eigen.cpp`: 对称矩阵 Jacobi 迭代求特征值
- `power_iteration.cpp`: 主特征值与主特征向量的幂法
- `qr_iteration.cpp`: 对称矩阵 QR 迭代求特征值近似
- `hessenberg_reduction.cpp`: 对应 `5HHBG.C` 的 Hessenberg 约化
- `inverse_iteration.cpp`: 给定 shift 的反幂法，用于逼近目标特征向量
