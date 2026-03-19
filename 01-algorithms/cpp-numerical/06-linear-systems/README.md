# 第6章 线性代数方程组的求解

## 背景
V5 第 6 章覆盖高斯消元、主元策略、带状矩阵、对称正定方程组、迭代法等。代表旧文件包括 `6GAUS.C`、`6CGAS.C`、`6CHLK.C`、`6GMIV.C`。

## 代表算法
这里采用带列交换记录的高斯消元。相比完全照搬旧版裸数组实现，现代 C++ 版本更强调：
- 方阵检查
- 接近奇异矩阵的显式异常
- 向量与矩阵边界安全

## 文件
- `gaussian_elimination.cpp`: 带部分主元的线性方程组求解
- `conjugate_gradient.cpp`: 对称正定线性系统的共轭梯度法
- `tridiagonal_solver.cpp`: 三对角线性系统的 Thomas 算法
- `jacobi_iteration.cpp`: Jacobi 迭代法，对应线性迭代法主线
- `gauss_seidel_sor.cpp`: Gauss-Seidel 与 SOR 统一实现框架
