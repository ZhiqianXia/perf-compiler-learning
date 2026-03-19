# 第9章 数值积分

## 背景
V5 第 9 章收录复化梯形、复化 Simpson、自适应求积、高斯求积和 Romberg 外推。代表旧文件包括 `9FFTS.C`、`9FPQG.C`、`9GAUS.C`。

## 代表算法
这里选 Romberg 积分。它把复化梯形公式和 Richardson 外推结合起来，能系统地提高收敛阶。

## 背景知识
设 $T(h)$ 为步长为 $h$ 的复化梯形结果，则 Romberg 外推利用误差展开
$$
T(h) = I + c_2 h^2 + c_4 h^4 + \cdots
$$
通过构造外推表逐列消去低阶误差项。

## 文件
- `romberg_integration.cpp`: 用梯形递推和 Richardson 外推实现 Romberg 积分
- `adaptive_simpson.cpp`: 自适应 Simpson 积分，适合局部变化明显的被积函数
- `composite_simpson.cpp`: 复化 Simpson 公式
- `gauss_legendre_quadrature.cpp`: 三点 Gauss-Legendre 求积，对应 `9GAUS.C`
- `composite_trapezoidal.cpp`: 对应 `9FFTS.C` 的复化梯形公式
- `monte_carlo_integration.cpp`: 随机积分视角的简单 Monte Carlo 求积
