# 第7章 非线性方程与方程组

## 背景
V5 第 7 章包含不动点迭代、Aitken 加速、牛顿法、割线法、方程组求根。代表旧文件有 `7ATKN.C`、`7CSRT.C`、`7CMTC.C`。

## 代表算法
这里选择 Aitken 的 $\Delta^2$ 加速法。它对线性收敛的不动点迭代非常实用，能显著压缩迭代步数。

## 背景知识
若原始迭代为 $x_{k+1} = g(x_k)$，则 Aitken 加速给出
$$
\hat{x}_k = x_k - \frac{(x_{k+1} - x_k)^2}{x_{k+2} - 2 x_{k+1} + x_k}
$$
分母接近 0 时必须回退，不能硬算。

## 文件
- `aitken_fixed_point.cpp`: 带保护项的 Aitken 加速不动点迭代
- `newton_secant.cpp`: Newton 法与割线法的并列对照实现
