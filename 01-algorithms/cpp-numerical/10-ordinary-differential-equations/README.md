# 第10章 常微分方程组的求解

## 背景
V5 第 10 章覆盖 Euler 改进法、Runge-Kutta、Adams、Gear 等。代表旧文件有 `10ELR1.C`、`10RKT1.C`、`10ADMS.C`、`10GEAR.C`。

## 代表算法
本章保留两个层次：
- 改进 Euler 法：理解一步法最直接
- Adams-Bashforth-Moulton 四阶预测校正：对应旧版 `10ADMS.C` 的主线思想

## 数值背景
- 一步法容易实现，但精度和稳定域有限
- 多步法在平滑问题上效率更高，但需要起步器
- 预测校正法本质上把显式预测和隐式修正结合起来

## 文件
- `ode_methods.cpp`: 改进 Euler 与四阶 Adams-Bashforth-Moulton
- `runge_kutta4.cpp`: 经典四阶 Runge-Kutta，一步法中的基准方法
- `backward_euler.cpp`: 隐式 Euler，对应刚性方程场景的最基础隐式法
- `adams_bashforth4.cpp`: 四阶 Adams-Bashforth 显式多步法
