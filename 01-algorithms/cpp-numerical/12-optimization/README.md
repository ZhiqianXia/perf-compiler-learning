# 第12章 极值问题的求解

## 背景
V5 第 12 章包含一维搜索、线性规划单纯形法以及无约束优化。代表旧文件有 `12MAX1.C`、`12MAXN.C`、`12JSIM.C`、`12LPLQ.C`。

## 代表算法
这里直接对应旧版 `12JSIM.C`，给出 Nelder-Mead 单纯形法。它不需要显式梯度，适合黑盒目标函数、噪声较大的工程调参场景。

## 背景知识
Nelder-Mead 的核心操作是：
- reflection
- expansion
- contraction
- shrink

它本质上是在函数值排序后的单纯形上做几何更新，而不是沿梯度方向前进。

## 文件
- `nelder_mead.cpp`: 二维示例上的 Nelder-Mead 极小化
- `golden_section_search.cpp`: 一维无导数搜索，对应第 12 章一维极值问题主线
