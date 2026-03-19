# 第2章 复数运算

## 背景
V5 第 2 章给出了复数的四则运算、幂、对数、三角函数与指数函数。代表旧文件有 `2CEXP.C`、`2CLOG.C`、`2CMUL.C`、`2CDIV.C`。

## 代表算法
这里以复指数 `exp(x + i y) = exp(x) (cos y + i sin y)` 为入口。这个公式在谱方法、常微分方程解析解和傅里叶分析中都很常见。

## 数值要点
- 实部过大时 `exp(x)` 可能上溢
- 使用标准库 `std::complex` 可避免手写实虚部接口错误
- 对教学场景，先展示解析公式，再给出标准库结果对比最清楚

## 文件
- `complex_exp.cpp`: 用解析公式和 `std::complex` 双重实现复指数
- `complex_log.cpp`: 对应 `2CLOG.C` 的复对数主值实现
- `complex_division.cpp`: 对应 `2CDIV.C` 的稳定复除法
