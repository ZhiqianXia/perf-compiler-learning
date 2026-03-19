# 第14章 特殊函数的计算

## 背景
V5 第 14 章覆盖 Gamma、Beta、不完全 Beta、误差函数、Bessel、Legendre 等。代表旧文件包括 `14BETA.C`、`14BSL1.C`、`14GAM1.C`、`14CHII.C`。

## 代表算法
这里保留两条主线：
- 不完全 Beta 函数的思路在 README 中说明
- 代码示例用第一类 Bessel 函数 $J_n(x)$ 的递推计算，对应旧版 `14BSL1.C`

## 背景知识
Bessel 函数在圆柱坐标分离变量、波动方程和频域问题里非常常见。计算时常结合：
- 低阶近似
- 向前递推
- 向后递推和归一化

## 文件
- `bessel_j.cpp`: 用级数展开计算 $J_0, J_1$，再递推得到 $J_n$
- `incomplete_beta.cpp`: 用连分式实现正则化不完全 Beta 函数，对应 `14BETA.C`
- `gamma_lanczos.cpp`: Lanczos 近似实现 Gamma 函数
- `error_function.cpp`: 误差函数 `erf(x)` 的经典近似公式
- `legendre_polynomials.cpp`: Legendre 多项式递推
- `normal_cdf.cpp`: 通过 `erf` 近似实现正态分布函数
