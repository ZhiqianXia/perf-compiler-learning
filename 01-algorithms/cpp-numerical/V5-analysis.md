# V5 代码整理分析

## 来源
- 旧代码目录: `/back/XuAlgrithms-master/CPP/V5`
- 代码形态: 16 章、400+ 个 `.C` 文件，典型风格为 K&R C、裸数组、通过输出参数返回结果

## 为什么没有逐文件硬搬
旧代码数量大、接口风格陈旧、许多文件只是同一算法的演示入口或测试包装。直接逐个拷贝会带来三个问题：
- 可读性差，不适合作为学习仓库内容
- 缺少类型安全与边界检查
- 很难和当前仓库的“原理 + 现代实现 + 验证”结构对齐

因此本次整理采用“章节映射 + 代表算法现代 C++ 改写”的方法。

## 章节映射
- 第1章 多项式: `1BPLY.C`, `1CPLY.C`, `1PDIV.C`
- 第2章 复数运算: `2CEXP.C`, `2CLOG.C`, `2CDIV.C`
- 第3章 随机数: `3RND1.C`, `3RNDS.C`, `3GRN1.C`
- 第4章 矩阵运算: `4CHOL.C`, `4LLUU.C`, `4MAQR.C`
- 第5章 特征值/特征向量: `5JCBI.C`, `5HHBG.C`, `5HHQR.C`
- 第6章 线性方程组: `6GAUS.C`, `6CGAS.C`, `6CHLK.C`
- 第7章 非线性方程: `7ATKN.C`, `7CSRT.C`, `7CMTC.C`
- 第8章 插值与逼近: `8LG3.C`, `8LGR.C`, `8HMT.C`, `8ATK.C`
- 第9章 数值积分: `9FFTS.C`, `9FPQG.C`, `9GAUS.C`
- 第10章 常微分方程: `10ELR1.C`, `10ADMS.C`, `10RKT1.C`, `10GEAR.C`
- 第11章 数据处理: `11SQT1.C`, `11SQT2.C`, `11LOG1.C`
- 第12章 优化: `12MAX1.C`, `12MAXN.C`, `12JSIM.C`, `12LPLQ.C`
- 第13章 变换与滤波: `13FOUR.C`, `13KFFT.C`, `13KFWT.C`
- 第14章 特殊函数: `14BETA.C`, `14BSL1.C`, `14GAM1.C`

## 当前已整理出的代表实现
- 第1章: Horner 多项式求值
- 第1章: Horner, polynomial division, polynomial multiply
- 第2章: complex exponential, complex logarithm, complex division
- 第3章: Box-Muller, linear congruential generator, normal sample statistics
- 第4章: Cholesky, QR, LU, matrix inverse
- 第5章: Jacobi, power iteration, QR iteration, Hessenberg reduction, inverse iteration
- 第6章: Gaussian elimination, conjugate gradient, tridiagonal solver, Jacobi iteration, SOR
- 第7章: Aitken, Newton, Secant
- 第8章: Local Lagrange, Neville, Hermite, natural cubic spline
- 第9章: Romberg, adaptive Simpson, composite Simpson, composite trapezoidal, Gauss-Legendre, Monte Carlo
- 第10章: Improved Euler / ABM4, RK4, backward Euler, Adams-Bashforth4
- 第11章: Linear regression, multiple linear regression, correlation analysis
- 第12章: Nelder-Mead, Golden-section
- 第13章: Fourier series, FFT
- 第14章: Bessel J, regularized incomplete beta, Gamma, error function, Legendre polynomials, normal CDF

## 本次现代化改写策略
- 数据结构: 裸数组改为 `std::vector`, `std::complex`
- 错误处理: 返回码改为异常或显式结果结构体
- 接口风格: 输出参数改为返回值语义
- 教学形式: 每章一个代表实现，强调背景知识、数值稳定性与复杂度

## 暂未纳入 `cpp-numerical` 的章节
- 第15章 排序
- 第16章 查找

这两章更适合放到算法基础或数据结构目录，而不是数值计算目录。
