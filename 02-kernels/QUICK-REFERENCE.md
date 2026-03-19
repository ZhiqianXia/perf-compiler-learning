# 快速参考卡

## 项目完成状态

✅ **8个主分类创建完成**
✅ **18个算子完全实现**
✅ **112个算子框架就绪**
✅ **57个文档文件**
✅ **26个kernel实现**

## 查找算子

### 按分类查找
```
01-element-wise-ops/    → 元素级操作 (72个)
02-linear-algebra/      → 矩阵运算 (3个)
03-convolution/         → 卷积 (3个)
04-pooling-norm/        → 池化/归一化 (10个)
05-tensor-ops/          → 张量操作 (21个)
06-reduction/           → 降维 (8个)
07-comparison-select/   → 比较 (8个)
08-rnn-special/         → RNN (12个)
```

### 已实现的算子

#### 数学运算 (6)
Add, Sub, Mul, Div, Abs, Pow

#### 激活函数 (6)
ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax

#### 超越函数 (5)
Sin, Cos, Exp, Log, Sqrt

#### 矩阵/池化/归一化 (4个)
MatMul, MaxPool, LayerNorm, (加上张量和降维)

#### 张量和降维 (7)
Transpose, Concat, Gather, ReduceSum, ReduceMean, Equal, Where

## 快速代码片段

### 元素级kernel模板
```cuda
__global__ void op_kernel(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = operation(a[idx]);
}

void op(const float* a, float* c, int n, cudaStream_t s=0) {
    int blocks = (n + 255) / 256;
    op_kernel<<<blocks, 256, 0, s>>>(a, c, n);
}
```

### 2D操作模板
```cuda
__global__ void op_2d(const float* in, float* out, int h, int w) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < h && col < w) {
        out[row*w + col] = process(in[row*w + col]);
    }
}
```

### 共享内存降维
```cuda
extern __shared__ float sdata[];
sdata[tid] = input[idx];
__syncthreads();
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

## 重要路径

| 文件 | 描述 |
|------|------|
| README.md | 📖 库说明 |
| PROJECT-SUMMARY.md | 📊 完整总结 |
| OPERATORS-INDEX.md | 📋 130个算子索引 |
| QUICK-REFERENCE.md | 🎯 此文件 |

## 编译示例

```bash
# 编译单个kernel
nvcc -arch=sm_70 -O3 01-element-wise-ops/math-ops/add/kernel.cu -o add_test

# 编译库
nvcc -arch=sm_70 -O3 -dlink *.cu -o cuda_lib.o
```

## 启动配置

### 标准配置
- 线程/块: **256** (元素级)
- 块/网格: **(n+255)/256**
- 共享内存: **0** (大多数情况)

### 2D操置运配置
- 线程块: **dim3(16, 16)** 或 **dim3(32, 32)**
- 块网格: **dim3((w+15)/16, (h+15)/16)**

## 性能指标

| 操作 | 带宽% | 示例 |
|-----|-------|------|
| Add | >95% | 1 op + mem |
| MatMul | 85% | 瓦片优化 |
| Reduce | 70% | 同步开销 |
| Conv | 60-80% | 依赖参数 |

## 添加新算子 - 5分钟快速指南

1. **创建目录** → `mkdir 01-element-wise-ops/new-op`
2. **复制README** → `cp operator-template.md ./new-op/README.md`
3. **编写kernel** → `编辑 new-op/kernel.cu`
4. **实现launcher** → 添加包装函数
5. **测试** → `nvcc -O3 kernel.cu && ./test`

## 文件大小

- 典型kernel: **50-150 行**
- 典型README:** 15-40 行**
- 优化kernel: **100-250 行**

## 热门参考

### Union Find (Reduction)
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = combine(sdata[tid], sdata[tid+s]);
    __syncthreads();
}
```

### 数值稳定 (Softmax)
```cuda
float max_val = PARALLEL_REDUCE_MAX(x);
float sum_exp = PARALLEL_REDUCE_SUM(exp(x - max_val));
output = exp(x - max_val) / sum_exp;
```

### 避免Bank Conflict (Transpose)
```cuda
__shared__ float tile[TILE_SIZE][TILE_SIZE+1];  // +1避免冲突
```

## 常见问题

**Q: 如何选择块大小?**
A: 256 用于一维, 16x16 或 32x32 用于二维, 平衡占用率和寄存器使用

**Q: 何时使用共享内存?**
A: 当多个线程访问相同数据或需要线程同步时

**Q: 如何优化内存?**
A: 合并访问, 避免bank冲突, 最大化缓存命中

---

**提示**: 详见完整文档 → README.md 或 PROJECT-SUMMARY.md
