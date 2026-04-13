# PyTorch Profiler × Compiler 硬件效率 & 性能分析方法论

## 核心理念

建立一套覆盖 **PyTorch.compile → Triton → LLVM → GPU Arch** 端到端链路的分析方法轮，
让每一层都有可观测、可量化、可对比的手段，推动真实应用负载的性能瓶颈定位与优化。

---

## 1. 方法轮总览

```
                    ┌──────────────────────┐
                    │  Application Layer   │
                    │  (PyTorch Profiler)  │
                    └──────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │       torch.compile 层          │
              │  TorchDynamo → AOTAutograd →    │
              │  Inductor (graph lowering)      │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         Triton IR 层            │
              │  Triton kernel 生成 & 调优      │
              │  (tile size, num_warps, stages) │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │          LLVM IR 层             │
              │  PTX / AMDGPU 代码生成          │
              │  (register pressure, spills)    │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │       GPU Architecture 层       │
              │  Nsight Compute / rocprof       │
              │  (occupancy, bandwidth, FLOPS)  │
              └─────────────────────────────────┘
```

---

## 2. 每层的观测手段与关键指标

### Layer 0: Application — PyTorch Profiler

| 手段 | 关键指标 | 对应 Lab |
|------|---------|----------|
| `torch.profiler.profile()` | 算子 wall-time, CPU/CUDA 耗时 | lab1-lab8 |
| `with_flops=True` | 每算子 FLOPs | lab14 |
| `profile_memory=True` | 峰值显存, 分配/释放 | lab4, lab17-20 |
| `record_function()` | 自定义区域标注 | lab12 |
| `torch.utils.benchmark.Timer` | 微基准 median/IQR | lab13 |
| Chrome Trace / TensorBoard | 时间线可视化 | lab5, lab10 |

**核心问题**: 我的模型在 eager 模式下，哪些算子是热点？内存是否是瓶颈？

### Layer 1: torch.compile — Dynamo + AOTAutograd + Inductor

| 手段 | 关键指标 |
|------|---------|
| `TORCH_LOGS="dynamo"` | graph break 原因与位置 |
| `TORCH_LOGS="aot"` | AOTAutograd 前向/反向图 |
| `TORCH_LOGS="output_code"` | Inductor 生成的 Triton kernel 源码 |
| `torch._dynamo.explain(model)(inputs)` | graph break 诊断报告 |
| `torch.compile(fullgraph=True)` | 验证是否能一次图捕获 |
| eager vs compiled profiler 对比 | 算子融合率, kernel 数量减少比 |

**核心问题**: compile 后图是否完整？有多少 graph break？融合了多少算子？

### Layer 2: Triton IR — 生成 Kernel 分析

| 手段 | 关键指标 |
|------|---------|
| `TORCH_LOGS="output_code"` 查看 `@triton.jit` | tile 尺寸, num_warps, num_stages |
| `triton.testing.do_bench()` | kernel 级别 latency |
| `triton.compiler.ASTSource` + `make_launcher` | 查看 Triton IR |
| Inductor 的 `config.coordinate_descent_tuning=True` | 自动调优结果 |
| kernel 数量 & launch overhead | total kernel count per forward |

**核心问题**: 生成的 Triton kernel 是否高效？tile/warp 配置是否合理？

### Layer 3: LLVM IR → PTX/AMDGPU

| 手段 | 关键指标 |
|------|---------|
| `TRITON_CACHE_DIR` 查看 `.llir` / `.ptx` | register 使用量 |
| `cuobjdump -sass` | 实际 SASS 指令 |
| register spill count | spill to local memory 次数 |
| PTX 中 `ld.global` / `st.global` 比例 | 访存效率 |

**核心问题**: 编译器是否引入了register spill？指令选择是否高效？

### Layer 4: GPU Architecture — 硬件级

| 手段 | 关键指标 |
|------|---------|
| `nsys profile` | GPU timeline, kernel overlap |
| `ncu --set full` (Nsight Compute) | SM occupancy, memory throughput, compute throughput |
| Roofline Model | Arithmetic Intensity vs 硬件天花板 |
| `ncu --metrics` 自选 | `dram__throughput`, `sm__throughput`, `l1/l2 hit rate` |

**核心问题**: 是 compute-bound 还是 memory-bound？距离硬件极限还差多少？

---

## 3. 端到端分析工作流

```
Step 1: Baseline Profiling (Eager)
  └─ lab8 模板 → 定位 Top-N 热点算子
  └─ lab14 → 统计 FLOPs
  └─ lab4/17 → 显存峰值

Step 2: torch.compile 对比
  └─ lab9 → eager vs compiled 耗时
  └─ TORCH_LOGS → graph break 诊断
  └─ lab32 (新) → 完整端到端链路检查

Step 3: Inductor 生成代码审查
  └─ lab33 (新) → 查看 Triton kernel 源码
  └─ 检查 tile/warp 配置
  └─ 对比 nvfuser / eager 的 kernel 数量

Step 4: 硬件级验证
  └─ lab34 (新) → Roofline 分析
  └─ nsys/ncu 验证 occupancy & bandwidth
  └─ 确认是否达到硬件效率上限

Step 5: 迭代优化
  └─ 调整模型结构消除 graph break
  └─ 尝试不同 compile 模式 (default/reduce-overhead/max-autotune)
  └─ 手写 Triton kernel 替换低效生成代码
```

---

## 4. 关键公式

### 硬件效率 (Hardware Efficiency)

$$
\eta_{compute} = \frac{\text{Achieved FLOPS}}{\text{Peak FLOPS}} \times 100\%
$$

$$
\eta_{bandwidth} = \frac{\text{Achieved Bandwidth (GB/s)}}{\text{Peak Bandwidth (GB/s)}} \times 100\%
$$

### Arithmetic Intensity (Roofline)

$$
AI = \frac{\text{FLOPs}}{\text{Bytes Accessed (DRAM)}} \quad (\text{FLOP/Byte})
$$

### Roofline 天花板

$$
\text{Attainable FLOPS} = \min(\text{Peak FLOPS}, \; AI \times \text{Peak Bandwidth})
$$

### 算子融合率

$$
\text{Fusion Ratio} = 1 - \frac{\text{Compiled Kernel Count}}{\text{Eager Kernel Count}}
$$

### Compile Speedup

$$
\text{Speedup} = \frac{T_{eager}}{T_{compiled}}
$$

---

## 5. 新增实验索引

| Lab | 内容 | 覆盖层 |
|-----|------|--------|
| `lab32_compile_e2e.py` | 端到端 compile 诊断: graph break + 生成代码 + profile 对比 | L0-L2 |
| `lab33_triton_inspect.py` | 查看 Inductor 生成的 Triton kernel，分析 tile/warp 配置 | L2-L3 |
| `lab34_roofline.py` | 基于 profiler 数据构建 Roofline 分析 | L0+L4 |

---

## 6. 工具链速查

| 工具 | 用途 | 安装 / 使用 |
|------|------|------------|
| `torch.profiler` | Python 级算子剖析 | PyTorch 内置 |
| `torch.compile` | JIT 编译优化 | PyTorch 2.0+ |
| `TORCH_LOGS` | 编译器子系统日志 | 环境变量 |
| `torch._dynamo.explain()` | graph break 诊断 | PyTorch 2.0+ |
| `triton` | GPU kernel DSL | `pip install triton` |
| `nsys` | GPU 时间线分析 | NVIDIA Nsight Systems |
| `ncu` | Kernel 级硬件计数器 | NVIDIA Nsight Compute |
| Perfetto UI | Chrome Trace 可视化 | https://ui.perfetto.dev |
| TensorBoard | Profiler 可视化 | `pip install torch-tb-profiler` |

---

## 7. 真实应用负载分析建议

1. **用真实 batch size 和 sequence length**：避免 toy input 掩盖真实瓶颈
2. **测量训练 + 推理**：训练多了 backward + optimizer，瓶颈不同
3. **包含数据加载**：lab15 DataLoader 瓶颈不容忽视
4. **多次采样取中位数**：lab13 benchmark Timer 方法
5. **对比不同 compile 模式**：`default` / `reduce-overhead` / `max-autotune`
6. **关注 graph break**：一个 graph break 可能毁掉所有融合收益
7. **建立性能基线**：每次优化前后都要有 profile 数据做对比

---

## 8. 芯片研发阶段落地路线图

方法轮不只服务于"已有GPU上的模型调优"，其更大价值在于**贯穿芯片生命周期**，
让 PyTorch → Triton → LLVM → Arch 这条链路在芯片每个阶段都产出可量化的交付物。

### 总览：五阶段 × 五层方法轮映射

```
时间轴 ──────────────────────────────────────────────────────────────────────────────►

 Architecture         Pre-silicon          Bring-up (Day 0)   Perf Opt (Wk 1-4)   Ecosystem (Mo 2+)
 ┌─────────┐          ┌─────────┐          ┌─────────┐        ┌─────────┐          ┌─────────┐
 │ L0-L3   │          │ L3-L4   │          │ L0-L4   │        │ L0-L4   │          │ L0-L2   │
 │ 负载画像│   ──►    │ LLVM/   │   ──►    │ 全链路  │ ──►    │ Roofline│   ──►    │ Triton  │
 │ 参数探索│          │ ISA验证 │          │ 冒烟测试│        │ 系统调优│          │ Backend │
 └─────────┘          └─────────┘          └─────────┘        └─────────┘          └─────────┘
   交付物:              交付物:              交付物:            交付物:              交付物:
   架构规格建议书       kernel正确性报告     硬件bug清单        性能基线+优化报告    新op/fusion覆盖率
```

---

### 8.0 Architecture（前期架构设计）

> **这是方法轮最被低估的价值点**：在 RTL 一行代码都没写之前，
> 用真实 AI 负载数据驱动架构决策，避免 "先拍脑袋定参数，流片后发现跑不好" 的经典陷阱。

**痛点**: 传统架构设计依赖:
- 竞品逆向 + 经验直觉 → 容易刻舟求剑
- SPEC/MLPerf 宏观 benchmark → 太粗，指导不了微架构参数
- 手工建 Excel 估算 → 假设太多，与真实负载脱节

**核心理念**: 用 **PyTorch → Triton → LLVM IR 这条链路做"编译器驱动的架构探索"(Compiler-Driven Architecture Exploration)**。

#### 8.0.1 价值一：真实负载画像 (Workload Characterization)

不再猜"AI 模型需要什么"，而是直接从真实模型的编译产物中统计。

```
Model Zoo (LLaMA, ViT, Stable Diffusion, Whisper, ...)
    │
    ▼ torch.compile + profiler (with_flops, profile_memory)
    │
    ├─ 算子频率分布: 哪些 aten op 占了 90% 的时间?
    ├─ 计算/访存比分布: 每个 kernel 的 Arithmetic Intensity 直方图
    ├─ Tensor shape 分布: 典型的 M/N/K 是什么范围?
    ├─ 数据类型分布: FP32 vs FP16 vs BF16 vs INT8 的占比
    └─ 内存访问模式: sequential vs strided vs random
```

**指导架构决策**:

| 负载画像数据 | 对应架构参数 | 决策示例 |
|-------------|------------|---------|
| matmul 占 75% 的 FLOPS | Tensor Core / Systolic Array 规格 | 选 256×256 还是 128×128 的 MAC array |
| 中位数 AI = 30 FLOP/Byte | DRAM 带宽 vs 算力配比 | Ridge point 放在 AI=50，而非盲目堆算力 |
| 85% kernel 的 M,N ∈ [128, 4096] | SRAM (shared memory) 容量 | 确保 128KB SRAM 能容纳一个 tile |
| 60% 时间在 BF16 | 专用精度单元 | BF16 吞吐做到 FP32 的 2x，不需要 FP64 |
| reduce op 占 15% 时间 | warp-level reduce 硬件 | 是否需要 cross-warp reduce 指令 |
| 5% 时间在 softmax | 特殊函数单元 (SFU) | 是否值得做 exp/log 的硬件加速 |

$$
\text{Balance Point} = \frac{\text{Median AI of workload}}{\text{Peak FLOPS / Peak BW}} \rightarrow \text{理想应} \approx 1
$$

#### 8.0.2 价值二：ISA 完备性验证 (ISA Sufficiency Check)

Triton compiler 会把高层算子降解为一组 **有限的底层原语**，这组原语就是 ISA 的最小需求集。

```
Triton kernel (真实模型生成)
    │
    ▼ Triton → LLVM IR
    │
    ├─ 指令频率统计: 哪些 LLVM IR 指令被用到?
    ├─ 缺失指令检测: 哪些 intrinsic 在你的后端未实现?
    ├─ 关键 pattern: 有多少 atomic? barrier? 向量宽度?
    └─ 寄存器压力: 典型 kernel 需要多少寄存器?
```

**直接输出 ISA 设计 checklist**:

```
┌─────────────────────────────────────────────────────────────┐
│  ISA Sufficiency Report (from 50 Triton kernels)           │
├──────────────────────────┬──────────┬───────────────────────┤
│ Capability               │ Required │ Status                │
├──────────────────────────┼──────────┼───────────────────────┤
│ FMA (f32, f16, bf16)     │    ✓     │ ✅ ISA v1.0 已有      │
│ dot product (tile-level) │    ✓     │ ✅ ISA v1.0 已有      │
│ atomic_add (f32)         │    ✓     │ ⚠️ 仅 i32, 需补 f32   │
│ barrier (block-level)    │    ✓     │ ✅ ISA v1.0 已有      │
│ async_copy (global→smem) │    ✓     │ ❌ 缺失，影响 prefetch │
│ warp_shuffle             │    ✓     │ ❌ 缺失，影响 reduce   │
│ exp / log (SFU)          │   可选   │ ⚠️ 可用软件模拟        │
│ vector width ≥ 128 bit   │    ✓     │ ✅ 256 bit             │
│ register file ≥ 128 regs │    ✓     │ ⚠️ 只有 96, 需评估     │
└──────────────────────────┴──────────┴───────────────────────┘
```

#### 8.0.3 价值三：内存子系统设计 (Memory Hierarchy Sizing)

```
从 Triton kernel 的 tiling pattern 中提取:
    │
    ├─ 每个 tile 的数据量 → SRAM (shared memory) 最小容量
    ├─ tile 间的复用率 → SRAM 最优容量 (避免 thrashing)
    ├─ global memory 带宽需求 → DRAM 通道数 / HBM 选型
    └─ 多级 prefetch 深度 → 流水线 stage 数
```

**关键公式**:

$$
\text{SRAM}_{min} = \text{TileM} \times \text{TileK} \times \text{dtype\_bytes} \times \text{num\_buffers}
$$

$$
\text{DRAM BW}_{req} = \frac{\text{Total Bytes per kernel}}{\text{Target Latency}} \times \text{Safety Margin}
$$

$$
\text{Reuse Ratio} = \frac{\text{Total FLOPS}}{\text{Total Bytes Loaded from DRAM}} = AI
$$

| Triton kernel 特征 | 内存设计决策 |
|-------------------|------------|
| BLOCK_M=128, BLOCK_K=64, FP16 | SRAM 至少 16KB per buffer, 双缓冲需 32KB |
| num_stages=3 (software pipelining) | 三级流水: SRAM 需 48KB |
| 10 个 kernel 并发 (SM 级) | 总 SRAM = 48KB × 10 = 480KB per SM |
| 峰值 DRAM 读取 400 GB/s | 需要 HBM2e (3.2 TB/s 总线) × N stack |

#### 8.0.4 价值四：计算单元设计空间探索 (Compute DSE)

用 profiler 数据做参数扫描，而非拍脑袋:

```
对于同一组 workload:
  ├─ 扫描 Peak FLOPS = [10, 20, 40, 80, 160] TFLOPS
  ├─ 扫描 Peak BW    = [500, 1000, 2000, 4000] GB/s
  ├─ 对每组 (FLOPS, BW) 计算 Roofline
  └─ 在 Roofline 上标注所有 kernel 的 (AI, achieved_perf)
      │
      ▼
  找到 "边际收益拐点": 再加算力/带宽，workload 也上不去了
```

$$
\text{Utilization}(F, B) = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{Achieved}_i}{\min(F, \; AI_i \times B)}
$$

其中 $F$ = Peak FLOPS, $B$ = Peak BW, $N$ = kernel 数量。
**目标**: 找到使 Utilization 最大的 $(F, B)$ 组合，就是最优算力/带宽配比。

#### 8.0.5 价值五：Fusion 收益预估 (Fusion Benefit Estimation)

在硬件还不存在时，预估算子融合能省多少:

```
Eager mode:  kernel 1 (DRAM→reg→DRAM) + kernel 2 (DRAM→reg→DRAM)
Fused:       kernel 1+2 (DRAM→reg→reg→DRAM)
                                                    
省掉的 = 中间 tensor 的 DRAM 读 + 写 = 2 × tensor_size × dtype_bytes
```

$$
\text{Fusion Bandwidth Saving} = \sum_{\text{fused pairs}} 2 \times \text{intermediate\_tensor\_bytes}
$$

$$
\text{Fusion Speedup}_{upper} = \frac{T_{unfused}}{T_{unfused} - \text{saved\_mem\_time}}
$$

这直接影响:
- **是否值得在硬件上做 kernel fusion 支持** (需要更大 register file / SRAM)
- **fusion 的粒度**: 2-op fusion 够了，还是需要 10-op mega kernel?

#### 8.0.6 对应 Lab 与交付物

**对应 Lab**: `lab38_arch_workload_analysis.py`

**架构设计阶段交付物清单**:

| 交付物 | 来源 | 用途 |
|--------|------|------|
| Workload Characterization Report | Profiler + Triton compile | 确定算力/带宽/精度配比 |
| ISA Sufficiency Checklist | LLVM IR 分析 | 确认 ISA 无遗漏 |
| SRAM Sizing Recommendation | Triton tiling 分析 | 确定 shared memory 容量 |
| Compute DSE Pareto Front | Roofline 参数扫描 | 选择最优 (FLOPS, BW) 点 |
| Fusion Benefit Matrix | Inductor fusion 分析 | 决定硬件 fusion 支持范围 |
| Register File Sizing | LLVM IR register pressure | 确定 RF 深度 (64/128/256) |

---

### 8.1 Pre-silicon（硅前验证）

**痛点**: 硬件模拟器慢 (10万倍+)，无法跑真实模型，传统方法只能跑手写汇编片段。

**方法轮落地**:

```
PyTorch model (参考实现)
    │
    ▼ torch.compile + TORCH_LOGS="output_code"
Inductor 生成 Triton kernel 源码
    │
    ▼ Triton compiler (--target=virtual_isa)
LLVM IR / 虚拟 ISA 汇编
    │
    ▼ 在硬件模拟器上执行
正确性验证 (golden vs 模拟器输出)
```

| 步骤 | 做什么 | 用什么 | 产出 |
|------|--------|--------|------|
| 1. 提取算子图 | 从真实模型提取计算图 | `torch.fx.symbolic_trace` / `torch.export` | `ExportedProgram` |
| 2. 生成 Triton kernel | 不跑 GPU，只要源码 | `TORCH_LOGS="output_code"` | `.py` (Triton kernel) |
| 3. 编译到虚拟 ISA | Triton → LLVM → 你的后端 | 自定义 Triton backend (`triton.compiler`) | `.ll` / `.s` |
| 4. 模拟器执行 | 在 cycle-accurate sim 上跑 | 自研 ISA 模拟器 | 输出 tensor |
| 5. Golden 对比 | PyTorch eager 结果 vs 模拟器结果 | `torch.allclose(atol, rtol)` | 正确性报告 |

**关键收益**:
- 不再只验证"1+1=2"的简单指令，而是验证 **Triton 自动生成的真实 kernel pattern**
- 提前暴露 ISA 设计缺陷（例如缺少 atomic、barrier 语义不对等）
- 测试覆盖率从"几十条手写 case" 提升到"整个模型的所有算子"

**对应 Lab**: `lab35_presilicon_extract.py`

---

### 8.2 Bring-up（Day 0 硬件到手）

**痛点**: 传统做法 — 写一个小 CUDA kernel 验证 GEMM 能跑，然后花几周扩展。

**方法轮落地**:

```
Day 0 第 1 小时:
  python lab32_compile_e2e.py     ← 直接跑完整 Transformer
       │
       ├─ 成功 → 硬件基本功能 OK，进入 Step 2
       │
       └─ 失败 → 精确定位到哪个 Triton kernel 挂了
                  ↓
            TORCH_LOGS="output_code" 提取该 kernel
                  ↓
            单独在硬件上跑这个 kernel → 定位硬件 bug
```

| 阶段 | 传统方法 | 方法轮加速 |
|------|---------|-----------|
| 功能验证 | 手写 10 个 kernel，逐个跑 | `torch.compile` 自动生成 100+ kernel，一次跑完 |
| Bug 定位 | printf / 二分法 | `lab36` 自动 bisect: 逐算子对比 eager vs compiled 输出 |
| 回归测试 | 无体系 | `lab35` 提取的 golden 直接复用 |
| 覆盖率度量 | "感觉差不多了" | 量化: 多少 aten op 通过 / 失败 / 精度不达标 |

**自动化 Bisect 流程**:

```
全模型 compile 失败
    │
    ▼ 二分 graph
前半段 OK? ──── Yes ──► 后半段有问题
    │                        │
    No                       ▼ 继续二分
    │
    ▼ 继续二分 ...
    
最终定位到: aten::scaled_dot_product_attention 的 Triton kernel
    │
    ▼ 在硬件上单独跑这个 kernel
    ▼ 对比 CPU golden
    ▼ 发现: 硬件 shared memory barrier 实现有 bug
```

**对应 Lab**: `lab36_bringup_bisect.py`

---

### 8.3 性能优化（Week 1-4）

**痛点**: 传统做法 — 逐个算子手动调优，缺乏全局视角。

**方法轮落地**:

```
Week 1: 建立性能基线
  ├─ lab34 Roofline → 全模型每层的 AI / 效率
  ├─ lab32 → Eager vs Compiled speedup
  └─ nsys → GPU timeline, idle 时间占比

Week 2-3: 定向优化
  ├─ Memory-bound 算子 → 检查访存模式, 优化 tiling
  ├─ Compute-bound 算子 → 检查 occupancy, 减少 register spill
  ├─ Launch-bound → reduce-overhead 模式 / CUDA Graph
  └─ lab33 → 逐 kernel 审查 Triton 生成代码

Week 4: 验证 & 报告
  ├─ 对比优化前后 Roofline
  ├─ 量化 compile mode 差异 (default / reduce-overhead / max-autotune)
  └─ 输出性能报告: 每算子效率 + 整体 throughput
```

**系统性分析决策树**:

```
              达到峰值性能了吗？
                    │
         ┌─── No ──┴── Yes ──┐
         │                    │
    Roofline 中               Done! 记录基线
    在哪个区域？
         │
    ┌────┴─────┐
    │          │
 Memory     Compute
  Bound      Bound
    │          │
    ▼          ▼
 检查:       检查:
 - L1/L2     - occupancy
   hit rate  - warp divergence
 - coalesce  - register spill
 - tiling    - instruction mix
    │          │
    ▼          ▼
 优化:       优化:
 - 更大tile  - 减少register
 - 预取      - 调warp配置
 - 数据布局  - 算法替换
```

**关键指标仪表板**:

$$
\text{Throughput Gap} = \frac{\text{Roofline Limit} - \text{Achieved}}{\text{Roofline Limit}} \times 100\%
$$

$$
\text{Kernel Overhead Ratio} = \frac{\text{Total Kernel Launch Time}}{\text{Total Kernel Execution Time}}
$$

$$
\text{Memory Efficiency} = \frac{\text{Useful Bytes}}{\text{Total DRAM Bytes Accessed}}
$$

**对应 Lab**: `lab34_roofline.py` (已有) + `lab32_compile_e2e.py` (已有)

---

### 8.4 生态兼容（Month 2+）

**痛点**: 等 PyTorch 官方支持新硬件要 6-12 个月。自建生态需要体系化方法。

**方法轮落地**:

```
PyTorch (不改) ──► torch.compile (不改) ──► Inductor ──► Triton Backend（自研）
                                               │
                                               ▼
                                          你的 LLVM 后端
                                               │
                                               ▼
                                          你的硬件
```

**自研 Triton Backend 路线图**:

| 阶段 | 目标 | 方法轮支撑 |
|------|------|-----------|
| Phase 1: 骨架 | `@triton.jit` 能编译到你的 ISA | Triton compiler plugin API |
| Phase 2: 基础 op | matmul / elementwise / reduce 通过 | `lab35` golden 对比 |
| Phase 3: Inductor 对接 | `torch.compile` → 你的后端 E2E 跑通 | `lab36` bisect 定位问题 |
| Phase 4: 覆盖率 | 真实模型跑通率 > 90% | `lab37` 自动化覆盖率统计 |
| Phase 5: 性能 | 达到理论峰值 70%+ | `lab34` Roofline 分析 |

**op 覆盖率追踪**:

```
┌──────────────────────────────────────────────────────┐
│  aten Op Coverage Dashboard                          │
├──────────────┬──────────┬──────────┬─────────────────┤
│ Category     │ Total    │ Passing  │ Coverage        │
├──────────────┼──────────┼──────────┼─────────────────┤
│ matmul       │    12    │    12    │ ████████████ 100%│
│ elementwise  │    45    │    38    │ ██████████░░  84%│
│ reduce       │    18    │    14    │ █████████░░░  78%│
│ attention    │     6    │     3    │ ██████░░░░░░  50%│
│ conv         │     8    │     2    │ ███░░░░░░░░░  25%│
│ Total        │    89    │    69    │ █████████░░░  78%│
└──────────────┴──────────┴──────────┴─────────────────┘
```

**Fusion Pattern 兼容性**:

```
Inductor 常见 fusion pattern:
  ✅ pointwise + pointwise       (已支持)
  ✅ matmul + bias + relu        (已支持)
  ⚠️  attention (flash variant)   (部分支持)
  ❌ conv + bn + relu            (未支持)
  ❌ custom backward fusion      (未支持)
```

**对应 Lab**: `lab37_backend_coverage.py`

---

### 8.5 四阶段联动：CI/CD 持续验证

把方法轮固化到 CI，每次硬件/编译器变更自动回归:

```yaml
# .github/workflows/chip_validation.yml 示意
stages:
  pre-silicon:
    - extract golden from PyTorch models (lab35)
    - compile to virtual ISA
    - run on simulator
    - compare golden

  bringup:
    - run lab36 bisect on real hardware
    - report: pass/fail per aten op
    - flag: new hardware bugs

  perf:
    - run lab34 roofline on benchmark suite
    - compare vs previous build
    - alert if regression > 5%

  ecosystem:
    - run lab37 coverage on model zoo
    - report: op coverage %, failed ops list
    - track fusion pattern support
```

**度量看板汇总**:

| 指标 | Pre-silicon | Bring-up | Perf Opt | Ecosystem |
|------|-----------|----------|----------|-----------|
| Kernel 正确性 | ✅ golden 对比 | ✅ bisect 定位 | — | ✅ 回归测试 |
| Op 覆盖率 | — | ✅ pass/fail | — | ✅ % tracking |
| 硬件效率 η | — | — | ✅ Roofline | ✅ 基线对比 |
| Compile Speedup | — | — | ✅ 3 模式对比 | ✅ 版本间跟踪 |
| Fusion 覆盖 | — | — | ✅ 融合率 | ✅ pattern 追踪 |

---

## 9. 新增实验索引（完整）

| Lab | 内容 | 覆盖层 | 芯片阶段 |
|-----|------|--------|---------|
| `lab32_compile_e2e.py` | 端到端 compile 诊断 | L0-L2 | Bring-up, Perf |
| `lab33_triton_inspect.py` | Triton kernel 审查 | L2-L3 | Perf, Ecosystem |
| `lab34_roofline.py` | Roofline 分析 | L0+L4 | Arch, Perf |
| `lab35_presilicon_extract.py` | 硅前 golden 提取 & 虚拟 ISA 编译 | L1-L3 | Pre-silicon |
| `lab36_bringup_bisect.py` | Bring-up 自动 bisect 定位硬件 bug | L0-L4 | Bring-up |
| `lab37_backend_coverage.py` | Triton backend op 覆盖率追踪 | L0-L2 | Ecosystem |
| `lab38_arch_workload_analysis.py` | 架构设计负载画像: 算子/shape/AI/ISA/SRAM 分析 | L0-L3 | Architecture |
