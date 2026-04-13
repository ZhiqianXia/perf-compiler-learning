# From PyTorch to Silicon

**A Compiler-Driven Approach to AI Accelerator Design**

> *"Connecting Software to Silicon — this book is the bridge."*

---

## 书籍元数据

- **目标页数**: ~300 页 (正文约 270 页 + 前言/附录/索引 30 页)
- **目标读者**: 芯片架构师、AI 编译器工程师、GPU 性能工程师、研究生
- **前置知识**: 基础 Python、了解神经网络概念、接触过 C/C++
- **配套代码**: 38 个可执行 lab + `run_analysis.py` 自动化框架
- **核心卖点**: 唯一一本用 PyTorch.compile → Triton → LLVM → 流片芯片 贯穿全书的教材

---

## 全书结构

```
                        全书 ~300 页
  ┌───────────────────────────────────────────────────┐
  │  前言 + 导读                              (~8 页)  │
  ├───────────────────────────────────────────────────┤
  │  Part I   基础: 从模型到硬件              (~60 页) │  Ch 1-4
  ├───────────────────────────────────────────────────┤
  │  Part II  观测: 性能分析方法论            (~56 页) │  Ch 5-8
  ├───────────────────────────────────────────────────┤
  │  Part III 编译: 从 Python 到机器码        (~64 页) │  Ch 9-12
  ├───────────────────────────────────────────────────┤
  │  Part IV  协同设计: 编译器驱动架构探索    (~52 页) │  Ch 13-16
  ├───────────────────────────────────────────────────┤
  │  Part V   工程落地: 从仿真到流片          (~40 页) │  Ch 17-20
  ├───────────────────────────────────────────────────┤
  │  附录 A-D + 索引                          (~20 页) │
  └───────────────────────────────────────────────────┘
```

---

## 前言 (~8 页)

### 致读者 (2 页)
- 为什么写这本书：AI 芯片行业的 "最后一公里" 问题
- 三种读者都只看到大象的一部分：框架工程师、编译器工程师、芯片架构师
- 本书的使命：把 PyTorch → Silicon 这条路完整走一遍

### 本书的路线图 (2 页)
- 五部分的逻辑：基础 → 观测 → 编译 → 协同设计 → 落地
- 每部分对应芯片生命周期的哪个阶段
- 如何用配套 lab 实践 (38 个可执行脚本 + 自动化框架)

### 如何阅读本书 (2 页)
- 三条建议路径：
  - **路径 A (架构师)**: Ch1-2 → Part IV → Part V → Part III 按需
  - **路径 B (编译器工程师)**: Ch3-4 → Part III → Part II → Part IV
  - **路径 C (性能工程师)**: Part II → Part III → Part IV → Part V
- 每章的 Lab 索引和预估实践时间

### 环境配置 (2 页)
- 软件: PyTorch 2.x, Triton, LLVM, Nsight
- 硬件: 至少一块 NVIDIA GPU (Volta+)；CPU-only 可跑部分 lab
- 配套代码仓库结构总览

---

## Part I — 基础: 从模型到硬件 (~60 页)

> 建立贯穿全书的计算模型和心智框架。读者读完 Part I 后，
> 应该能画出一个 PyTorch 算子从 Python 到 GPU SM 的完整数据流。

---

### Chapter 1: AI 计算的硬件基础 (~16 页)

**核心问题**: 一个 `torch.mm(A, B)` 在 GPU 上究竟发生了什么？

#### 1.1 从 CPU 到 GPU: 并行计算范式转变 (3 页)
- 延迟导向 vs 吞吐导向
- SIMT 执行模型: thread → warp → block → grid
- 为什么 AI 负载天然适合 GPU

#### 1.2 GPU 微架构速览 (4 页)
- SM (Streaming Multiprocessor) 内部结构
- Tensor Core / Matrix Engine: 从标量 FMA 到 tile-level MMA
- 内存层次: register → shared memory → L1/L2 → HBM
- 图: SM 内部数据流图 (寄存器 → ALU → 共享内存 → 全局内存)

#### 1.3 关键硬件参数及其含义 (3 页)
- Peak FLOPS 怎么算: SM 数 × Cores/SM × 2 (FMA) × 频率
- Peak Bandwidth 怎么算: 总线宽度 × 频率 × 通道数
- 算力/带宽比 (Compute-to-Bandwidth Ratio)
- 表: 主流 GPU 参数对照 (V100 / A100 / H100 / MI300X)

#### 1.4 延伸: 自定义 AI 加速器的通用架构模板 (4 页)
- 从 GPU 抽象出通用加速器骨架: PE Array + SRAM + NoC + DRAM Controller
- Systolic Array vs SIMT vs Dataflow: 三种主流计算模型
- 为什么 "编译器适配性" 是架构选型的隐形约束

#### 1.5 实践与思考 (2 页)
- **Lab 0**: 环境确认 (`lab0_env_check.py`)
- **Lab 3**: GPU profiling 观察 CPU/CUDA 活动 (`lab3_gpu_profile.py`)
- 思考题: 你的 GPU 理论峰值是多少? 实际能达到多少?

---

### Chapter 2: Roofline Model — 性能分析的第一原理 (~14 页)

**核心问题**: 我的程序离硬件极限还有多远? 瓶颈在计算还是在访存?

#### 2.1 Arithmetic Intensity: 一个比值统一所有分析 (3 页)
- 定义: $AI = \text{FLOPs} / \text{Bytes}$
- 直觉: AI 高 = 计算密集, AI 低 = 访存密集
- 常见算子的 AI 范围: element-wise (~0.1) → matmul (~100+) → softmax (~1-5)

#### 2.2 Roofline 图: 一张图定位瓶颈 (4 页)
- 两条线: 算力天花板 (水平) + 带宽天花板 (斜线)
- Ridge Point: 两条线的交点, 即硬件 "平衡点"
- $\text{Attainable FLOPS} = \min(\text{Peak FLOPS}, \; AI \times \text{Peak BW})$
- 图: 标注了 matmul / softmax / layernorm / attention 的典型 AI

#### 2.3 从理论到测量: PyTorch Profiler 构建 Roofline (4 页)
- `with_flops=True` 获取算子 FLOPs
- `profile_memory=True` 获取内存分配
- 将 profiler 数据映射到 Roofline 图上
- 代码演示: 完整的 Roofline 构建流程

#### 2.4 多层 Roofline: L1 / L2 / HBM (2 页)
- 不同内存层级有不同的带宽天花板
- Nsight Compute 的精确 Roofline 支持
- `ncu --set roofline` 实战

#### 2.5 实践与思考 (1 页)
- **Lab 14**: FLOPs 统计 (`lab14_flops_count.py`)
- **Lab 34**: 完整 Roofline 分析 (`lab34_roofline.py`)
- 思考题: 你的模型是 compute-bound 还是 memory-bound?

---

### Chapter 3: PyTorch 执行模型 (~16 页)

**核心问题**: 从 `model(x)` 到 GPU kernel launch，中间经过了哪些层？

#### 3.1 Eager Mode: 即时执行的优势与代价 (3 页)
- Python → Dispatcher → ATen → CUDA kernel 的调用链
- 每个算子独立 launch 的开销
- 为什么 eager mode 简单但难以优化

#### 3.2 计算图: 从动态到静态 (3 页)
- torch.fx 符号追踪: 捕获计算图
- TorchScript (jit.script / jit.trace): 历史方案与局限
- torch.export: 新一代图捕获
- 图模式的性能优势: 提前知道全局信息 → 可以做融合

#### 3.3 Autograd: 反向传播的实现机制 (3 页)
- grad_fn 链: 前向图如何构建反向图
- `save_for_backward` 与内存 trade-off
- Gradient Checkpointing: 时间换空间

#### 3.4 Dispatcher: PyTorch 的算子路由系统 (3 页)
- 多后端路由: CPU / CUDA / XLA / PrivateUse1
- 功能性 key: Autograd / Autocast / FuncTorchBatched
- 为什么理解 dispatcher 对接入新硬件至关重要

#### 3.5 Tensor 内存布局 (2 页)
- contiguous vs non-contiguous, stride 语义
- 内存对齐对 kernel 性能的影响
- channels_last (NHWC) 在 Tensor Core 上的优势

#### 3.6 实践与思考 (2 页)
- **Lab 31**: torch.fx 符号追踪 (`lab31_fx_graph.py`)
- **Lab 21**: contiguous 性能差异 (`lab21_contiguous_perf.py`)
- **Lab 30**: forward/backward hook 调试 (`lab30_model_hooks.py`)

---

### Chapter 4: AI 负载核心算子 (~14 页)

**核心问题**: 哪些算子撑起了 90% 的 AI 计算？ 它们的计算/访存特征是什么？

#### 4.1 矩阵乘法: 所有 AI 计算的基石 (3 页)
- GEMM / batched GEMM: 定义与 FLOPs 公式
- $\text{FLOPs}(\text{GEMM}) = 2MNK$
- Tiling 策略: 为什么分块是性能的关键
- 图: 不同 M/N/K 下的 Arithmetic Intensity 变化

#### 4.2 注意力机制: 从标准到 Flash Attention (4 页)
- $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d})V$
- 标准实现: $O(N^2)$ 内存, memory-bound
- Flash Attention: 分块 + 在线 softmax → $O(N)$ 内存
- 为什么 Flash Attention 本质是一个编译器优化问题

#### 4.3 Normalization / Activation / Element-wise (3 页)
- LayerNorm, BatchNorm: reduce + scale → memory-bound
- GELU, SiLU, ReLU: element-wise → 极低 AI → 融合候选
- 融合的本质: 避免中间 tensor 经过 DRAM

#### 4.4 卷积 / Embedding / 其他 (2 页)
- Conv2d: im2col + GEMM vs direct convolution
- Embedding lookup: 随机访存, 对带宽的独特需求
- Reduction (sum, mean, max): warp-level 并行

#### 4.5 负载画像: 典型模型的算子分布 (2 页)
- LLM (GPT/LLaMA): matmul 60-70%, attention 15-20%, elementwise 10%
- Vision Transformer: matmul 50%, attention 20%, conv (patch embed) 10%
- Stable Diffusion: conv 40%, attention 30%, matmul 20%
- 表: 跨模型算子频率统计

---

## Part II — 观测: 性能分析方法论 (~56 页)

> 把 "性能分析" 从零散的工具使用，升级为系统化的工程方法论。
> 读者读完 Part II 后，应该能对任意 PyTorch 模型做出完整的性能诊断报告。

---

### Chapter 5: PyTorch Profiler 体系 (~16 页)

**核心问题**: 如何用最小侵入性获取最大信息量？

#### 5.1 Profiler 基础: 一次完整的 profile 过程 (3 页)
- `torch.profiler.profile()` 核心参数
- `ProfilerActivity.CPU` + `ProfilerActivity.CUDA`
- `key_averages().table()`: 第一份有意义的输出
- 为什么 warmup 至关重要

#### 5.2 按维度下钻: shape / stack / memory / flops (4 页)
- `group_by_input_shape`: 同一算子不同 shape 的性能差异
- `with_stack=True`: 追踪到 Python 源码行号
- `profile_memory=True`: 峰值显存与分配热点
- `with_flops=True`: 计算量统计
- 代码: 综合模板 (`lab8_full_template.py` 完整注释版)

#### 5.3 可视化: Chrome Trace 与 TensorBoard (3 页)
- `export_chrome_trace()` → Perfetto UI
- `tensorboard_trace_handler()` → TensorBoard Profiler
- 如何阅读时间线: CPU 活动 / CUDA kernel / 内存操作
- 图: 标注过的典型 trace 截图解读

#### 5.4 Schedule 与长期采样 (2 页)
- `schedule(wait, warmup, active, repeat)` 机制
- `prof.step()` 控制采样窗口
- 多 step 训练循环的 profile 策略

#### 5.5 编程式分析: 从 event 到自动化 (2 页)
- 遍历 `key_averages()` 提取结构化数据
- `record_function()` 自定义标注区域
- 将 profiler 数据导出为 JSON / DataFrame 做进一步分析

#### 5.6 实践与思考 (2 页)
- **Lab 1-8**: 基础 profiling 全系列
- **Lab 12**: record_function 标注
- **Lab 16**: 编程式事件分析
- 综合练习: 对一个真实模型输出完整 profile 报告

---

### Chapter 6: 内存分析与优化 (~14 页)

**核心问题**: 显存是怎么用完的？哪些用量是必要的，哪些可以省？

#### 6.1 GPU 内存模型: 分配、缓存、碎片 (3 页)
- `memory_allocated()` vs `memory_reserved()`: 已用 vs 已占
- PyTorch caching allocator 的作用与副作用
- `empty_cache()` 的正确使用时机
- 图: 显存使用随训练 step 的变化曲线

#### 6.2 峰值追踪与内存快照 (3 页)
- `max_memory_allocated()` / `max_memory_reserved()`
- Memory Snapshot: `memory._record_memory_history()`
- 快照可视化: https://pytorch.org/memory_viz
- 案例: 找到一个隐藏的中间 tensor 峰值

#### 6.3 常见内存问题与修复 (3 页)
- 内存泄漏: grad_fn 意外保持引用
- 过大的 activation 缓存: gradient checkpointing 解法
- 非原地操作的临时分配: in-place ops 的收益与风险

#### 6.4 Host-Device 数据传输 (3 页)
- `pin_memory` + `non_blocking`: 异步传输
- CPU-GPU 同步开销: `torch.cuda.synchronize()` 的代价
- DataLoader 配置: `num_workers`, `prefetch_factor`

#### 6.5 实践与思考 (2 页)
- **Lab 17-20**: 内存调试全系列
- **Lab 23**: 同步开销量化
- **Lab 24**: pin_memory 实测

---

### Chapter 7: 精度、量化与混合精度 (~12 页)

**核心问题**: 如何用更低的精度获得更高的吞吐，同时不损失模型质量？

#### 7.1 浮点格式: FP32 / FP16 / BF16 / TF32 / INT8 (3 页)
- 每种格式的范围、精度、硬件吞吐倍率
- TF32: NVIDIA 的 "免费午餐" —— 输入 FP32, 内部截断
- BF16 vs FP16: 为什么 LLM 训练偏爱 BF16

#### 7.2 自动混合精度 (AMP) (3 页)
- `torch.amp.autocast`: 哪些算子会降精度?
- `GradScaler`: 为什么需要 loss scaling?
- AMP 的 profiling: 对比 FP32 vs AMP 的算子耗时与内存

#### 7.3 量化: 训练后 vs 训练感知 (3 页)
- 动态量化: `quantize_dynamic()` → INT8 推理
- 静态量化: 校准集 + observer
- 量化感知训练 (QAT): `prepare_qat()` + `convert()`

#### 7.4 精度调试 (3 页)
- `detect_anomaly()`: 自动定位 NaN/Inf 根源
- `gradcheck()`: 数值导数 vs 自动微分
- 精度容忍度: `atol`, `rtol` 的工程选择

#### 7.5 实践与思考
- **Lab 26**: AMP profiling
- **Lab 27**: 动态量化对比
- **Lab 28-29**: 异常检测 + gradcheck

---

### Chapter 8: 微基准与系统性度量 (~14 页)

**核心问题**: 如何得到可重复、可信赖的性能数字？

#### 8.1 torch.utils.benchmark: 做对微基准 (3 页)
- `Timer`: 自动 warmup、多次采样、统计 median/IQR
- `Compare`: 多配置对比表
- 与 `time.time()` 的本质区别: 去噪和统计显著性

#### 8.2 多维度对比范式 (3 页)
- 按 shape 对比: 小矩阵 vs 大矩阵的性能拐点
- 按精度对比: FP32 vs FP16 vs INT8
- 按模式对比: eager vs compiled vs 手写 kernel

#### 8.3 Nsight Systems: GPU 全局时间线 (4 页)
- `nsys profile --stats=true`: 一行命令的完整分析
- 时间线解读: kernel overlap, idle gap, CPU/GPU 同步点
- API trace vs CUDA trace vs OS runtime trace 的选择
- 案例: 发现 DataLoader 成为全局瓶颈

#### 8.4 Nsight Compute: Kernel 级深度剖析 (4 页)
- `ncu --set full`: 硬件计数器采集
- 关键 metric: `sm__throughput`, `dram__throughput`, `occupancy`
- Section: Memory Workload, Compute Workload, Occupancy, Roofline
- 案例: 一个 kernel 从 30% throughput 优化到 75%

#### 8.5 实践与思考
- **Lab 13**: benchmark Timer
- **Lab 15**: DataLoader 瓶颈定位
- 综合练习: 用 nsys + ncu + profiler 对同一模型出三层报告

---

## Part III — 编译: 从 Python 到机器码 (~64 页)

> 全书的技术核心。完整拆解 PyTorch 2.0 的编译栈:
> TorchDynamo → AOTAutograd → Inductor → Triton → LLVM。
> 读者读完 Part III 后，应该能看懂 `TORCH_LOGS` 的每一行输出。

---

### Chapter 9: TorchDynamo — Python 字节码捕获 (~14 页)

**核心问题**: 如何在不改用户代码的前提下，把动态 Python 变成静态图？

#### 9.1 Dynamo 的设计哲学 (3 页)
- 为什么选择字节码级别的捕获 (vs AST / tracing)
- Guard 机制: 什么条件下缓存的图会失效?
- "尽力而为" vs "fullgraph": 容错与性能的 trade-off

#### 9.2 Graph Break: 图捕获的边界 (4 页)
- 什么会导致 graph break: data-dependent control flow, 不支持的 Python 特性
- `torch._dynamo.explain()`: 诊断报告解读
- `TORCH_LOGS="graph_breaks"`: 实时日志
- 工程实践: 如何改写代码消除 graph break

#### 9.3 FX Graph: Dynamo 的输出格式 (3 页)
- `torch.fx.Graph` 节点类型: placeholder / call_function / call_module / output
- graph 的实参: FakeTensor 与 shape specialization
- 从 FX graph 生成等价 Python 代码

#### 9.4 Dynamo 与后端的接口 (2 页)
- `torch.compile(backend=...)`: 可插拔后端架构
- 内置后端: inductor / eager / aot_eager / cudagraphs
- 自定义后端: 如何写一个最简 backend

#### 9.5 实践与思考 (2 页)
- **Lab 9**: eager vs compiled 对比
- **Lab 32**: 完整 graph break 诊断 + 三模式对比
- 练习: 对自己的模型跑 explain(), 消除所有 graph break

---

### Chapter 10: AOTAutograd 与 Inductor (~16 页)

**核心问题**: 有了计算图之后，如何把它变成高效的 GPU 代码？

#### 10.1 AOTAutograd: 编译时生成反向图 (4 页)
- 传统 autograd: 运行时构建反向图 → 每次都有开销
- AOTAutograd: 编译时在 FX graph 上做前向+反向展开
- `TORCH_LOGS="aot"`: 查看前向图和联合图
- functionalization: 去除 mutation, 变成纯函数式

#### 10.2 Inductor: 计算图到 Triton 的 lowering (4 页)
- Inductor 的 IR: LoopLevel IR → SchedulerNode
- 关键 lowering 规则: 哪些 aten op 变成哪些 Triton 模板
- pointwise fusion: 多个 element-wise 合并为一个 kernel
- reduction fusion: reduce + scale 的合并

#### 10.3 Inductor 的调度与优化 (4 页)
- Scheduler: 决定哪些 node 放在同一个 kernel
- Tiling 策略: Inductor 如何选择 tile 尺寸
- `TORCH_LOGS="output_code"`: 查看生成的 Python + Triton 代码
- 案例: 一个 4-op 融合 kernel 的完整生成过程

#### 10.4 Compile 模式: default vs reduce-overhead vs max-autotune (2 页)
- default: 平衡编译时间与运行性能
- reduce-overhead: CUDA Graph 封装, 减少 launch 开销
- max-autotune: Triton 做 coordinate descent tuning
- 三种模式的适用场景与量化对比

#### 10.5 实践与思考 (2 页)
- **Lab 32**: 三模式对比 + 融合率量化
- **Lab 33**: 生成代码审查
- 练习: 查看你模型的 `output_code`, 数一数生成了几个 Triton kernel

---

### Chapter 11: Triton — 面向 Tile 的 GPU 编程 (~18 页)

**核心问题**: Triton 如何让 "写高效 GPU kernel" 变得像写 NumPy 一样简单？

#### 11.1 Triton 的设计理念 (3 页)
- 为什么不直接生成 CUDA: 显式线程管理太复杂
- Tile-based 编程模型: 程序员操作 block-level 数据, 编译器管理线程
- `@triton.jit`: 一个 Python decorator 背后的编译管线

#### 11.2 Triton 编程模型 (4 页)
- `tl.load` / `tl.store`: tile 级内存访问
- `tl.dot`: tile-level 矩阵乘 → 映射到 Tensor Core
- `tl.reduce`: warp-level reduction
- `program_id` + `block_size`: 并行维度表达
- 完整示例: 手写一个 Triton matmul (BLOCK_M=128, BLOCK_K=64)

#### 11.3 Triton Compiler Pipeline (4 页)
- Triton IR (TTIR) → Triton GPU IR (TTGIR) → LLVM IR → PTX/AMDGPU
- 每层 IR 做什么优化: coalesce / pipeline / swizzle / prefetch
- `TRITON_CACHE_DIR` 查看编译产物
- 图: Triton compiler pipeline 全景图

#### 11.4 Auto-tuning: 配置空间搜索 (3 页)
- `triton.autotune` decorator: 搜索 tile_size / num_warps / num_stages
- coordinate descent vs grid search
- Inductor 的 `config.coordinate_descent_tuning=True` 如何调用 Triton autotuner

#### 11.5 Triton 与 Inductor 的关系 (2 页)
- Inductor 生成 Triton 源码 → Triton 编译到 PTX
- Inductor 的 Triton 模板: `triton_ops/` 里的 matmul, softmax 等
- 何时该手写 Triton kernel 替换 Inductor 自动生成

#### 11.6 实践与思考 (2 页)
- **Lab 33**: 查看并分析 Inductor 生成的 Triton kernel
- 练习: 用 `triton.testing.do_bench` 对比手写 vs 自动生成 kernel

---

### Chapter 12: LLVM 后端 — 从 IR 到机器码 (~16 页)

**核心问题**: Triton 生成的 LLVM IR 如何变成 GPU 能执行的二进制？

#### 12.1 LLVM IR 基础 (3 页)
- SSA 形式: 每个值只赋值一次
- 基本块与控制流图
- 函数 / intrinsic / metadata
- 阅读一段 Triton 生成的 `.llir` 文件

#### 12.2 NVPTX 后端: LLVM → PTX (4 页)
- PTX (Parallel Thread Execution): NVIDIA 的虚拟 ISA
- PTX 关键指令: `ld.global`, `st.shared`, `mma.sync`, `bar.sync`
- Register Allocation: 为什么寄存器压力影响 occupancy
- Spill: 寄存器溢出到 local memory 的性能代价

#### 12.3 从 PTX 到 SASS: 最后一层编译 (3 页)
- `ptxas`: NVIDIA 的 PTX assembler
- SASS: 实际执行的机器指令
- `cuobjdump -sass`: 查看真实指令
- 关键观察点: 指令吞吐量、shared memory bank conflict 模式

#### 12.4 为自定义硬件写 LLVM 后端 (4 页)
- Triton 的后端插件机制: `triton.compiler.backends`
- 定义 Target: 寄存器文件、内存层次、指令集
- TableGen: LLVM 的指令描述语言
- 最小可行后端: 能编译一个 elementwise kernel 到你的 ISA

#### 12.5 实践与思考 (2 页)
- **Lab 33**: 检查 Triton cache 中的 .llir / .ptx
- 练习: 用 `cuobjdump -sass` 分析一个 kernel 的寄存器用量
- 思考题: 如果你的硬件没有 Tensor Core, matmul 的 IR 会有什么不同?

---

## Part IV — 协同设计: 编译器驱动架构探索 (~52 页)

> 本书最具原创性的部分。传统教科书把架构设计和编译器分开讲，
> 本书论证: 用编译器产物 (Triton/LLVM IR) 驱动架构决策，
> 可以在 RTL 之前就量化回答 "SRAM 要多大? 寄存器要多深? ISA 缺什么指令?"

---

### Chapter 13: 负载画像 — 用数据而非直觉做架构 (~14 页)

**核心问题**: 芯片架构的每个参数，能否从真实 AI 负载中推导出来？

#### 13.1 为什么手工估算总是错 (2 页)
- 案例: 某 AI 芯片为 Conv 优化，结果 Transformer 时代来了
- 架构决策的 "输入" 应该是什么: 不是竞品规格，而是目标负载的编译产物

#### 13.2 算子频率与计算热点 (3 页)
- 从 profiler 数据统计: 哪些 aten op 占了 90% 的时间
- 跨模型对比: LLM vs ViT vs Diffusion 的算子频率差异
- 决策: 硬件该优先加速哪些操作？
- 代码: `lab38` 的算子频率分析部分

#### 13.3 Arithmetic Intensity 分布 (3 页)
- 不是一个 AI, 而是一个分布: P25 / P50 / P75
- 直方图: 典型模型的 AI 分布形状
- 从 AI 分布确定 Ridge Point → 定算力/带宽配比
- $\text{Balance Point} \approx \text{Median AI of target workload}$

#### 13.4 Shape 分布与 MAC Array 设计 (3 页)
- 统计 M/N/K 的实际范围和频率
- 如果 90% 的 M ∈ [128, 4096], MAC array 选多大?
- 小矩阵 (batch=1 推理) vs 大矩阵 (训练): 架构能否兼顾?

#### 13.5 精度分布与计算单元配比 (2 页)
- FP32 / BF16 / FP16 / INT8 各占多少计算时间?
- 决策: BF16 做 FP32 的 2x? 还是 4x? 要不要 FP64?

#### 13.6 实践与思考 (1 页)
- **Lab 38**: 完整架构级负载画像
- 练习: 对你关心的 3 个模型跑负载画像, 生成架构参数建议

---

### Chapter 14: 内存层次设计 — 从 Triton Tiling 到 SRAM Sizing (~14 页)

**核心问题**: SRAM 多大才够？DRAM 带宽需要多少？答案在 Triton kernel 里。

#### 14.1 Triton 的 Tiling 即内存需求声明 (3 页)
- `BLOCK_M`, `BLOCK_K`, `num_stages`: 每个参数的内存含义
- $\text{SRAM}_{min} = \text{TileM} \times \text{TileK} \times \text{dtype\_bytes} \times \text{num\_buffers}$
- Software Pipelining: num_stages 如何变成 SRAM 需求倍数

#### 14.2 SRAM 容量设计 (4 页)
- 从 Triton autotuner 的搜索空间看 tile 范围
- 单 kernel SRAM 需求 × SM 级并发数 = 每 SM 的 SRAM 容量
- SRAM 大小 vs Occupancy 的 trade-off
- 案例: 128KB vs 256KB SRAM 对 matmul 的 throughput 影响

#### 14.3 DRAM 带宽设计 (3 页)
- 从所有 kernel 的 bytes 总量 + 目标延迟 → 带宽需求
- HBM2 vs HBM2e vs HBM3: 带宽与成本
- 数据复用率 = Arithmetic Intensity: 高复用 → 可以用更低带宽

#### 14.4 Register File 深度 (3 页)
- LLVM IR 的 register pressure 分析
- 典型 Triton kernel 需要多少寄存器
- spill 对性能的量化影响
- 64 / 128 / 256 regs per thread 的选型依据

#### 14.5 实践与思考 (1 页)
- **Lab 33**: 查看 Triton tile 配置
- **Lab 38**: SRAM sizing 估算
- 练习: 从你模型的 Triton kernel 中推算 SRAM 需求

---

### Chapter 15: ISA 完备性 — 从 LLVM IR 需求到指令集设计 (~12 页)

**核心问题**: 你的 ISA 能不能编译并正确执行 Triton 生成的所有 kernel?

#### 15.1 从 LLVM IR 提取 ISA 需求 (3 页)
- 统计 Triton → LLVM IR 中出现的所有 intrinsic
- 分类: 计算 / 内存 / 同步 / 特殊函数
- 输出: ISA Sufficiency Checklist

#### 15.2 关键 ISA 特性逐项分析 (4 页)
- 矩阵计算: `mma.sync` / dot product 指令
- 内存操作: `ld.global`, `ld.shared`, `async_copy`, `prefetch`
- 同步: `barrier`, `atomic`, `warp_shuffle`
- 特殊函数: `exp`, `log`, `rsqrt` — 硬件 SFU vs 软件模拟
- 每个特性都附: 使用频率 + 缺失时的性能影响估算

#### 15.3 ISA 设计风险评估 (3 页)
- "缺失" vs "可模拟": 哪些缺失是致命的, 哪些可以绕过
- atomic_add(f32): 缺失会导致所有 reduce kernel 不正确
- async_copy: 缺失会导致无法做 software pipelining, 性能掉 30-50%
- warp_shuffle: 缺失会导致 cross-lane reduce 必须经过 shared memory

#### 15.4 实践与思考 (2 页)
- **Lab 35**: 从真实模型提取 golden + ISA 编译
- 练习: 列出你的 ISA 的 sufficiency checklist

---

### Chapter 16: 设计空间探索 (DSE) (~12 页)

**核心问题**: 在 (算力, 带宽, SRAM, 精度) 的多维空间中，哪个点是最优的？

#### 16.1 DSE 方法论 (3 页)
- 传统 DSE: spice-level simulation, 太慢
- Compiler-Driven DSE: 用 profiler 数据 + 解析模型, 快 1000x
- 核心指标: $\text{Utilization}(F, B) = \frac{1}{N}\sum \frac{\text{Achieved}_i}{\min(F, AI_i \times B)}$

#### 16.2 算力 × 带宽 二维扫描 (3 页)
- 固定其他参数, 扫描 (Peak FLOPS, Peak BW) 网格
- 画等高线图: 找到 Utilization 的 plateau
- 找到边际收益拐点: 再加资源也提不了多少

#### 16.3 Fusion 收益分析 (3 页)
- $\text{Fusion BW Saving} = \sum 2 \times \text{intermediate\_bytes}$
- 不同 fusion 粒度的收益曲线: 2-op vs 5-op vs mega kernel
- 决策: 硬件该支持多大的 fusion (影响 RF 和 SRAM 大小)

#### 16.4 Pareto 前沿与最终选型 (3 页)
- 在面积 vs 性能 vs 功耗的三维空间中画 Pareto 前沿
- 从 Pareto 前沿选一个点 → 架构参数确定
- 输出: 架构规格建议书

#### 16.5 实践与思考
- **Lab 38**: 完整 DSE 代码
- 练习: 对你的 workload 跑参数扫描，输出最优 (FLOPS, BW) 推荐

---

## Part V — 工程落地: 从仿真到流片 (~40 页)

> 把 Part I-IV 的理论和方法，落地到芯片研发的每个真实阶段。
> 每章对应芯片生命周期的一个节点，都有明确的交付物定义。

---

### Chapter 17: 硅前验证 — Golden 提取与模拟器对接 (~10 页)

**核心问题**: 在硬件模拟器上怎么跑真实模型的 kernel?

#### 17.1 Golden 数据提取 (3 页)
- Hook-based 方法: 每个子模块的输入/输出存为 `.pt` 文件
- manifest.json: 自动化管理 golden 数据
- 粒度选择: 整模型 / 子模块 / 单算子

#### 17.2 Triton → 虚拟 ISA 编译 (3 页)
- 自定义 Triton backend 的最小实现
- 从 `TORCH_LOGS="output_code"` 提取 Triton kernel 源码
- 编译到 LLVM IR → 你的汇编 → 模拟器可执行

#### 17.3 模拟器上的正确性验证 (2 页)
- golden vs 模拟器输出: `torch.allclose(atol, rtol)`
- 精度选择: FP32 golden + FP16 模拟器 → 需要放宽 tolerance
- 自动化: CI 中持续跑 golden 对比

#### 17.4 Pre-silicon 覆盖率度量 (2 页)
- 从 "跑了几个手写 case" 到 "跑了整个模型的所有 kernel"
- 按 aten op 分类统计: pass / fail / 精度不达标
- 输出: 正确性覆盖率报告

#### 实践
- **Lab 35**: 完整硅前 golden 提取流程

---

### Chapter 18: Bring-up — 从第一个 Kernel 到全模型 (~10 页)

**核心问题**: Day 0 硬件到手，如何最快速度验证功能并定位 bug?

#### 18.1 全模型冒烟测试 (2 页)
- 一行命令 `python run_analysis.py --stage bringup`
- `torch.compile` 自动生成 100+ kernel, 一次跑完
- 成功 → 进入性能优化; 失败 → 进入 bisect

#### 18.2 自动化 Bisect: 定位到具体 Kernel (3 页)
- 逐子模块 eager vs compiled 对比
- 严重度分级: OK / IMPRECISE / WRONG / CRASH
- 二分法缩小到最小失败子图

#### 18.3 从失败 Kernel 到硬件 Bug (3 页)
- 导出最小复现脚本: `repro_<module>.py`
- 用 `TORCH_LOGS="output_code"` 提取该 kernel 的 Triton 源码
- 在硬件上单独跑该 kernel → 定位到具体硬件行为
- 常见 bug 模式: barrier 语义、atomic 精度、shared memory bank conflict

#### 18.4 回归测试框架 (2 页)
- lab35 的 golden 自动转为回归 test suite
- `run_analysis.py --diff`: 跨版本对比, 检测新引入的 bug
- CI 集成: 每次 RTL 变更自动跑全量验证

#### 实践
- **Lab 36**: Bring-up bisect 全流程

---

### Chapter 19: 性能优化 — Roofline 驱动的系统调优 (~10 页)

**核心问题**: 已经功能正确了，怎么系统性地逼近硬件极限？

#### 19.1 建立性能基线 (2 页)
- Roofline 全模型扫描: 每层 kernel 的 (AI, throughput)
- 三模式 compile 对比: default / reduce-overhead / max-autotune
- GPU timeline 分析: idle 时间占比

#### 19.2 瓶颈决策树 (3 页)
- compute-bound → 检查 occupancy, warp divergence, register spill
- memory-bound → 检查 coalescing, cache hit rate, tiling
- launch-bound → CUDA Graph / reduce-overhead 模式
- 图: 完整决策树

#### 19.3 Triton Kernel 级优化 (3 页)
- 审查 Inductor 生成的 Triton 代码
- tile 尺寸调优: 过大 (spill) vs 过小 (occupancy)
- Software pipelining: num_stages 对不同 kernel 的影响
- 何时该手写 Triton kernel 替换自动生成

#### 19.4 优化报告与持续跟踪 (2 页)
- 对比优化前后的 Roofline 图
- 量化: throughput gap 从 X% 降到 Y%
- `run_analysis.py --diff`: 持续跟踪性能变化

#### 实践
- **Lab 32**: 三模式对比
- **Lab 34**: Roofline 分析

---

### Chapter 20: 生态建设 — 自研 Triton Backend 到模型覆盖 (~10 页)

**核心问题**: 如何从 "能跑一个 kernel" 扩展到 "能跑所有 PyTorch 模型"?

#### 20.1 Triton Backend 插件架构 (3 页)
- `triton.compiler.backends` 插件接口
- 最小实现: 能编译 elementwise → 你的 ISA
- 逐步扩展: elementwise → reduce → matmul → attention

#### 20.2 Inductor 对接 (2 页)
- `torch.compile(backend="inductor")` → 你的 Triton backend
- PrivateUse1 设备注册
- 端到端: PyTorch 模型 → Inductor → Triton → 你的硬件

#### 20.3 覆盖率追踪 (3 页)
- 模型动物园: 5+ 类模型的编译/运行测试
- aten op 覆盖率仪表板: 按类别追踪 pass/fail
- fusion pattern 兼容性: 哪些融合模式已支持

#### 20.4 从覆盖率到产品化 (2 页)
- 90% 覆盖率的里程碑意义
- 未覆盖 op 的 fallback 策略: CPU / graph break
- 长期维护: PyTorch 版本升级时的兼容性跟踪

#### 实践
- **Lab 37**: 覆盖率追踪全流程
- **run_analysis.py --stage ecosystem**

---

## 附录 (~20 页)

### Appendix A: 工具链安装与配置 (5 页)
- PyTorch, Triton, LLVM 的版本矩阵
- Nsight Systems / Nsight Compute 安装
- Docker 镜像 (可选)
- 常见安装问题排查

### Appendix B: TORCH_LOGS 参考手册 (4 页)
- 所有 log 选项: dynamo / aot / inductor / output_code / graph_breaks / ...
- 组合使用示例
- 日志输出解读速查

### Appendix C: 关键公式汇总 (3 页)
- Roofline: $\text{Attainable} = \min(F, AI \times B)$
- 硬件效率: $\eta = \text{Achieved} / \text{Peak}$
- Fusion 收益: $\text{Saving} = \sum 2 \times \text{intermediate\_bytes}$
- DSE Utilization: $U(F,B) = \frac{1}{N}\sum \frac{A_i}{\min(F, AI_i \times B)}$
- SRAM sizing: $S_{min} = M \times K \times \text{bytes} \times \text{buffers}$
- 含完整符号定义

### Appendix D: Lab 索引 (5 页)
- 全部 38 个 lab 按章节映射
- 每个 lab 的: 目标 / 预估时间 / 前置 lab / 关键 API
- `run_analysis.py` 完整参数参考

### Appendix E: 术语表 (3 页)
- 按字母排序的专业术语中英对照

---

## 页数预算

| 部分 | 章节 | 页数 | 占比 |
|------|------|------|------|
| 前言 | — | 8 | 3% |
| Part I: 基础 | Ch 1-4 | 60 | 20% |
| Part II: 观测 | Ch 5-8 | 56 | 19% |
| Part III: 编译 | Ch 9-12 | 64 | 21% |
| Part IV: 协同设计 | Ch 13-16 | 52 | 17% |
| Part V: 工程落地 | Ch 17-20 | 40 | 13% |
| 附录 | A-E | 20 | 7% |
| **合计** | | **300** | **100%** |

---

## Lab 到章节映射

| Lab | 章节 | Lab | 章节 |
|-----|------|-----|------|
| lab0 | Ch1 | lab20 | Ch6 |
| lab1-2 | Ch5 | lab21 | Ch3 |
| lab3 | Ch1,5 | lab22 | Ch6 |
| lab4 | Ch5,6 | lab23-24 | Ch6 |
| lab5 | Ch5 | lab25 | Ch8 |
| lab6 | Ch5 | lab26 | Ch7 |
| lab7 | Ch5 | lab27 | Ch7 |
| lab8 | Ch5 | lab28-29 | Ch7 |
| lab9 | Ch9 | lab30 | Ch3 |
| lab10 | Ch5 | lab31 | Ch3,9 |
| lab11 | App D | lab32 | Ch9,10 |
| lab12 | Ch5 | lab33 | Ch11,12 |
| lab13 | Ch8 | lab34 | Ch2,16 |
| lab14 | Ch2 | lab35 | Ch17 |
| lab15 | Ch8 | lab36 | Ch18 |
| lab16 | Ch5 | lab37 | Ch20 |
| lab17-19 | Ch6 | lab38 | Ch13,14 |

---

## 写作原则

1. **每章以 "核心问题" 开头** — 读者先知道为什么要读这章
2. **理论紧跟实验** — 每个公式/概念后都有代码演示或 lab 引用
3. **图 > 表 > 文字** — 架构图、流水线图、Roofline 图大量使用
4. **案例贯穿** — 用同一个 4 层 Transformer 从 Ch1 走到 Ch20
5. **可复现** — 书中所有数据都可通过 `run_analysis.py` 一键生成
6. **流片芯片数据出现在 Part IV-V** — 用真实硅验证数据替代假设
