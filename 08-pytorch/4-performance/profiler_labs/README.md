# PyTorch Profiler & Debug 动手实验

按编号顺序逐个运行，每个脚本都是独立可执行的。

## 基础 Profiling（lab0-11）

| 脚本 | 内容 | 关键 API |
|---|---|---|
| `lab0_env_check.py` | 环境确认 | `torch.cuda.is_available()` |
| `lab1_basic_profile.py` | 最基础的 profile：算子耗时排序 | `profile()`, `key_averages().table()` |
| `lab2_group_by_shape.py` | 按 tensor shape 分组 | `group_by_input_shape=True` |
| `lab3_gpu_profile.py` | GPU profiling：CPU + CUDA 同时抓 | `ProfilerActivity.CUDA` |
| `lab4_memory_profile.py` | 内存分析 | `profile_memory=True` |
| `lab5_chrome_trace.py` | 导出 Chrome Trace 可视化 | `export_chrome_trace()` |
| `lab6_stack_trace.py` | 调用栈追踪：定位源码行号 | `with_stack=True` |
| `lab7_schedule.py` | 长时间任务 schedule 机制 | `schedule()`, `prof.step()` |
| `lab8_full_template.py` | 完整模板：替换成自己的模型 | 综合 |
| `lab9_torch_compile.py` | eager vs compiled 对比 | `torch.compile()` |
| `lab10_tensorboard_trace.py` | 输出 TensorBoard 可视化日志 | `tensorboard_trace_handler()` |
| `lab11_export_onnx.py` | 导出 ONNX 并查看图结构 | `torch.onnx.export()` |

## Profiling 深入（lab12-16）

| 脚本 | 内容 | 关键 API |
|---|---|---|
| `lab12_record_function.py` | 自定义 profiling 区域标注 | `record_function()` 嵌套 |
| `lab13_benchmark_timer.py` | 微基准测试与多尺寸对比 | `torch.utils.benchmark.Timer` |
| `lab14_flops_count.py` | FLOPs 统计 | `with_flops=True` |
| `lab15_dataloader_bottleneck.py` | DataLoader 瓶颈定位 | `DataLoader` + `num_workers` |
| `lab16_profiler_events_export.py` | 编程式分析 profiler 事件 | `key_averages()` 遍历 |

## 内存调试（lab17-20）

| 脚本 | 内容 | 关键 API |
|---|---|---|
| `lab17_cuda_memory_stats.py` | CUDA 显存统计详解 | `torch.cuda.memory_stats()` |
| `lab18_memory_snapshot.py` | 显存快照可视化 | `memory._record_memory_history()` |
| `lab19_memory_leak_detect.py` | 检测内存泄漏（grad_fn 引用） | `gc` + `profile_memory` |
| `lab20_gradient_checkpoint.py` | gradient checkpointing 省显存 | `torch.utils.checkpoint` |

## 性能模式对比（lab21-25）

| 脚本 | 内容 | 关键 API |
|---|---|---|
| `lab21_contiguous_perf.py` | contiguous vs non-contiguous | `is_contiguous()`, `stride()` |
| `lab22_inplace_ops.py` | in-place vs out-of-place 算子 | `relu_()`, `add_()`, `mul_()` |
| `lab23_sync_overhead.py` | CPU-GPU 同步开销量化 | `torch.cuda.synchronize()` |
| `lab24_pin_memory.py` | pin_memory & non_blocking 传输 | `pin_memory`, `non_blocking` |
| `lab25_inference_mode.py` | no_grad vs inference_mode | `torch.inference_mode()` |

## 调试工具 & 混合精度（lab26-31）

| 脚本 | 内容 | 关键 API |
|---|---|---|
| `lab26_amp_profile.py` | AMP 混合精度 profiling | `torch.amp.autocast`, `GradScaler` |
| `lab27_dynamic_quantization.py` | 动态量化 INT8 对比 | `quantize_dynamic()` |
| `lab28_anomaly_detection.py` | 自动检测 NaN/Inf 来源 | `detect_anomaly()` |
| `lab29_gradcheck.py` | 梯度数值验证 | `gradcheck()`, `gradgradcheck()` |
| `lab30_model_hooks.py` | forward/backward hook 调试 | `register_forward_hook()` |
| `lab31_fx_graph.py` | torch.fx 符号追踪查看计算图 | `torch.fx.symbolic_trace()` |

## Compiler × Profiler 端到端分析（lab32-34）

| 脚本 | 内容 | 关键 API / 工具 |
|---|---|---|
| `lab32_compile_e2e.py` | 端到端 compile 诊断: graph break + 三模式 profile 对比 + 融合率 | `torch._dynamo.explain()`, `torch.compile(mode=...)` |
| `lab33_triton_inspect.py` | 查看 Inductor 生成的 Triton kernel，分析 tile/warp 配置 | `TORCH_LOGS="output_code"`, Triton cache |
| `lab34_roofline.py` | 基于 Profiler 数据构建 Roofline 分析，判断 compute/memory bound | `with_flops=True`, Roofline Model |

> 方法论文档: [METHODOLOGY.md](METHODOLOGY.md) — 覆盖 PyTorch.compile → Triton → LLVM → Arch 全链路分析方法轮

## 芯片研发全阶段落地（lab35-37）

| 脚本 | 内容 | 芯片阶段 | 关键 API / 工具 |
|---|---|---|---|
| `lab35_presilicon_extract.py` | 硅前 Golden 提取: 收集子模块输入输出 + FX 算子图 + Inductor 生成代码 | Pre-silicon | `torch.fx`, `torch.export`, golden 对比 |
| `lab36_bringup_bisect.py` | Bring-up 自动 bisect: 逐子模块定位 eager vs compiled 差异，生成复现脚本 | Day 0 | `torch.compile`, bisect, severity 分级 |
| `lab37_backend_coverage.py` | Triton Backend Op 覆盖率: 模型动物园编译测试 + fusion pattern 检查 | Month 2+ | 覆盖率仪表板, JSON 报告 |

## 架构设计负载画像（lab38）

| 脚本 | 内容 | 芯片阶段 | 关键分析 |
|---|---|---|---|
| `lab38_arch_workload_analysis.py` | Compiler-Driven Architecture Exploration: 算子频率 / AI 直方图 / Shape 分布 / SRAM sizing / 算力带宽 DSE / Fusion 收益 | 架构设计 | 输出架构参数建议书 |

## 自动化方法轮

| 脚本 | 功能 |
|---|---|
| `run_analysis.py` | **一键运行器**: 按芯片阶段编排 8 个 Analysis Pass，自动 JSON 报告 + 回归检测 + CI 模式 |
| `model_registry.py` | **模型注册表**: 添加自定义模型，所有分析自动覆盖 |

```bash
# 全量分析
python run_analysis.py

# 按阶段
python run_analysis.py --stage bringup
python run_analysis.py --stage perf
python run_analysis.py --stage arch

# CI 模式 (失败返回非零退出码)
python run_analysis.py --ci

# 对比上次结果，检测回归
python run_analysis.py --diff reports/report_20260412.json

# 查看可用模型和 pass
python run_analysis.py --list-models
python run_analysis.py --list-passes
```

## 运行方式

```bash
cd profiler_labs
python3 lab0_env_check.py
python3 lab1_basic_profile.py
# ...依次运行
```

如需查看 lab10 的 Profile 页面，确保安装：

```bash
python3 -m pip install --user torch-tb-profiler
```

lab11 需要安装 `onnx`（用于导出与校验）：

```bash
python3 -m pip install --user onnx
```

若你的 PyTorch 版本在导出时提示缺少 `onnxscript`，可再安装：

```bash
python3 -m pip install --user onnxscript
```

## 生成的文件

- `trace.json` — lab5 导出的 Chrome Trace
- `trace_step*.json` — lab7 schedule 导出的分步 trace
- `my_model_trace.json` — lab8 模板导出的 trace
- `eager_trace.json` / `compiled_trace.json` — lab9 对比 trace
- `tb_logs/` — lab10 生成的 TensorBoard profiler 日志目录
- `tiny_mlp.onnx` — lab11 导出的 ONNX 文件
- `memory_snapshot.pickle` — lab18 导出的显存快照

用 `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev) 查看 trace 文件。
lab10 使用 `tensorboard --logdir tb_logs --port 6006` 查看。
lab18 快照可上传到 https://pytorch.org/memory_viz 查看。

---

## 完整使用手册

### 一、环境准备

```bash
# 基础依赖 (PyTorch 2.0+)
pip install torch torchvision

# 可选依赖
pip install torch-tb-profiler   # lab10 TensorBoard 可视化
pip install onnx onnxscript     # lab11 ONNX 导出
pip install triton              # lab33 Triton 检查 (PyTorch 通常自带)
```

确认环境:
```bash
cd profiler_labs
python3 lab0_env_check.py
```

### 二、单个 Lab 逐步学习

按编号顺序逐个运行，每个脚本独立可执行:

```bash
# 基础 profiling
python3 lab1_basic_profile.py      # 算子耗时排序
python3 lab3_gpu_profile.py        # GPU profiling (需要 CUDA)
python3 lab4_memory_profile.py     # 内存分析
python3 lab5_chrome_trace.py       # 导出 trace → chrome://tracing 查看

# 性能对比
python3 lab9_torch_compile.py      # eager vs compiled
python3 lab13_benchmark_timer.py   # 微基准测试
python3 lab14_flops_count.py       # FLOPs 统计

# 混合精度 & 量化
python3 lab26_amp_profile.py       # AMP profiling
python3 lab27_dynamic_quantization.py  # INT8 量化对比
```

### 三、端到端编译链路分析

按 `PyTorch → Triton → LLVM → Arch` 链路逐层深入:

```bash
# Step 1: compile 诊断 (graph break + 三模式对比 + 融合率)
python3 lab32_compile_e2e.py

# Step 2: 查看 Inductor 生成的 Triton kernel
python3 lab33_triton_inspect.py
# 更详细:
TORCH_LOGS="output_code" python3 lab33_triton_inspect.py 2>&1 | head -200

# Step 3: Roofline 分析 (compute-bound vs memory-bound)
python3 lab34_roofline.py

# Step 4: 架构级负载画像
python3 lab38_arch_workload_analysis.py
```

查看编译器中间产物:
```bash
# Dynamo graph break 详情
TORCH_LOGS="graph_breaks" python3 lab32_compile_e2e.py

# AOTAutograd 前向/反向图
TORCH_LOGS="aot" python3 lab32_compile_e2e.py

# 完整编译日志
TORCH_LOGS="dynamo,aot,inductor,output_code" python3 lab33_triton_inspect.py

# Triton cache 中的 LLVM IR / PTX
ls ~/.triton/cache/**/*.llir
ls ~/.triton/cache/**/*.ptx
```

### 四、芯片研发阶段专用

```bash
# ── 架构设计 (RTL 开始之前) ──
python3 lab38_arch_workload_analysis.py
#   输出: 算子频率 / AI 分布 / Shape 分布 / SRAM sizing / DSE / Fusion 收益
#   交付物: 架构参数建议书

# ── 硅前验证 (Pre-silicon) ──
python3 lab35_presilicon_extract.py
#   输出: presilicon_golden/ 目录 (每个子模块的 input/output .pt)
#   用途: 在硬件模拟器上跑 golden 对比

# ── Bring-up (Day 0 硬件到手) ──
python3 lab36_bringup_bisect.py
#   输出: 逐子模块 PASS/FAIL 报告 + severity 分级
#   若失败: 自动导出 repro_<module>.py 复现脚本

# ── 生态兼容 (Month 2+) ──
python3 lab37_backend_coverage.py
#   输出: 覆盖率仪表板 + coverage_report.json
#   追踪: op 通过率 / fusion pattern 支持情况
```

### 五、自动化运行器（推荐用于日常迭代）

`run_analysis.py` 将所有分析 pass 编排为自动化 pipeline:

```bash
# ── 全量运行 (所有模型 × 所有 pass) ──
python3 run_analysis.py

# ── 按芯片阶段运行 ──
python3 run_analysis.py --stage arch        # 架构设计: 算子频率 + AI分布 + fusion
python3 run_analysis.py --stage presilicon  # 硅前: golden 提取
python3 run_analysis.py --stage bringup     # Bring-up: 正确性 + graph break + bisect
python3 run_analysis.py --stage perf        # 性能: speedup + roofline
python3 run_analysis.py --stage ecosystem   # 生态: 正确性 + graph break + fusion

# ── 指定模型 ──
python3 run_analysis.py --models transformer_lm
python3 run_analysis.py --models transformer_lm,vision_cnn

# ── 查看可用模型和 pass ──
python3 run_analysis.py --list-models
python3 run_analysis.py --list-passes
```

**回归检测** — 跟上次结果自动 diff:
```bash
# 第一次运行，生成基线
python3 run_analysis.py --stage perf
#   → reports/report_20260413_100000.json

# 修改编译器/硬件后再次运行
python3 run_analysis.py --stage perf --diff reports/report_20260413_100000.json
#   → 自动对比 speedup / correctness / kernel count
#   → 回归 > 10% 自动标红
```

**CI/CD 集成** — 失败时返回非零退出码:
```bash
python3 run_analysis.py --stage bringup --ci
# exit 0 = all pass
# exit 1 = has failures or regressions
```

### 六、添加自定义模型

编辑 `model_registry.py`，添加 `register_model()` 调用:

```python
# model_registry.py
from run_analysis import register_model

class MyLLM(nn.Module):
    ...

register_model(
    "my_llm",                    # 唯一名
    "transformer",                # 类别
    lambda: MyLLM(),              # 模型工厂
    lambda dev: torch.randint(0, 32000, (4, 512), device=dev),  # 输入工厂
    "My custom LLM",             # 描述
)
```

注册后所有分析自动覆盖，无需改其他文件:
```bash
python3 run_analysis.py --models my_llm --stage perf
```

### 七、分析 Pass 速查

| Pass | 阶段 | 自动做什么 | 关键输出 |
|------|------|-----------|---------|
| `op_frequency` | arch | 算子频率 + shape 分布 | top10 热点 ops |
| `fusion_benefit` | arch | eager vs compiled kernel 减少量 | kernel_reduction % |
| `roofline` | perf | FLOPs + AI 分布 | throughput, ai_p50 |
| `compile_speedup` | perf | 3 种 compile 模式耗时对比 | speedup × 3 |
| `compile_correctness` | bringup | eager vs compiled 数值对比 | max_diff |
| `graph_break` | bringup | Dynamo graph break 诊断 | break 数量 |
| `bisect` | bringup | 逐子模块定位差异 | issues + severity |
| `golden_extract` | presilicon | 保存子模块输入输出 | .pt files + manifest |

### 八、可视化工具

| 工具 | 用途 | 打开方式 |
|------|------|---------|
| Perfetto UI | 查看 Chrome Trace (`*_trace.json`) | https://ui.perfetto.dev |
| TensorBoard | lab10 profiler 可视化 | `tensorboard --logdir tb_logs` |
| PyTorch Memory Viz | lab18 显存快照 | https://pytorch.org/memory_viz |
| Nsight Systems | GPU 时间线 | `nsys profile python3 lab8_full_template.py` |
| Nsight Compute | Kernel 级 Roofline | `ncu --set roofline python3 lab34_roofline.py` |

### 九、报告目录

自动化运行产生的文件:

```
profiler_labs/
├── reports/                              # run_analysis.py 自动生成
│   ├── report_20260413_100000.json       # 每次运行的完整 JSON 报告
│   └── golden/                           # presilicon stage 的 golden 数据
│       └── transformer_lm/
│           ├── manifest.json
│           ├── root_input.pt
│           └── root_output.pt
├── coverage_report.json                  # lab37 的覆盖率报告
├── inductor_traces/                      # lab33 的 Inductor trace
├── *_trace.json                          # 各 lab 导出的 Chrome Trace
└── tb_logs/                              # lab10 TensorBoard 日志
```
