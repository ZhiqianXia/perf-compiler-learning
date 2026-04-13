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
