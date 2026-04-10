# PyTorch Profiler 动手实验

按编号顺序逐个运行，每个脚本都是独立可执行的。

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

## 运行方式

```bash
cd profiler_labs
python lab0_env_check.py
python lab1_basic_profile.py
# ...依次运行
```

## 生成的文件

- `trace.json` — lab5 导出的 Chrome Trace
- `trace_step*.json` — lab7 schedule 导出的分步 trace
- `my_model_trace.json` — lab8 模板导出的 trace
- `eager_trace.json` / `compiled_trace.json` — lab9 对比 trace

用 `chrome://tracing` 或 [Perfetto UI](https://ui.perfetto.dev) 查看。
