# 2. GPU/硬件适配

- 设备管理
  - `torch.cuda.is_available()`, `torch.cuda.device_count()`, `torch.cuda.current_device()`。
  - 设备切换：`device = torch.device('cuda:0')`, `tensor.to(device)`。
  - `pin_memory`, `non_blocking` 提高主机-设备传输性能。
- 内存管理
  - 查询：`torch.cuda.memory_allocated()`, `torch.cuda.memory_reserved()`。
  - 缓存：`torch.cuda.empty_cache()`。
  - 高峰值跟踪：`max_memory_allocated()`, `max_memory_reserved()`。
  - 内存碎片化观察与减少：使用 `torch.cuda.mem_get_info()`。
- 并行计算
  - Stream/事件：`torch.cuda.Stream()`, `torch.cuda.Event()`。
  - 异步拷贝：`copy_()`, `to(non_blocking=True)`、`torch.cuda.synchronize()`。
  - 多流：在模型与数据复制之间重叠计算。
- 修切与库优化
  - cuDNN：`torch.backends.cudnn.enabled`, `torch.backends.cudnn.benchmark`, `cudnn.deterministic`。
  - Tensor Core：对齐、`float16`, `bfloat16`, `mixed precision` 使用。
  - NCCL 后端：`torch.distributed` 的底层通信。
- 平台兼容
  - ROCm/Ampere/Volta 之间性能差异与特性。
  - 观察一个算子在不同 arch 的kernel选择和调度（`torch.backends.cuda.matmul.allow_tf32`）。

> 重点：GPU 适配不仅是代码层面，还要监控显存、流、通信、异步并行。