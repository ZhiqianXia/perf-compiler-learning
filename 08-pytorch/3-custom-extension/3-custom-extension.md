# 3. 自定义扩展与底层编程

- C++/CUDA 扩展
  - `torch.utils.cpp_extension.load`, `load_inline` 创建动态库。
  - `setup.py` / `pyproject.toml` 配置加速包。
  - `ATen` 的张量 API、`torch::Tensor` 与 `at::Tensor` 及 `TensorOptions`。
  - `TORCH_LIBRARY`、`TORCH_LIBRARY_IMPL` 注册自定义算子。
- Autograd 扩展
  - 自定义反向：继承 `torch.autograd.Function`。
  - Forward/backward 中避免使用 `torch.no_grad()`，在backward避免修改输入。
  - `SavedTensor` / `save_for_backward` 机制。
- TorchScript 与 JIT
  - `torch.jit.script` vs `torch.jit.trace`。
  - `torch.jit.freeze`, `optimize_for_inference`。
  - 执行模式：`torch.jit.save`, `torch.jit.load`。
  - Python/按需函数兼容性、类型约束。
- 设备兼容性与跨平台
  - ROCm、XLA、NPU/FPGA 部署方式差异。
  - 汇编级性能优化：register、warp-level primitives、shared memory。
  - PyTorch 中 `aten::` 手写Kernel 与运行时选择。