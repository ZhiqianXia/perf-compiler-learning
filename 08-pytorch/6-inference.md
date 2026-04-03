# 6. 部署与模型推理

- TorchScript
  - `torch.jit.script`/`torch.jit.trace` 导出模型。
  - `torch.jit.save`, `torch.jit.load`, `torch.jit.freeze`。
- ONNX
  - `torch.onnx.export`，保持动态轴、ospet版本兼容。
  - 运行时：ONNX Runtime、TensorRT、OpenVINO。
- TensorRT/TVM
  - `torch_tensorrt`、`torch2trt`、`torch-mlir`。
  - 关注kernel替换、序列优化、精度降级。
- 服务与可观测性
  - TorchServe、FastAPI + uvicorn + gRPC。
  - 批处理、延迟/吞吐度、cold start、内存复用。
- 推理优化
  - 短路行为、销毁无用变量、volatile优化、op fusion。