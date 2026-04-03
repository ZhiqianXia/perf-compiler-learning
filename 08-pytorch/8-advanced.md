# 8. 进阶专题

- 训练与精度
  - FP32 -> AMP -> BF16 -> INT8 的使用场景与风险。
  - 动态 loss scaling、梯度溢出/下溢检测。
- 新特性与编译器
  - `torch.compile`（TorchDynamo + AOTAutograd + Inductor）、`torch.fx`。
  - Lazy Tensor、`torch._dynamo`、`torch._inductor`。
- 自定义算子与融合
  - `fuser`、`TensorExpr`、`nvfuser`。
  - 图模式改写、形状传播、schedulers。
- 领域扩展
  - torchvision、torchaudio、torchtext、torchmetrics、pytorch-lightning。
  - ONNX、DeepSpeed、Accelerate、FairScale。
- 未来趋势
  - AOT/TFRT、编译器整合（MLIR）、自动混合精度拓展、通用算子图加速。