# 1. PyTorch 基础（必须掌握）

- Tensor API
  - 常见类型：`torch.Tensor`, `torch.FloatTensor`, `torch.cuda.FloatTensor`
  - 创建：`torch.zeros`, `torch.ones`, `torch.arange`, `torch.randn`, `torch.empty`.
  - 索引/切片：`x[0]`, `x[:,1]`, `x[..., -1]`, `x.index_select`.
  - 维度操作：`view`, `reshape`, `permute`, `transpose`, `unsqueeze`, `squeeze`.
  - 广播规则、`dtype`转换（`float16`, `float32`, `int64`等）。
  - 设备：`x.to('cuda')`, `x.to(device)`, `torch.cuda.is_available()`。
- Autograd
  - `requires_grad`, 叶子张量与非叶子张量区别。
  - `backward()`, 收集梯度，`grad`属性。
  - `no_grad()`, `torch.set_grad_enabled(False)` 用于推理。
  - `detach()` 和 `detach_()`、梯度截断和梯度值检验。
  - 自定义函数：继承 `torch.autograd.Function`，定义 `forward()`/`backward()`。
- nn.Module
  - 构建模型：定义 `__init__()` 和 `forward()`。
  - 子模块注册：`self.add_module`, `nn.Sequential`。
  - 参数访问：`state_dict()`, `load_state_dict()`, `parameters()`。
- 训练循环
  - 数据：`Dataset`, `DataLoader`, `Sampler`, `BatchSampler`。
  - 优化：`torch.optim`（SGD、Adam、AdamW、RMSprop）。
  - 学习率调度：`LambdaLR`, `StepLR`, `CosineAnnealingLR`。
  - 损失函数：`nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.BCEWithLogitsLoss`。
  - 典型流程：零梯度、前向、损失、反向、优化、指标计算、梯度清空。
- 经典模块
  - `nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`, `nn.LayerNorm`。
  - 循环与Transformer：`nn.RNN`, `nn.LSTM`, `nn.GRU`, `nn.Transformer`, `nn.MultiheadAttention`。
  - 激活与正则：`ReLU`, `GELU`, `Dropout`, `LayerNorm`, `AdaptiveAvgPool`。

> 重点：通过小型例子快速跑通前向反向训练，关注梯度与设备移动。