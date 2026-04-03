# 7. 关键生态与工具链

- 环境与版本
  - PyTorch 与 CUDA/cuDNN/NCCL 版本矩阵匹配。
  - `pip`, `conda`, `mamba`, Docker 镜像管理。
- 代码质量
  - linter：`flake8`, `pylint`; format：`black`; pre-commit。
  - 单测：`pytest`, `torch.testing.assert_allclose`。
- 日志与调试
  - `torch.utils.tensorboard`, `torch.utils.bottleneck`, `torch.autograd.set_detect_anomaly`。
  - Python 断点与异步事件、`faulthandler`。
- 社区资源
  - 官方文档、论坛、GitHub issue、RFC 讨论。
  - PyTorch 代码阅读（`aten/src/ATen`、`c10`）。