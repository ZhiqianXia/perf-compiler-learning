"""实验 0：环境确认 — 确认 PyTorch 版本和 GPU 可用性"""

import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("将仅使用 CPU 进行 profiling")
