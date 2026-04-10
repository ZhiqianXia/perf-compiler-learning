"""实验 8：完整 profile 模板 — 替换成你自己的模型

使用方法：将下面 === 替换这部分 === 中的内容改成你自己的模型和输入即可。
"""

import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

# 默认模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ====== 替换这部分 ======
model = SimpleCNN().eval()  # 换成你的模型
inputs = torch.randn(1, 3, 224, 224)  # 换成你的输入
# ========================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = inputs.to(device)

activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)

# Warmup（重要！消除首次运行的额外开销）
print("Warmup...")
for _ in range(3):
    with torch.no_grad():
        model(inputs)

# Profile
print("Profiling...")
with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("inference"):
        with torch.no_grad():
            model(inputs)

# 1. 耗时 Top 10
sort_key = f"self_{device}_time_total"
print("=== 耗时 Top 10 ===")
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

# 2. 内存 Top 10
print("\n=== 内存 Top 10 ===")
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# 3. 导出可视化
trace_path = os.path.join(os.path.dirname(__file__), "my_model_trace.json")
prof.export_chrome_trace(trace_path)
print(f"\n→ 用 chrome://tracing 或 ui.perfetto.dev 打开: {trace_path}")
