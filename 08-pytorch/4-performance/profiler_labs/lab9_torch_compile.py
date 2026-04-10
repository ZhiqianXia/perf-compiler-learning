"""实验 9：torch.compile 对比 — eager vs compiled 的 profile

对比两种模式下的算子耗时，观察 torch.compile 的算子融合效果。
"""

import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

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

device = "cuda" if torch.cuda.is_available() else "cpu"
activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)
sort_key = f"self_{device}_time_total"

model = SimpleCNN().eval().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

output_dir = os.path.dirname(__file__)

# ---------- Eager 模式 ----------
for _ in range(3):
    with torch.no_grad():
        model(inputs)

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        model(inputs)

print("=== Eager 模式 Top 10 ===")
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

eager_trace = os.path.join(output_dir, "eager_trace.json")
prof.export_chrome_trace(eager_trace)
print(f"Eager trace: {eager_trace}")

# ---------- Compiled 模式 ----------
compiled_model = torch.compile(model)

# 先跑几次让 compile 完成
print("\nCompiling...")
for _ in range(5):
    with torch.no_grad():
        compiled_model(inputs)

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        compiled_model(inputs)

print("\n=== Compiled 模式 Top 10 ===")
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))

compiled_trace = os.path.join(output_dir, "compiled_trace.json")
prof.export_chrome_trace(compiled_trace)
print(f"Compiled trace: {compiled_trace}")

print("\n→ 对比两个 trace 文件，观察 torch.compile 的算子融合效果")
