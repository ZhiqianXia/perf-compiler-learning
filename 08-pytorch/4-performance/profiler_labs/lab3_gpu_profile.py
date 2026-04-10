"""实验 3：GPU profiling — 同时抓 CPU 和 CUDA kernel"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

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

# 自动检测设备
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities.append(ProfilerActivity.CUDA)
    sort_key = "cuda_time_total"
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    sort_key = "cpu_time_total"
    print("未检测到 GPU，仅 profile CPU")

model = SimpleCNN().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

# Warmup（消除首次 CUDA 运行的额外开销）
for _ in range(3):
    model(inputs)

with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(f"=== 按 {sort_key} 排序 ===")
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
