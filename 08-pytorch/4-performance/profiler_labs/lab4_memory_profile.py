"""实验 4：内存分析 — 谁分配了最多内存"""

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

model = SimpleCNN()
inputs = torch.randn(5, 3, 224, 224)

with profile(
    activities=[ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True,
) as prof:
    model(inputs)

# 按 self memory 排序 — 谁直接分配了最多内存
print("=== 自身内存分配 Top 10 ===")
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# 按 total memory 排序 — 谁（含子算子）总共用了最多内存
print("\n=== 总内存占用 Top 10 ===")
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
