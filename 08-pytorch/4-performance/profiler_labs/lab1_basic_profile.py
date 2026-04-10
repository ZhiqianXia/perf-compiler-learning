"""实验 1：最简单的 profile — 看哪个算子最慢"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

# 简单 CNN 模型
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

# 准备模型和输入
model = SimpleCNN()
inputs = torch.randn(5, 3, 224, 224)

# profile 开始
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

# 打印结果：按 CPU 总耗时排序，看 top 10
print("=== 按 CPU total 排序（含子算子）===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 动手试：按 self CPU 排序 — 谁自身消耗最多
print("\n=== 按 Self CPU 排序（不含子算子）===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
