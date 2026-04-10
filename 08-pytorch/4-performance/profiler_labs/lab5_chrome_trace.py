"""实验 5：导出 Chrome Trace — 可视化时间线

运行后用 chrome://tracing 或 ui.perfetto.dev 打开生成的 trace.json
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

activities = [ProfilerActivity.CPU]
device = "cpu"
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)
    device = "cuda"
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

model = SimpleCNN().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

# Warmup
for _ in range(3):
    model(inputs)

with profile(activities=activities) as prof:
    model(inputs)

output_path = os.path.join(os.path.dirname(__file__), "trace.json")
prof.export_chrome_trace(output_path)
print(f"Trace 已导出到: {output_path}")
print("打开方式:")
print("  1. Chrome 浏览器 → 地址栏输入 chrome://tracing → Load")
print("  2. 或打开 https://ui.perfetto.dev → Open trace file")
