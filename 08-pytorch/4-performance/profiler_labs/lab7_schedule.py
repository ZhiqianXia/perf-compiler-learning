"""实验 7：长时间训练任务的 profile — schedule 机制

schedule 时间线:
  step:  0  1  2  3  4  5  6  7
         skip   W  H  A  A
         first  a  e  c  c
                i  a  t  t
                t  t  i  i
                   u  v  v
                   p  e  e
"""

import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule

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

model = SimpleCNN().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

output_dir = os.path.dirname(__file__)


def trace_handler(p):
    """每个 profiling cycle 结束时调用"""
    print(p.key_averages().table(sort_by=sort_key, row_limit=5))
    trace_path = os.path.join(output_dir, f"trace_step{p.step_num}.json")
    p.export_chrome_trace(trace_path)
    print(f"Trace 导出: {trace_path}")


with profile(
    activities=activities,
    schedule=torch.profiler.schedule(
        skip_first=2,  # 跳过前 2 步（初始化开销）
        wait=1,  # 空闲 1 步
        warmup=1,  # 热身 1 步（结果丢弃）
        active=2,  # 真正记录 2 步
        repeat=1,  # 只做 1 轮
    ),
    on_trace_ready=trace_handler,
) as prof:
    for step in range(8):
        print(f"Step {step}...")
        model(inputs)
        prof.step()  # 告诉 profiler "一步结束了"

print("Done.")
