"""实验 6：看调用栈 — 定位到 Python 代码行号"""

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

model = SimpleCNN().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

sort_key = f"self_{device}_time_total"

# with_stack=True 会记录 Python 调用栈
with profile(
    activities=activities,
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    model(inputs)

# group_by_stack_n=5 表示按调用栈前 5 层分组
print(f"=== 按调用栈分组，{sort_key} 排序 ===")
print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_key, row_limit=5))
