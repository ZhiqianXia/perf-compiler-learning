"""实验 25：no_grad vs inference_mode — 推理模式对比

torch.inference_mode 比 torch.no_grad 更彻底，禁用所有 autograd 追踪。
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class MediumMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = f"self_{device}_time_total"

    model = MediumMLP().eval().to(device)
    x = torch.randn(256, 1024, device=device)
    n = 50

    # warmup
    for _ in range(5):
        model(x)

    # ---- 默认（开启 autograd）----
    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("DEFAULT"):
            for _ in range(n):
                model(x)

    print("=== 默认模式（autograd 开启）===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=5))

    # ---- torch.no_grad ----
    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("NO_GRAD"):
            with torch.no_grad():
                for _ in range(n):
                    model(x)

    print("\n=== torch.no_grad ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=5))

    # ---- torch.inference_mode ----
    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("INFERENCE_MODE"):
            with torch.inference_mode():
                for _ in range(n):
                    model(x)

    print("\n=== torch.inference_mode ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=5))

    print("\n对比要点:")
    print("  1. no_grad: 不记录梯度，但仍创建 autograd 元数据")
    print("  2. inference_mode: 完全禁用 autograd 内部簿记，更快、更省内存")
    print("  3. inference_mode 下的 tensor 不能用于后续 backward()")


if __name__ == "__main__":
    main()
