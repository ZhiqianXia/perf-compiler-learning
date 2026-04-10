"""实验 14：FLOPs 统计 — 用 profiler 估算模型计算量

profiler 可以自动统计每个算子的 FLOPs（浮点运算次数）。
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU())
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model = SmallResNet().eval().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)

    for _ in range(3):
        model(x)

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,  # 关键参数
    ) as prof:
        with record_function("inference"):
            with torch.no_grad():
                model(x)

    events = prof.key_averages()

    print("=== FLOPs Top 10 ===")
    print(events.table(sort_by="flops", row_limit=10))

    total_flops = sum(e.flops for e in events if e.flops)
    print(f"\n总 FLOPs ≈ {total_flops:,.0f}  ({total_flops / 1e9:.2f} GFLOPs)")


if __name__ == "__main__":
    main()
