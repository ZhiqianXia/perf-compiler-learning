"""实验 16：profiler 事件列表编程 — 用 Python 分析每条事件

不依赖 table()，而是直接遍历 profiler 事件对象，做自定义分析。
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model = SimpleMLP().to(device)
    x = torch.randn(128, 256, device=device)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("MY_REGION"):
            model(x)

    events = prof.key_averages()

    # 1. 找出 CPU time 最大的 5 个事件
    sorted_by_cpu = sorted(events, key=lambda e: e.cpu_time_total, reverse=True)
    print("=== CPU time Top 5 ===")
    for e in sorted_by_cpu[:5]:
        print(f"  {e.key:40s} cpu={e.cpu_time_total:>10.0f}us  count={e.count}")

    # 2. 找出分配内存最多的事件
    sorted_by_mem = sorted(events, key=lambda e: e.cpu_memory_usage, reverse=True)
    print("\n=== Memory alloc Top 5 ===")
    for e in sorted_by_mem[:5]:
        print(f"  {e.key:40s} mem={e.cpu_memory_usage:>12.0f}B  count={e.count}")

    # 3. 汇总：总事件数、总 CPU 时间
    total_cpu_us = sum(e.self_cpu_time_total for e in events)
    print(f"\n总事件类型: {len(events)}, 总 self CPU 时间: {total_cpu_us}us")

    # 4. 导出到 list[dict] — 可以进一步存 csv / json
    print("\n=== 所有事件字段示例（第一条）===")
    e0 = events[0]
    print(f"  key={e0.key}")
    print(f"  count={e0.count}")
    print(f"  cpu_time_total={e0.cpu_time_total}")
    print(f"  self_cpu_time_total={e0.self_cpu_time_total}")
    print(f"  cpu_memory_usage={e0.cpu_memory_usage}")
    print(f"  self_cpu_memory_usage={e0.self_cpu_memory_usage}")
    if device == "cuda":
        print(f"  cuda_time_total={e0.cuda_time_total}")
        print(f"  self_cuda_time_total={e0.self_cuda_time_total}")


if __name__ == "__main__":
    main()
