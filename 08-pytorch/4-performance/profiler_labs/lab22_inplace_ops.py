"""实验 22：in-place ops vs out-of-place — 对比内存与速度

in-place 操作（如 relu_、add_）可以减少内存分配，但不一定更快。
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = f"self_{device}_time_total"

    n = 50

    # ---- out-of-place ----
    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("OUT_OF_PLACE"):
            for _ in range(n):
                x = torch.randn(2048, 2048, device=device)
                y = torch.relu(x)       # 新分配
                y = y + 1.0              # 新分配
                y = y * 0.5             # 新分配

    print("=== out-of-place (relu, add, mul) ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=8))

    # ---- in-place ----
    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("IN_PLACE"):
            for _ in range(n):
                x = torch.randn(2048, 2048, device=device)
                x.relu_()               # 原地
                x.add_(1.0)             # 原地
                x.mul_(0.5)             # 原地

    print("\n=== in-place (relu_, add_, mul_) ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=8))

    print("\n对比要点:")
    print("  1. in-place 版本内存分配次数应更少")
    print("  2. 速度差异取决于 tensor 大小和 GPU 缓存命中")
    print("  3. in-place 不能用于需要保留原始值的 autograd 场景")


if __name__ == "__main__":
    main()
