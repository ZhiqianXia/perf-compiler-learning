"""实验 23：CPU-GPU 同步开销

torch.cuda.synchronize() 会阻塞 CPU 直到所有 GPU 操作完成。
频繁同步是性能杀手 — 本实验量化这个开销。
"""

import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function


def main():
    if not torch.cuda.is_available():
        print("此实验需要 CUDA GPU。")
        print("核心 API: torch.cuda.synchronize()")
        print("  每次调用都会阻塞 CPU 等 GPU 完成所有 kernel。")
        return

    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    n_iters = 100

    # warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # ---- 不同步（正常流水线）----
    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    no_sync_time = time.perf_counter() - t0

    # ---- 每步同步 ----
    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.mm(a, b)
        torch.cuda.synchronize()  # 每步强制等待
    sync_time = time.perf_counter() - t0

    print(f"不同步 {n_iters} 次 matmul: {no_sync_time * 1000:.1f} ms")
    print(f"每步同步 {n_iters} 次 matmul: {sync_time * 1000:.1f} ms")
    print(f"同步开销: {(sync_time - no_sync_time) * 1000:.1f} ms  ({sync_time / no_sync_time:.1f}x 慢)")

    # ---- 用 profiler 看细节 ----
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities) as prof:
        with record_function("NO_SYNC"):
            for _ in range(10):
                torch.mm(a, b)
            torch.cuda.synchronize()

    print("\n=== 不同步 profile ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    with profile(activities=activities) as prof:
        with record_function("PER_STEP_SYNC"):
            for _ in range(10):
                torch.mm(a, b)
                torch.cuda.synchronize()

    print("\n=== 每步同步 profile ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    print("\n结论: 避免不必要的 synchronize()，让 CPU-GPU 流水线并行。")


if __name__ == "__main__":
    main()
