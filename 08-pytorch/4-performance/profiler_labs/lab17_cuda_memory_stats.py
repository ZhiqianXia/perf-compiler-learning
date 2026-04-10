"""实验 17：CUDA 显存统计 — torch.cuda.memory_stats

了解 PyTorch CUDA 缓存分配器的内部状态：
  active / reserved / allocated / freed 等。
"""

import torch
import torch.nn as nn


def fmt_bytes(b):
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.2f} MB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.2f} KB"
    return f"{b} B"


def print_mem_snapshot(tag: str):
    print(f"\n--- {tag} ---")
    print(f"  allocated : {fmt_bytes(torch.cuda.memory_allocated())}")
    print(f"  reserved  : {fmt_bytes(torch.cuda.memory_reserved())}")
    print(f"  max alloc : {fmt_bytes(torch.cuda.max_memory_allocated())}")


def main():
    if not torch.cuda.is_available():
        print("此实验需要 CUDA GPU。仅展示 API 用法。")
        print("  torch.cuda.memory_allocated()")
        print("  torch.cuda.memory_reserved()")
        print("  torch.cuda.memory_stats()")
        print("  torch.cuda.reset_peak_memory_stats()")
        return

    torch.cuda.reset_peak_memory_stats()

    print_mem_snapshot("初始状态")

    # 1. 分配一个大的 tensor
    a = torch.randn(1024, 1024, device="cuda")
    print_mem_snapshot("分配 1024x1024 float32")

    # 2. 模型前向
    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)).cuda()
    out = model(a)
    print_mem_snapshot("模型前向后")

    # 3. 删除中间变量
    del out, a
    torch.cuda.empty_cache()
    print_mem_snapshot("del + empty_cache 后")

    # 4. 详细统计
    stats = torch.cuda.memory_stats()
    print("\n=== 详细 memory_stats（部分）===")
    keys_of_interest = [
        "allocation.all.current",
        "allocation.all.peak",
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "num_alloc_retries",
        "num_ooms",
    ]
    for k in keys_of_interest:
        if k in stats:
            print(f"  {k:45s} = {stats[k]}")


if __name__ == "__main__":
    main()
