"""实验 19：检测内存泄漏

常见泄漏模式：
  - 张量引用被无意保留（如存入 list）
  - 计算图未释放（忘记 .detach()）

本实验演示如何用 profiler + gc 发现泄漏。
"""

import gc
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        return self.fc(x)


def leaky_loop(model, device):
    """故意制造内存泄漏：把中间结果存进 list 不释放"""
    history = []
    for i in range(20):
        x = torch.randn(128, 256, device=device)
        out = model(x)
        # BUG: 保留了带 grad_fn 的 tensor，整个计算图不会释放
        history.append(out)
    return history


def clean_loop(model, device):
    """修复后的写法：detach 或不保留"""
    for i in range(20):
        x = torch.randn(128, 256, device=device)
        out = model(x)
        # 如果只需要数值，用 .detach() 切断计算图
        _ = out.detach().cpu()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyModel().to(device)

    # ---- 泄漏版 ----
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function("LEAKY_LOOP"):
            history = leaky_loop(model, device)

    print("=== 泄漏版: 内存分配 Top 5 ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=5))
    if device == "cuda":
        print(f"Peak CUDA mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    del history
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- 修复版 ----
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function("CLEAN_LOOP"):
            clean_loop(model, device)

    print("\n=== 修复版: 内存分配 Top 5 ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=5))
    if device == "cuda":
        print(f"Peak CUDA mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    print("\n关键: 对比两个版本的内存占用差异")
    print("  泄漏原因: 保留带 grad_fn 的 tensor 导致计算图不释放")
    print("  修复方法: .detach() 或 with torch.no_grad()")


if __name__ == "__main__":
    main()
