"""实验 20：gradient checkpointing — 用计算换显存

对比开启 / 关闭 gradient checkpointing 的显存峰值。
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.profiler import profile, ProfilerActivity, record_function


class HeavyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class StackedModel(nn.Module):
    def __init__(self, dim=512, n_blocks=8, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([HeavyBlock(dim) for _ in range(n_blocks)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


def run_once(use_ckpt: bool, device: str):
    tag = "with_checkpoint" if use_ckpt else "no_checkpoint"
    model = StackedModel(dim=512, n_blocks=8, use_checkpoint=use_ckpt).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(64, 512, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function(tag):
            out = model(x)
            loss = out.sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    print(f"\n=== {tag} ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=5))
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak CUDA memory: {peak:.1f} MB")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_once(use_ckpt=False, device=device)
    run_once(use_ckpt=True, device=device)
    print("\n结论: gradient checkpointing 以额外前向计算为代价，显著降低显存峰值。")


if __name__ == "__main__":
    main()
