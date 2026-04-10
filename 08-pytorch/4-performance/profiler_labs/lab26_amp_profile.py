"""实验 26：AMP 混合精度 profiling

对比 FP32 训练 vs AMP（FP16/BF16）训练的算子耗时和内存。
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def train_step(model, x, y, opt, criterion, use_amp, scaler):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model(x)
        loss = criterion(out, y)
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    return loss.item()


def run_profile(tag, model, x, y, opt, criterion, use_amp, device):
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device == "cuda") else None
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = f"self_{device}_time_total"

    # warmup
    for _ in range(3):
        train_step(model, x, y, opt, criterion, use_amp, scaler)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with profile(activities=activities, profile_memory=True) as prof:
        with record_function(tag):
            for _ in range(5):
                train_step(model, x, y, opt, criterion, use_amp, scaler)

    print(f"\n=== {tag} ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=8))
    if device == "cuda":
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("此实验在 CPU 上也可运行，但 AMP 效果主要体现在 GPU。")

    x = torch.randn(32, 3, 64, 64, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    criterion = nn.CrossEntropyLoss()

    # FP32
    model_fp32 = ConvNet().to(device)
    opt_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
    run_profile("FP32", model_fp32, x, y, opt_fp32, criterion, use_amp=False, device=device)

    # AMP
    model_amp = ConvNet().to(device)
    opt_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)
    run_profile("AMP_FP16", model_amp, x, y, opt_amp, criterion, use_amp=True, device=device)

    print("\n对比要点:")
    print("  1. AMP 下 conv / matmul 会用 FP16 kernel，通常更快")
    print("  2. AMP 峰值显存通常更低（权重和激活占用减半）")
    print("  3. GradScaler 处理 loss scaling，防止 FP16 下溢")


if __name__ == "__main__":
    main()
