"""实验 12：record_function 嵌套标注 — 自定义 profiling 区域

在 trace 中标注自定义区域名称，方便定位"你的代码"而非底层算子。
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class TwoStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
        self.decoder = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        with record_function("ENCODER"):
            h = self.encoder(x)
        with record_function("DECODER"):
            out = self.decoder(h)
        return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = f"self_{device}_time_total"

    model = TwoStageModel().to(device)
    x = torch.randn(64, 256, device=device)

    # warmup
    for _ in range(3):
        model(x)

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("FULL_INFERENCE"):
            model(x)

    print("=== 自定义区域在 profiling 表中可见 ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    print("\n提示: 注意 FULL_INFERENCE / ENCODER / DECODER 这几行")


if __name__ == "__main__":
    main()
