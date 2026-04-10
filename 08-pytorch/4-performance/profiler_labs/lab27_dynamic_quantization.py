"""实验 27：动态量化 profiling

将 Linear 层量化为 INT8，对比推理速度和模型大小。
"""

import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def model_size_mb(model):
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1e6


def main():
    # 动态量化只在 CPU 上支持
    model = MLP().eval()
    x = torch.randn(128, 1024)
    n = 100

    print(f"原始模型大小: {model_size_mb(model):.2f} MB")

    # 动态量化
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # 量化 Linear 层
        dtype=torch.qint8,
    )
    print(f"量化模型大小: {model_size_mb(quantized_model):.2f} MB")

    # warmup
    for _ in range(5):
        with torch.no_grad():
            model(x)
            quantized_model(x)

    # ---- 原始模型 profile ----
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("FP32_INFERENCE"):
            with torch.no_grad():
                for _ in range(n):
                    model(x)

    print("\n=== FP32 推理 ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))

    # ---- 量化模型 profile ----
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("INT8_INFERENCE"):
            with torch.no_grad():
                for _ in range(n):
                    quantized_model(x)

    print("\n=== INT8 量化推理 ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))

    # 输出对比
    with torch.no_grad():
        out_fp32 = model(x)
        out_int8 = quantized_model(x)
    diff = (out_fp32 - out_int8).abs().mean().item()
    print(f"\nFP32 vs INT8 平均输出差异: {diff:.6f}")
    print("结论: 动态量化可减小模型体积、加速 CPU 推理，精度损失通常很小。")


if __name__ == "__main__":
    main()
