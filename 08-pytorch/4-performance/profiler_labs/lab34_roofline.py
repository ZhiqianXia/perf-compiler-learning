"""实验 34：Roofline 分析 — 基于 Profiler 数据构建硬件效率模型

利用 PyTorch Profiler 的 FLOPs + 内存数据，结合 GPU 硬件规格，
构建 Roofline Model，判断每个算子/模型整体是 compute-bound 还是 memory-bound。

输出:
  - 每层算子的 Arithmetic Intensity (AI) 和达到的性能
  - 与 GPU 理论峰值的差距
  - ASCII Roofline 图
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


# ---------- GPU 硬件规格 (根据你的 GPU 修改) ----------
# 常见 GPU 规格参考:
#   A100 SXM:   peak_flops_fp32=19.5 TFLOPS, peak_bw=2039 GB/s
#   A100 PCIe:  peak_flops_fp32=19.5 TFLOPS, peak_bw=1555 GB/s
#   V100:       peak_flops_fp32=15.7 TFLOPS, peak_bw=900 GB/s
#   RTX 4090:   peak_flops_fp32=82.6 TFLOPS, peak_bw=1008 GB/s
#   RTX 3090:   peak_flops_fp32=35.6 TFLOPS, peak_bw=936 GB/s
#   L40:        peak_flops_fp32=90.5 TFLOPS, peak_bw=864 GB/s

def get_gpu_specs():
    """自动检测 GPU 并返回估计的峰值规格"""
    if not torch.cuda.is_available():
        return {
            "name": "CPU (估计)",
            "peak_flops_tflops": 0.5,    # rough estimate
            "peak_bw_gbs": 50,           # DDR bandwidth
        }

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    # 估算 FP32 峰值 FLOPS:
    # peak = SM数 × 每SM FP32 cores × 2 (FMA) × 频率
    # 这是粗略估计，精确值需查 spec sheet
    clock_ghz = props.clock_rate / 1e6  # kHz → GHz
    # 假设每 SM ~128 FP32 cores (Ampere), 旧架构可能不同
    sm_count = props.multi_processor_count
    estimated_fp32_cores = sm_count * 128  # 粗略值
    peak_flops_tflops = estimated_fp32_cores * 2 * clock_ghz / 1000  # TFLOPS

    # 显存带宽: 从 properties 拿不到精确值，用总线宽度估算
    # total_memory 可以大致推算
    peak_bw_gbs = props.total_memory / 1e9 * 10  # 非常粗略的估计

    return {
        "name": name,
        "peak_flops_tflops": peak_flops_tflops,
        "peak_bw_gbs": peak_bw_gbs,
        "sm_count": sm_count,
        "clock_ghz": clock_ghz,
    }


# ---------- 模型 ----------
class ConvResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layer1 = self._make_block(64, 128)
        self.layer2 = self._make_block(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 100)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TransformerLM(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, vocab=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.embed(x)
        h = self.encoder(h)
        return self.head(h)


def profile_model(model, inputs, device, tag):
    """Profile 模型，收集 FLOPs 和内存数据"""
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    for _ in range(5):
        with torch.no_grad():
            model(inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with record_function(tag):
            with torch.no_grad():
                model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()

    return prof


def compute_roofline_data(prof, device):
    """从 profiler 事件中提取 Roofline 所需数据"""
    data = []
    for evt in prof.key_averages():
        flops = evt.flops or 0
        if flops == 0:
            continue

        # 耗时 (秒)
        time_s = (evt.self_cuda_time_total if device == "cuda" else evt.self_cpu_time_total) / 1e6
        if time_s <= 0:
            continue

        # 内存访问量 (bytes): self_cpu_memory_usage 是 net allocation, 不完全等同于 bytes accessed
        # 这里用一个启发式估计: 输入输出 tensor 的大小
        mem_bytes = abs(evt.self_cpu_memory_usage) + abs(getattr(evt, "self_cuda_memory_usage", 0))
        if mem_bytes == 0:
            # 粗略估计: FLOPs / AI_typical
            mem_bytes = flops / 10  # 假设 AI ≈ 10 作为后备

        achieved_flops = flops / time_s  # FLOPS
        ai = flops / mem_bytes if mem_bytes > 0 else 0  # FLOP/Byte

        data.append({
            "name": evt.key,
            "flops": flops,
            "time_s": time_s,
            "mem_bytes": mem_bytes,
            "achieved_tflops": achieved_flops / 1e12,
            "ai": ai,
        })

    return sorted(data, key=lambda d: d["flops"], reverse=True)


def print_roofline_table(data, specs):
    """打印 Roofline 分析表格"""
    peak_tflops = specs["peak_flops_tflops"]
    peak_bw = specs["peak_bw_gbs"]
    ridge_ai = peak_tflops * 1e3 / peak_bw  # FLOP/Byte at ridge point

    print(f"\nGPU: {specs['name']}")
    print(f"Peak FP32: {peak_tflops:.1f} TFLOPS | Peak BW: {peak_bw:.0f} GB/s | Ridge AI: {ridge_ai:.1f} FLOP/Byte")
    print()
    print(f"{'Operator':<40} {'GFLOPS':>8} {'Time(ms)':>10} {'AI':>8} {'Ach.TFLOPS':>12} {'%Peak':>8} {'Bound':>10}")
    print("-" * 100)

    for d in data[:20]:
        bound = "compute" if d["ai"] > ridge_ai else "memory"
        # 该 AI 下的 roofline 天花板
        roofline_limit = min(peak_tflops, d["ai"] * peak_bw / 1e3)
        pct_peak = d["achieved_tflops"] / roofline_limit * 100 if roofline_limit > 0 else 0

        name = d["name"][:39]
        print(f"{name:<40} {d['flops']/1e9:>8.1f} {d['time_s']*1000:>10.3f} "
              f"{d['ai']:>8.1f} {d['achieved_tflops']:>12.3f} {pct_peak:>7.1f}% {bound:>10}")


def ascii_roofline(data, specs, width=70, height=20):
    """打印 ASCII Roofline 图"""
    peak_tflops = specs["peak_flops_tflops"]
    peak_bw = specs["peak_bw_gbs"]

    print(f"\n{'='*60}")
    print("  ASCII Roofline Model")
    print(f"{'='*60}")

    # 确定绘图范围
    if not data:
        print("  (无 FLOPs 数据)")
        return

    ai_values = [d["ai"] for d in data if d["ai"] > 0]
    perf_values = [d["achieved_tflops"] for d in data]

    if not ai_values:
        print("  (无有效 AI 数据)")
        return

    import math
    ai_min = max(min(ai_values) * 0.1, 0.01)
    ai_max = max(max(ai_values) * 10, 100)
    perf_min = 0.001
    perf_max = peak_tflops * 2

    # Log scale
    def log_x(ai):
        return (math.log10(ai) - math.log10(ai_min)) / (math.log10(ai_max) - math.log10(ai_min))

    def log_y(perf):
        perf = max(perf, perf_min)
        return (math.log10(perf) - math.log10(perf_min)) / (math.log10(perf_max) - math.log10(perf_min))

    # 画布
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    # 画 Roofline 线
    for col in range(width):
        ai = ai_min * (ai_max / ai_min) ** (col / width)
        roofline = min(peak_tflops, ai * peak_bw / 1e3)
        row = int((1 - log_y(roofline)) * (height - 1))
        row = max(0, min(height - 1, row))
        canvas[row][col] = '-' if roofline >= peak_tflops else '/'

    # 画数据点
    markers = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    legend = []
    for i, d in enumerate(data[:min(len(data), len(markers))]):
        if d["ai"] <= 0:
            continue
        col = int(log_x(d["ai"]) * (width - 1))
        row = int((1 - log_y(d["achieved_tflops"])) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        canvas[row][col] = markers[i]
        legend.append(f"  {markers[i]} = {d['name'][:50]}")

    # 打印
    print(f"\n  TFLOPS (log) ^")
    for row in canvas:
        print(f"  {''.join(row)} |")
    print(f"  {'─' * width}> AI (FLOP/Byte, log)")
    print(f"  AI range: [{ai_min:.2f}, {ai_max:.0f}]")
    print(f"  Perf range: [{perf_min:.3f}, {perf_max:.1f}] TFLOPS")
    print(f"\n  Legend:")
    for l in legend[:15]:
        print(l)
    if len(legend) > 15:
        print(f"  ... and {len(legend)-15} more")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = get_gpu_specs()
    print(f"Device: {device}")
    print(f"GPU specs: {specs}")

    # ========== Model 1: CNN ==========
    print(f"\n{'='*60}")
    print("  Model 1: ConvResNet — Roofline Analysis")
    print(f"{'='*60}")

    model1 = ConvResNet().eval().to(device)
    x1 = torch.randn(16, 3, 224, 224, device=device)
    prof1 = profile_model(model1, x1, device, "ConvResNet")
    data1 = compute_roofline_data(prof1, device)
    print_roofline_table(data1, specs)
    ascii_roofline(data1, specs)

    # ========== Model 2: Transformer ==========
    print(f"\n{'='*60}")
    print("  Model 2: TransformerLM — Roofline Analysis")
    print(f"{'='*60}")

    model2 = TransformerLM().eval().to(device)
    x2 = torch.randint(0, 1000, (8, 256), device=device)
    prof2 = profile_model(model2, x2, device, "TransformerLM")
    data2 = compute_roofline_data(prof2, device)
    print_roofline_table(data2, specs)
    ascii_roofline(data2, specs)

    # ========== 对比 Eager vs Compiled ==========
    print(f"\n{'='*60}")
    print("  EagervsCompiled Roofline 对比")
    print(f"{'='*60}")

    torch._dynamo.reset()
    compiled2 = torch.compile(model2, mode="default")
    for _ in range(3):
        with torch.no_grad():
            compiled2(x2)

    prof2c = profile_model(compiled2, x2, device, "TransformerLM_compiled")
    data2c = compute_roofline_data(prof2c, device)

    print("\n--- Eager ---")
    print_roofline_table(data2, specs)
    print("\n--- Compiled ---")
    print_roofline_table(data2c, specs)

    # 汇总
    eager_total_flops = sum(d["flops"] for d in data2)
    eager_total_time = sum(d["time_s"] for d in data2)
    comp_total_flops = sum(d["flops"] for d in data2c)
    comp_total_time = sum(d["time_s"] for d in data2c)

    print(f"\n{'Mode':<20} {'Total GFLOPS':>15} {'Total Time(ms)':>15} {'Throughput TFLOPS':>20}")
    print("-" * 75)
    if eager_total_time > 0:
        print(f"{'Eager':<20} {eager_total_flops/1e9:>15.1f} {eager_total_time*1000:>15.3f} "
              f"{eager_total_flops/eager_total_time/1e12:>20.3f}")
    if comp_total_time > 0:
        print(f"{'Compiled':<20} {comp_total_flops/1e9:>15.1f} {comp_total_time*1000:>15.3f} "
              f"{comp_total_flops/comp_total_time/1e12:>20.3f}")

    # ========== 提示 ==========
    print(f"\n{'='*60}")
    print("  进阶: 用 Nsight Compute 获得精确 Roofline")
    print(f"{'='*60}")
    print("""
本实验使用 PyTorch Profiler 数据做近似 Roofline 分析。
更精确的分析需要硬件计数器:

1. Nsight Compute (NVIDIA):
   ncu --set roofline -o my_profile python my_script.py
   → 自动生成精确 Roofline 图

2. 手动采集关键 metrics:
   ncu --metrics \\
     sm__throughput.avg.pct_of_peak_sustained_elapsed, \\
     dram__throughput.avg.pct_of_peak_sustained_elapsed, \\
     gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \\
     python my_script.py

3. Nsight Systems 全局时间线:
   nsys profile --stats=true python my_script.py

4. AMD GPU (rocprof):
   rocprof --stats python my_script.py
""")


if __name__ == "__main__":
    main()
