"""实验 38：架构设计负载画像 — Compiler-Driven Architecture Exploration

在芯片 RTL 开始之前，用真实 AI 模型的编译产物驱动架构参数决策。

本实验从 PyTorch 模型中提取:
  1. 算子频率分布 — 哪些 op 该做硬件加速
  2. 计算/访存比 (AI) 直方图 — 定算力/带宽配比
  3. Tensor shape 分布 — 定 MAC array / tile 尺寸
  4. 数据类型分布 — 定精度单元配比
  5. 寄存器压力估计 — 定 register file 深度
  6. SRAM sizing 估算 — 定 shared memory 容量
  7. Fusion 收益分析 — 定硬件 fusion 支持范围
  8. 计算单元 DSE (设计空间探索) — 扫描 (FLOPS, BW) 找最优点
"""

import os
import math
from collections import Counter, defaultdict
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


# ==================== Model Zoo ====================
class TransformerLM(nn.Module):
    """LLM-style model"""
    def __init__(self, vocab=2000, d=512, nhead=8, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        enc = nn.TransformerEncoderLayer(d, nhead, d * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.head(self.encoder(self.embed(x)))


class VisionModel(nn.Module):
    """CNN-style model"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(512, 100)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class MLPMixer(nn.Module):
    """MLP-heavy model"""
    def __init__(self, d=512, seq=64, layers=4):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(d),
                "mlp_token": nn.Sequential(nn.Linear(seq, seq * 4), nn.GELU(), nn.Linear(seq * 4, seq)),
                "norm2": nn.LayerNorm(d),
                "mlp_channel": nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)),
            }))
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            y = blk["norm1"](x).transpose(1, 2)
            y = blk["mlp_token"](y).transpose(1, 2)
            x = x + y
            x = x + blk["mlp_channel"](blk["norm2"](x))
        return self.head(x.mean(dim=1))


WORKLOADS = {
    "transformer_lm": {
        "model": lambda: TransformerLM(vocab=2000, d=512, nhead=8, layers=4),
        "input": lambda dev: torch.randint(0, 2000, (8, 128), device=dev),
        "desc": "LLM-style Transformer",
    },
    "vision_cnn": {
        "model": lambda: VisionModel(),
        "input": lambda dev: torch.randn(16, 3, 224, 224, device=dev),
        "desc": "CNN image classification",
    },
    "mlp_mixer": {
        "model": lambda: MLPMixer(d=512, seq=64, layers=4),
        "input": lambda dev: torch.randn(8, 64, 512, device=dev),
        "desc": "MLP-heavy mixer",
    },
}


# ==================== Analysis 1: Op Frequency ====================
def analyze_op_frequency(prof):
    """统计算子频率与时间占比"""
    op_time = defaultdict(float)
    op_count = defaultdict(int)
    total_time = 0

    for evt in prof.key_averages():
        t = evt.self_cuda_time_total if torch.cuda.is_available() else evt.self_cpu_time_total
        op_time[evt.key] += t
        op_count[evt.key] += evt.count
        total_time += t

    # 按时间排序
    sorted_ops = sorted(op_time.items(), key=lambda x: -x[1])
    return sorted_ops, op_count, total_time


# ==================== Analysis 2: AI Distribution ====================
def analyze_arithmetic_intensity(prof):
    """计算每个算子的 Arithmetic Intensity"""
    ai_data = []
    for evt in prof.key_averages():
        flops = evt.flops or 0
        if flops == 0:
            continue
        mem = abs(evt.self_cpu_memory_usage) + abs(getattr(evt, "self_cuda_memory_usage", 0))
        if mem == 0:
            mem = max(flops / 100, 1)  # fallback
        ai = flops / mem
        time_us = evt.self_cuda_time_total if torch.cuda.is_available() else evt.self_cpu_time_total
        ai_data.append({
            "name": evt.key,
            "flops": flops,
            "mem_bytes": mem,
            "ai": ai,
            "time_us": time_us,
        })
    return ai_data


# ==================== Analysis 3: Shape Distribution ====================
def analyze_shapes(prof):
    """从 profiler 事件中提取 tensor shape 分布"""
    shapes = []
    for evt in prof.key_averages():
        if evt.input_shapes:
            for shape in evt.input_shapes:
                if shape and len(shape) >= 2:
                    shapes.append(tuple(shape))
    return Counter(shapes)


# ==================== Analysis 4: SRAM Sizing ====================
def estimate_sram_requirements(ai_data):
    """基于 Triton tiling 惯例估算 SRAM 需求"""
    # Triton 常用 tile 尺寸
    common_tiles = [
        {"BLOCK_M": 64, "BLOCK_K": 32, "desc": "small tile"},
        {"BLOCK_M": 128, "BLOCK_K": 64, "desc": "medium tile"},
        {"BLOCK_M": 256, "BLOCK_K": 64, "desc": "large tile"},
    ]

    print("\n  SRAM Sizing 估算 (基于 Triton tiling 惯例):")
    print(f"  {'Tile Config':<30} {'Single Buffer':>15} {'Double Buffer':>15} {'Triple (3-stage)':>18}")
    print("  " + "-" * 80)

    for tile in common_tiles:
        m, k = tile["BLOCK_M"], tile["BLOCK_K"]
        # A tile (M×K) + B tile (K×N, N≈M) in FP16
        single = (m * k + k * m) * 2  # FP16 = 2 bytes
        double = single * 2
        triple = single * 3
        print(f"  {tile['desc']:12} (M={m},K={k})  "
              f"{single/1024:>12.1f} KB  {double/1024:>12.1f} KB  {triple/1024:>15.1f} KB")

    # 考虑 SM 级并发
    print(f"\n  SM 级并发需求 (假设每 SM 跑 N 个 tile):")
    base_sram = 128 * 64 * 2 * 2 * 3  # medium tile, double-buffer, 3-stage
    for concurrent in [1, 2, 4, 8]:
        total = base_sram * concurrent
        print(f"    {concurrent} concurrent tile → {total/1024:.0f} KB per SM")


# ==================== Analysis 5: Compute DSE ====================
def compute_dse(ai_data, device):
    """设计空间探索: 扫描不同 (Peak FLOPS, Peak BW) 找最优利用率"""
    if not ai_data:
        print("  (无 AI 数据, 跳过 DSE)")
        return

    flops_range = [5, 10, 20, 40, 80, 160]    # TFLOPS
    bw_range = [200, 500, 1000, 2000, 4000]     # GB/s

    print(f"\n  Compute DSE: 平均利用率 (%) 在不同 (FLOPS, BW) 配置下")
    print(f"  {'':>8}", end="")
    for bw in bw_range:
        print(f"  BW={bw:>4}GB/s", end="")
    print()
    print("  " + "-" * (8 + 14 * len(bw_range)))

    best_util = 0
    best_config = (0, 0)

    for flops_t in flops_range:
        print(f"  {flops_t:>4}TF  ", end="")
        for bw in bw_range:
            # 对每个 kernel 计算在此配置下能达到的峰值比例
            utils = []
            for d in ai_data:
                roofline = min(flops_t * 1e12, d["ai"] * bw * 1e9)
                achieved = d["flops"] / (d["time_us"] / 1e6) if d["time_us"] > 0 else 0
                util = min(achieved / roofline, 1.0) if roofline > 0 else 0
                utils.append(util)
            avg_util = sum(utils) / len(utils) * 100 if utils else 0
            print(f"  {avg_util:>10.1f}%  ", end="")
            if avg_util > best_util:
                best_util = avg_util
                best_config = (flops_t, bw)
        print()

    print(f"\n  最优配置: Peak={best_config[0]} TFLOPS, BW={best_config[1]} GB/s "
          f"(avg utilization={best_util:.1f}%)")
    print(f"  Ridge AI at optimal: {best_config[0]*1e3/best_config[1]:.1f} FLOP/Byte")

    return best_config


# ==================== Analysis 6: Fusion Benefit ====================
def analyze_fusion_benefit(model, example_input, device):
    """对比 eager vs compiled 的 kernel 数量和中间 tensor 节省"""
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Eager
    with torch.no_grad():
        model(example_input)
    with profile(activities=activities, profile_memory=True) as prof_eager:
        with torch.no_grad():
            model(example_input)

    eager_kernels = len(prof_eager.key_averages())
    eager_mem = sum(abs(e.self_cpu_memory_usage) for e in prof_eager.key_averages())

    # Compiled
    torch._dynamo.reset()
    compiled = torch.compile(model, mode="default")
    with torch.no_grad():
        for _ in range(3):
            compiled(example_input)

    with profile(activities=activities, profile_memory=True) as prof_compiled:
        with torch.no_grad():
            compiled(example_input)

    compiled_kernels = len(prof_compiled.key_averages())
    compiled_mem = sum(abs(e.self_cpu_memory_usage) for e in prof_compiled.key_averages())

    saved_mem = max(eager_mem - compiled_mem, 0)
    kernel_reduction = 1 - compiled_kernels / eager_kernels if eager_kernels > 0 else 0

    return {
        "eager_kernels": eager_kernels,
        "compiled_kernels": compiled_kernels,
        "kernel_reduction": kernel_reduction,
        "eager_mem_bytes": eager_mem,
        "compiled_mem_bytes": compiled_mem,
        "saved_mem_bytes": saved_mem,
    }


# ==================== Analysis 7: Dtype Distribution ====================
def analyze_dtype_distribution(model):
    """统计模型参数和 buffer 的数据类型分布"""
    dtype_counts = Counter()
    dtype_bytes = defaultdict(int)

    for name, param in model.named_parameters():
        dtype_counts[str(param.dtype)] += 1
        dtype_bytes[str(param.dtype)] += param.nelement() * param.element_size()

    for name, buf in model.named_buffers():
        dtype_counts[str(buf.dtype)] += 1
        dtype_bytes[str(buf.dtype)] += buf.nelement() * buf.element_size()

    return dtype_counts, dtype_bytes


# ==================== Main Report ====================
def run_workload_analysis(name, spec, device):
    """对单个 workload 执行完整分析"""
    print(f"\n{'='*70}")
    print(f"  Workload: {name} — {spec['desc']}")
    print(f"{'='*70}")

    model = spec["model"]().eval().to(device)
    x = spec["input"](device)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(x)

    # Profile
    with profile(activities=activities, record_shapes=True, with_flops=True, profile_memory=True) as prof:
        with record_function(name):
            with torch.no_grad():
                model(x)

    # --- 1. Op Frequency ---
    print(f"\n  ── 1. 算子频率分布 (Top 15) ──")
    sorted_ops, op_count, total_time = analyze_op_frequency(prof)
    cumulative = 0
    print(f"  {'Op':<45} {'Time(us)':>10} {'%Total':>8} {'Cumul%':>8} {'Count':>6}")
    print("  " + "-" * 80)
    for op, t in sorted_ops[:15]:
        pct = t / total_time * 100 if total_time > 0 else 0
        cumulative += pct
        print(f"  {op[:44]:<45} {t:>10.0f} {pct:>7.1f}% {cumulative:>7.1f}% {op_count[op]:>6}")

    # --- 2. AI Distribution ---
    print(f"\n  ── 2. Arithmetic Intensity 分布 ──")
    ai_data = analyze_arithmetic_intensity(prof)
    if ai_data:
        ais = [d["ai"] for d in ai_data]
        ai_sorted = sorted(ais)
        p25 = ai_sorted[len(ai_sorted) // 4] if len(ai_sorted) >= 4 else ai_sorted[0]
        p50 = ai_sorted[len(ai_sorted) // 2]
        p75 = ai_sorted[3 * len(ai_sorted) // 4] if len(ai_sorted) >= 4 else ai_sorted[-1]

        print(f"  有效 kernel 数: {len(ai_data)}")
        print(f"  AI 分位数: P25={p25:.1f}  P50(中位数)={p50:.1f}  P75={p75:.1f}")
        print(f"  AI 范围: [{min(ais):.1f}, {max(ais):.1f}] FLOP/Byte")

        # ASCII 直方图
        bins = [0, 1, 5, 10, 20, 50, 100, 500, float("inf")]
        bin_labels = ["<1", "1-5", "5-10", "10-20", "20-50", "50-100", "100-500", ">500"]
        hist = [0] * (len(bins) - 1)
        for ai in ais:
            for i in range(len(bins) - 1):
                if bins[i] <= ai < bins[i + 1]:
                    hist[i] += 1
                    break

        max_count = max(hist) if hist else 1
        print(f"\n  AI 直方图:")
        for i, label in enumerate(bin_labels):
            bar_len = int(hist[i] / max_count * 30) if max_count > 0 else 0
            print(f"    {label:>8} | {'█' * bar_len} {hist[i]}")
        print(f"\n  → 中位数 AI={p50:.1f} 意味着架构的 Ridge Point 应在此附近")

    # --- 3. Shape Distribution ---
    print(f"\n  ── 3. Tensor Shape 分布 (Top 10) ──")
    shapes = analyze_shapes(prof)
    for shape, count in shapes.most_common(10):
        print(f"    {str(shape):<40} ×{count}")
    if shapes:
        # 提取常见的 M, N, K 维度
        dims = []
        for shape, count in shapes.items():
            dims.extend(shape)
        dim_counter = Counter(dims)
        print(f"\n  常见维度值: {dim_counter.most_common(10)}")

    # --- 4. Dtype Distribution ---
    print(f"\n  ── 4. 数据类型分布 ──")
    dtype_counts, dtype_bytes_map = analyze_dtype_distribution(model)
    for dtype, count in dtype_counts.most_common():
        mb = dtype_bytes_map[dtype] / 1e6
        print(f"    {dtype:<20} {count:>5} params/buffers  {mb:>10.2f} MB")

    # --- 5. SRAM Sizing ---
    print(f"\n  ── 5. SRAM Sizing 估算 ──")
    estimate_sram_requirements(ai_data)

    # --- 6. Fusion Benefit ---
    print(f"\n  ── 6. Fusion 收益分析 ──")
    fusion = analyze_fusion_benefit(model, x, device)
    print(f"    Eager kernels:    {fusion['eager_kernels']}")
    print(f"    Compiled kernels: {fusion['compiled_kernels']}")
    print(f"    Kernel 减少:      {fusion['kernel_reduction']:.1%}")
    print(f"    内存节省估计:     {fusion['saved_mem_bytes']/1024:.1f} KB")

    return ai_data, fusion


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    all_ai_data = []
    all_fusion = {}

    for name, spec in WORKLOADS.items():
        ai_data, fusion = run_workload_analysis(name, spec, device)
        all_ai_data.extend(ai_data)
        all_fusion[name] = fusion

    # ==================== 跨 Workload 汇总 ====================
    print(f"\n{'='*70}")
    print(f"  跨 Workload 汇总 — 架构设计建议")
    print(f"{'='*70}")

    # 汇总 AI
    if all_ai_data:
        all_ais = sorted([d["ai"] for d in all_ai_data])
        global_p50 = all_ais[len(all_ais) // 2]
        print(f"\n  全局 AI 中位数: {global_p50:.1f} FLOP/Byte")
        print(f"  → 建议设计 Ridge Point ≈ {global_p50:.0f} FLOP/Byte")

        # DSE
        print(f"\n  ── 7. 计算单元设计空间探索 (DSE) ──")
        best = compute_dse(all_ai_data, device)

    # 汇总 Fusion
    print(f"\n  ── 8. Fusion 收益汇总 ──")
    print(f"  {'Workload':<25} {'Eager':>8} {'Compiled':>10} {'Reduction':>12} {'Saved KB':>10}")
    print("  " + "-" * 68)
    for name, f in all_fusion.items():
        print(f"  {name:<25} {f['eager_kernels']:>8} {f['compiled_kernels']:>10} "
              f"{f['kernel_reduction']:>11.1%} {f['saved_mem_bytes']/1024:>10.1f}")
    avg_reduction = sum(f["kernel_reduction"] for f in all_fusion.values()) / len(all_fusion)
    print(f"\n  平均 kernel 融合率: {avg_reduction:.1%}")
    print(f"  → fusion 减少的 kernel launch 可省去 {avg_reduction*100:.0f}% 的调度开销")

    # ==================== 架构参数建议汇总 ====================
    print(f"\n{'='*70}")
    print(f"  架构参数建议 (基于 {len(WORKLOADS)} 个 workload)")
    print(f"{'='*70}")
    print(f"""
  ┌────────────────────────┬───────────────────────────────────────────┐
  │ 架构参数               │ 建议值 (数据驱动)                         │
  ├────────────────────────┼───────────────────────────────────────────┤
  │ 算力/带宽配比          │ Ridge AI ≈ {global_p50:.0f} FLOP/Byte              │
  │ SRAM per SM            │ ≥ 128 KB (3-stage pipeline, medium tile)  │
  │ Register File          │ ≥ 128 regs/thread (避免 spill)           │
  │ 精度单元               │ FP32 + BF16 (BF16 throughput = 2× FP32)  │
  │ 向量宽度               │ ≥ 256 bit                                │
  │ Fusion 硬件支持        │ ≥ 2-op fusion (kernel 减少 {avg_reduction:.0%})       │
  │ Warp/Wave size         │ 32 (兼容 Triton 生态)                    │
  │ Barrier 语义           │ block-level + warp-level 均需             │
  │ Atomic                 │ f32 + i32 + 可选 f16                     │
  └────────────────────────┴───────────────────────────────────────────┘
""" if all_ai_data else "  (无有效 AI 数据)")

    print("""
  下一步:
    1. 扩大 Model Zoo: 加入 LLaMA, Stable Diffusion, Whisper 等
    2. 用 TORCH_LOGS="output_code" 提取 Triton kernel → 统计 LLVM IR 指令频率
    3. 建立参数化模型: Utilization(FLOPS, BW, SRAM) → 找 Pareto 最优
    4. 将本实验数据导入架构模拟器做 cycle-level 验证
""")


if __name__ == "__main__":
    main()
