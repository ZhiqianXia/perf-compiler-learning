"""实验 32：torch.compile 端到端诊断 — graph break + 生成代码 + profile 对比

完整覆盖 compile 链路:
  1. torch._dynamo.explain() 诊断 graph break
  2. TORCH_LOGS 查看 Inductor 生成的代码
  3. eager vs compiled vs max-autotune 三模式 profile 对比
  4. 算子融合率 & speedup 量化
"""

import os
import time
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


# ---------- 模型定义 ----------
class TransformerBlock(nn.Module):
    """单层 Transformer encoder block，贴近真实工作负载"""
    def __init__(self, d_model=256, nhead=8, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm Transformer
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x) + self.pos[:, :x.size(1), :]
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def count_kernels(prof):
    """统计 profile 中的 CUDA kernel 数量"""
    count = 0
    for evt in prof.key_averages():
        if evt.device_type is not None and "cuda" in str(evt.device_type).lower():
            count += evt.count
    return count


def profile_mode(tag, model, inputs, device):
    """Profile 并返回耗时与 kernel 统计"""
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = f"self_{device}_time_total"

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    # Profile
    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        with record_function(tag):
            with torch.no_grad():
                for _ in range(10):
                    model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    # 统计
    total_time = sum(e.self_cuda_time_total if device == "cuda" else e.self_cpu_time_total
                     for e in prof.key_averages())
    n_kernels = count_kernels(prof)
    total_flops = sum(e.flops for e in prof.key_averages() if e.flops)

    return {
        "tag": tag,
        "total_us": total_time,
        "n_kernels": n_kernels,
        "total_flops": total_flops,
        "prof": prof,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    model = SmallTransformer().eval().to(device)
    inputs = torch.randint(0, 1000, (4, 64), device=device)  # batch=4, seq_len=64

    # ========== Step 1: Graph Break 诊断 ==========
    print("\n" + "=" * 60)
    print("  Step 1: torch._dynamo.explain() — Graph Break 诊断")
    print("=" * 60)

    try:
        explanation = torch._dynamo.explain(model)(inputs)
        print(explanation)
    except Exception as e:
        print(f"explain() 失败 (可能版本不支持): {e}")

    # ========== Step 2: 三模式 Profile 对比 ==========
    print("\n" + "=" * 60)
    print("  Step 2: Eager vs Compiled vs Max-Autotune")
    print("=" * 60)

    # Mode 1: Eager
    stats_eager = profile_mode("Eager", model, inputs, device)

    # Mode 2: torch.compile (default)
    torch._dynamo.reset()
    compiled_default = torch.compile(model, mode="default")
    # 预热编译
    for _ in range(3):
        with torch.no_grad():
            compiled_default(inputs)
    stats_default = profile_mode("Compiled (default)", compiled_default, inputs, device)

    # Mode 3: torch.compile (max-autotune)
    torch._dynamo.reset()
    compiled_autotune = torch.compile(model, mode="max-autotune")
    for _ in range(3):
        with torch.no_grad():
            compiled_autotune(inputs)
    stats_autotune = profile_mode("Compiled (max-autotune)", compiled_autotune, inputs, device)

    # ========== Step 3: 量化对比 ==========
    print("\n" + "=" * 60)
    print("  Step 3: 量化对比汇总")
    print("=" * 60)

    all_stats = [stats_eager, stats_default, stats_autotune]
    baseline = stats_eager["total_us"] if stats_eager["total_us"] > 0 else 1

    print(f"\n{'Mode':<30} {'Time(us)':>12} {'Speedup':>10} {'Kernels':>10} {'GFLOPS':>10}")
    print("-" * 75)
    for s in all_stats:
        speedup = baseline / s["total_us"] if s["total_us"] > 0 else 0
        gflops = s["total_flops"] / 1e9 if s["total_flops"] else 0
        print(f"{s['tag']:<30} {s['total_us']:>12.0f} {speedup:>10.2f}x {s['n_kernels']:>10} {gflops:>10.2f}")

    if stats_eager["n_kernels"] > 0 and stats_default["n_kernels"] > 0:
        fusion_ratio = 1 - stats_default["n_kernels"] / stats_eager["n_kernels"]
        print(f"\n算子融合率 (default): {fusion_ratio:.1%}")

    # ========== Step 4: 导出 Trace ==========
    output_dir = os.path.dirname(__file__)
    for s in all_stats:
        tag_clean = s["tag"].replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(output_dir, f"trace_lab32_{tag_clean}.json")
        s["prof"].export_chrome_trace(path)
        print(f"Trace: {path}")

    # ========== 提示 ==========
    print("\n" + "=" * 60)
    print("  下一步分析建议")
    print("=" * 60)
    print("""
1. 查看 graph break:
   TORCH_LOGS="graph_breaks" python lab32_compile_e2e.py

2. 查看 Inductor 生成的 Triton 代码:
   TORCH_LOGS="output_code" python lab32_compile_e2e.py 2>&1 | head -200

3. 查看 AOTAutograd 生成的前向/反向图:
   TORCH_LOGS="aot" python lab32_compile_e2e.py

4. 用 Perfetto 对比 trace 文件:
   打开 https://ui.perfetto.dev 加载 trace_lab32_*.json
""")


if __name__ == "__main__":
    main()
