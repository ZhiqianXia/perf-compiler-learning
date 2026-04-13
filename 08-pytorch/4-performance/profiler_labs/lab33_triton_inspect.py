"""实验 33：Triton / Inductor 生成代码检查

深入 torch.compile 的 Inductor 后端:
  1. 捕获并展示 Inductor 生成的 Triton kernel 源码
  2. 分析 kernel 的 tile size / num_warps / num_stages 配置
  3. 检查 Triton 缓存目录中的 .ttir / .llir / .ptx
  4. 对比不同 compile 模式生成的 kernel 差异
"""

import os
import sys
import glob
import tempfile
import torch
import torch.nn as nn


class FusionTarget(nn.Module):
    """一个容易被 Inductor 融合的模型片段"""
    def __init__(self, d=512):
        super().__init__()
        self.linear1 = nn.Linear(d, d * 4)
        self.linear2 = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        # linear → gelu → linear → residual + layernorm
        h = self.linear1(x)
        h = torch.nn.functional.gelu(h)
        h = self.linear2(h)
        return self.norm(x + h)


class MatMulChain(nn.Module):
    """矩阵乘法链，用于观察 tiling 策略"""
    def __init__(self, d=1024):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d, d) * 0.01)
        self.w2 = nn.Parameter(torch.randn(d, d) * 0.01)

    def forward(self, x):
        return torch.relu(x @ self.w1) @ self.w2


def inspect_triton_cache():
    """检查 Triton 缓存目录中的编译产物"""
    cache_dir = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    print(f"\nTriton cache dir: {cache_dir}")

    if not os.path.isdir(cache_dir):
        print("  (目录不存在，跳过)")
        return

    # 查找不同类型的编译产物
    for ext, desc in [(".ttir", "Triton IR"), (".llir", "LLVM IR"), (".ptx", "PTX"), (".cubin", "CUBIN")]:
        files = glob.glob(os.path.join(cache_dir, "**", f"*{ext}"), recursive=True)
        print(f"  {desc} ({ext}): {len(files)} files")
        if files:
            # 显示最新的一个文件的前几行
            newest = max(files, key=os.path.getmtime)
            size = os.path.getsize(newest)
            print(f"    最新: {newest} ({size} bytes)")
            if ext in (".ttir", ".llir", ".ptx"):
                try:
                    with open(newest, "r") as f:
                        lines = f.readlines()[:20]
                    print(f"    --- 前 {len(lines)} 行 ---")
                    for line in lines:
                        print(f"    {line.rstrip()}")
                    print(f"    --- (共 {sum(1 for _ in open(newest))} 行) ---")
                except Exception as e:
                    print(f"    读取失败: {e}")


def inspect_generated_code(model, inputs, tag="model"):
    """通过 Inductor 的内部 API 捕获生成的代码"""
    print(f"\n{'='*60}")
    print(f"  {tag}: Inductor 生成代码检查")
    print(f"{'='*60}")

    # 方法 1: 使用环境变量 (推荐通过命令行)
    print(f"\n提示: 运行以下命令查看完整生成代码:")
    print(f"  TORCH_LOGS='output_code' python {os.path.basename(__file__)}")
    print()

    # 方法 2: 通过 torch._inductor.config 控制
    try:
        import torch._inductor.config as inductor_config
        # 保存原始值
        orig_debug = inductor_config.debug
        orig_trace_dir = getattr(inductor_config, "trace.output_dir", None)

        # 设置 trace 输出目录
        trace_dir = os.path.join(os.path.dirname(__file__), "inductor_traces", tag)
        os.makedirs(trace_dir, exist_ok=True)
        inductor_config.debug = True
        inductor_config.trace.enabled = True

        print(f"Inductor trace 输出目录: {trace_dir}")
    except (ImportError, AttributeError) as e:
        print(f"  无法设置 inductor config: {e}")
        trace_dir = None

    # 编译并运行
    torch._dynamo.reset()
    compiled = torch.compile(model, mode="default")
    with torch.no_grad():
        for _ in range(3):
            compiled(inputs)

    # 检查是否有 Triton kernel 生成日志
    print(f"\n--- Dynamo 统计 ---")
    try:
        from torch._dynamo.utils import counters
        for key, val in counters.items():
            if val:
                print(f"  {key}: {dict(val)}")
    except ImportError:
        pass

    return compiled


def compare_compile_modes(model, inputs, device):
    """对比不同 compile 模式生成的 kernel 配置"""
    print(f"\n{'='*60}")
    print(f"  不同 compile 模式对比")
    print(f"{'='*60}")

    modes = ["default", "reduce-overhead", "max-autotune"]
    results = {}

    for mode in modes:
        torch._dynamo.reset()
        try:
            compiled = torch.compile(model, mode=mode)
            # Warmup & compile
            with torch.no_grad():
                for _ in range(3):
                    compiled(inputs)
                if device == "cuda":
                    torch.cuda.synchronize()

            # Benchmark
            import torch.utils.benchmark as benchmark
            timer = benchmark.Timer(
                stmt="compiled(inputs)",
                globals={"compiled": compiled, "inputs": inputs},
                label=f"compile_{mode}",
            )
            result = timer.blocked_autorange(min_run_time=1.0)
            results[mode] = result
            print(f"\n  {mode}: {result}")
        except Exception as e:
            print(f"\n  {mode}: 失败 - {e}")
            results[mode] = None

    # 对比
    if all(v is not None for v in results.values()):
        baseline = results["default"].median
        print(f"\n{'Mode':<25} {'Median(ms)':>12} {'vs default':>12}")
        print("-" * 50)
        for mode, r in results.items():
            ratio = r.median / baseline if baseline > 0 else 0
            print(f"{mode:<25} {r.median*1000:>12.3f} {ratio:>12.2f}x")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # ========== Part 1: FusionTarget 检查 ==========
    model1 = FusionTarget(d=512).eval().to(device)
    x1 = torch.randn(32, 128, 512, device=device)
    inspect_generated_code(model1, x1, tag="fusion_target")

    # ========== Part 2: MatMulChain 检查 ==========
    model2 = MatMulChain(d=1024).eval().to(device)
    x2 = torch.randn(32, 1024, device=device)
    inspect_generated_code(model2, x2, tag="matmul_chain")

    # ========== Part 3: 模式对比 ==========
    compare_compile_modes(model1, x1, device)

    # ========== Part 4: Triton Cache 检查 ==========
    inspect_triton_cache()

    # ========== 提示 ==========
    print(f"\n{'='*60}")
    print("  进阶检查命令")
    print(f"{'='*60}")
    print("""
1. 查看生成的 Triton kernel:
   TORCH_LOGS="output_code" python lab33_triton_inspect.py 2>&1 | grep -A 50 "@triton.jit"

2. 查看完整编译日志:
   TORCH_LOGS="dynamo,aot,inductor,output_code" python lab33_triton_inspect.py

3. 查看 LLVM IR (需要 Triton):
   ls ~/.triton/cache/**/*.llir

4. 查看 PTX:
   ls ~/.triton/cache/**/*.ptx
   cuobjdump -sass <cubin_file>  # 查看实际 SASS 指令

5. 控制 Inductor 行为:
   TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1   # 自动调优 tile 尺寸
   TORCHINDUCTOR_MAX_AUTOTUNE=1                 # 最大化自动调优
   TORCHINDUCTOR_BENCHMARK_KERNEL=1             # benchmark 每个 kernel
""")


if __name__ == "__main__":
    main()
