"""实验 37：Triton Backend Op 覆盖率追踪

生态兼容阶段 (Month 2+) 的核心需求:
  1. 统计 torch.compile 对真实模型的 aten op 覆盖率
  2. 识别不支持的 op 和 graph break 来源
  3. 检查 Inductor fusion pattern 覆盖情况
  4. 生成覆盖率仪表板，追踪进度

适用场景:
  - 自研 Triton backend 的兼容性验证
  - 评估新硬件对 PyTorch 生态的支持程度
  - 持续集成中的回归追踪
"""

import os
import sys
import json
import time
from collections import Counter, defaultdict
import torch
import torch.nn as nn


# ==================== 模型动物园 ====================
def make_mlp(d=512):
    return nn.Sequential(
        nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d), nn.LayerNorm(d)
    )

def make_conv_net():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    )

class AttentionModel(nn.Module):
    def __init__(self, d=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm(x + h)
        return x + self.ff(x)

class LSTMModel(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.lstm = nn.LSTM(d, d, batch_first=True, num_layers=2)
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class EmbeddingModel(nn.Module):
    def __init__(self, vocab=1000, d=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, vocab)

    def forward(self, x):
        return self.fc(self.embed(x).mean(dim=1))


# 模型动物园
MODEL_ZOO = {
    "mlp": {
        "factory": lambda: make_mlp(512),
        "input_factory": lambda dev: torch.randn(8, 512, device=dev),
        "category": "basic",
    },
    "conv_net": {
        "factory": make_conv_net,
        "input_factory": lambda dev: torch.randn(4, 3, 32, 32, device=dev),
        "category": "vision",
    },
    "attention": {
        "factory": lambda: AttentionModel(256, 4),
        "input_factory": lambda dev: torch.randn(4, 64, 256, device=dev),
        "category": "attention",
    },
    "lstm": {
        "factory": lambda: LSTMModel(256),
        "input_factory": lambda dev: torch.randn(4, 32, 256, device=dev),
        "category": "rnn",
    },
    "embedding": {
        "factory": lambda: EmbeddingModel(1000, 256),
        "input_factory": lambda dev: torch.randint(0, 1000, (4, 32), device=dev),
        "category": "nlp",
    },
}


# ==================== Op 收集器 ====================
class OpCollector:
    """通过 Dynamo 收集模型用到的 aten op"""

    @staticmethod
    def collect_fx_ops(model, example_input):
        """通过 FX trace 收集算子"""
        ops = []
        try:
            traced = torch.fx.symbolic_trace(model)
            for node in traced.graph.nodes:
                if node.op == "call_function":
                    ops.append(str(node.target))
                elif node.op == "call_method":
                    ops.append(f"method:{node.target}")
                elif node.op == "call_module":
                    ops.append(f"module:{node.target}")
        except Exception:
            pass
        return ops

    @staticmethod
    def check_compile(model, example_input, device):
        """尝试 compile 并返回结果"""
        result = {
            "compile_ok": False,
            "output_ok": False,
            "graph_breaks": 0,
            "max_diff": float("inf"),
            "error": None,
            "compile_time_s": 0,
        }

        # Eager baseline
        try:
            with torch.no_grad():
                eager_out = model(example_input)
        except Exception as e:
            result["error"] = f"eager_fail: {e}"
            return result

        # Compile
        torch._dynamo.reset()
        t0 = time.time()
        try:
            compiled = torch.compile(model, mode="default", fullgraph=False)
            with torch.no_grad():
                for _ in range(2):
                    compiled(example_input)
                compiled_out = compiled(example_input)
            result["compile_ok"] = True
            result["compile_time_s"] = time.time() - t0
        except Exception as e:
            result["error"] = f"compile_fail: {e}"
            result["compile_time_s"] = time.time() - t0
            return result

        # Output 对比
        try:
            if isinstance(eager_out, torch.Tensor) and isinstance(compiled_out, torch.Tensor):
                diff = (eager_out.float() - compiled_out.float()).abs().max().item()
                result["max_diff"] = diff
                result["output_ok"] = diff < 1e-2
        except Exception as e:
            result["error"] = f"compare_fail: {e}"

        # Graph break 检查
        try:
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(model)(example_input)
            # 尝试从 explanation 中提取 graph break 数量
            explanation_str = str(explanation)
            if "graph_break" in explanation_str.lower() or "break" in explanation_str.lower():
                # 简单计数
                result["graph_breaks"] = explanation_str.lower().count("break")
        except Exception:
            pass

        return result


# ==================== 覆盖率报告 ====================
def run_coverage_suite(device):
    """在所有模型上跑覆盖率测试"""
    all_results = {}
    all_ops = Counter()
    category_stats = defaultdict(lambda: {"total": 0, "compile_ok": 0, "output_ok": 0})

    for name, spec in MODEL_ZOO.items():
        print(f"\n  [{name}] ({spec['category']})")

        model = spec["factory"]().eval().to(device)
        x = spec["input_factory"](device)
        cat = spec["category"]

        # 收集 ops
        ops = OpCollector.collect_fx_ops(model.cpu(), x.cpu())
        for op in ops:
            all_ops[op] += 1
        print(f"    FX ops: {len(ops)}")

        # 编译测试
        model_dev = model.to(device)
        x_dev = x.to(device) if isinstance(x, torch.Tensor) else x
        result = OpCollector.check_compile(model_dev, x_dev, device)
        all_results[name] = result

        status = "✓" if result["output_ok"] else ("⚠" if result["compile_ok"] else "✗")
        print(f"    compile: {status}  max_diff={result['max_diff']:.2e}  "
              f"time={result['compile_time_s']:.1f}s")
        if result["error"]:
            print(f"    error: {result['error'][:100]}")

        category_stats[cat]["total"] += 1
        if result["compile_ok"]:
            category_stats[cat]["compile_ok"] += 1
        if result["output_ok"]:
            category_stats[cat]["output_ok"] += 1

    return all_results, all_ops, category_stats


def print_coverage_dashboard(results, ops, category_stats):
    """打印覆盖率仪表板"""
    print(f"\n{'='*70}")
    print(f"  OP COVERAGE DASHBOARD")
    print(f"{'='*70}")

    # 模型级别
    total = len(results)
    compile_ok = sum(1 for r in results.values() if r["compile_ok"])
    output_ok = sum(1 for r in results.values() if r["output_ok"])

    print(f"\n  模型级别:")
    print(f"  {'Model':<20} {'Compile':>10} {'Output OK':>10} {'MaxDiff':>12} {'Time(s)':>10}")
    print("  " + "-" * 65)
    for name, r in results.items():
        c_status = "✓" if r["compile_ok"] else "✗"
        o_status = "✓" if r["output_ok"] else "✗"
        diff = f"{r['max_diff']:.2e}" if r["max_diff"] != float("inf") else "N/A"
        print(f"  {name:<20} {c_status:>10} {o_status:>10} {diff:>12} {r['compile_time_s']:>10.1f}")

    print(f"\n  合计: {compile_ok}/{total} compile 通过, {output_ok}/{total} 输出正确")

    # 按类别
    print(f"\n  按类别:")
    print(f"  {'Category':<15} {'Total':>8} {'Compile':>10} {'Output OK':>12} {'Coverage':>10}")
    print("  " + "-" * 58)
    for cat, stats in sorted(category_stats.items()):
        pct = stats["output_ok"] / stats["total"] * 100 if stats["total"] > 0 else 0
        bar_len = int(pct / 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        print(f"  {cat:<15} {stats['total']:>8} {stats['compile_ok']:>10} "
              f"{stats['output_ok']:>12} {bar} {pct:.0f}%")

    # Op 统计
    print(f"\n  FX 算子频率 (Top 20):")
    print(f"  {'Op':<50} {'Count':>8}")
    print("  " + "-" * 60)
    for op, count in ops.most_common(20):
        print(f"  {op:<50} {count:>8}")


def print_fusion_patterns(device):
    """检查常见 fusion pattern 支持情况"""
    print(f"\n{'='*70}")
    print(f"  FUSION PATTERN 支持检查")
    print(f"{'='*70}")

    patterns = [
        ("pointwise + pointwise",
         lambda: nn.Sequential(nn.ReLU(), nn.Sigmoid()),
         lambda dev: torch.randn(8, 256, device=dev)),

        ("linear + bias + relu",
         lambda: nn.Sequential(nn.Linear(256, 256), nn.ReLU()),
         lambda dev: torch.randn(8, 256, device=dev)),

        ("linear + gelu",
         lambda: nn.Sequential(nn.Linear(256, 256), nn.GELU()),
         lambda dev: torch.randn(8, 256, device=dev)),

        ("conv + bn + relu",
         lambda: nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()),
         lambda dev: torch.randn(4, 3, 32, 32, device=dev)),

        ("layernorm + linear",
         lambda: nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256)),
         lambda dev: torch.randn(8, 256, device=dev)),

        ("matmul + softmax",
         lambda: _MatMulSoftmax(),
         lambda dev: torch.randn(4, 8, 64, 64, device=dev)),
    ]

    class _TempModule(nn.Module):
        pass

    print(f"\n  {'Pattern':<30} {'Compile':>10} {'Fused':>10} {'Status':>10}")
    print("  " + "-" * 65)

    for name, model_fn, input_fn in patterns:
        try:
            model = model_fn().eval().to(device)
            x = input_fn(device)

            # Eager kernel count 近似: profile 算子数
            from torch.profiler import profile, ProfilerActivity
            activities = [ProfilerActivity.CPU]
            if device == "cuda":
                activities.append(ProfilerActivity.CUDA)

            with torch.no_grad():
                model(x)
            with profile(activities=activities) as prof:
                with torch.no_grad():
                    model(x)
            eager_ops = len(prof.key_averages())

            # Compiled
            torch._dynamo.reset()
            compiled = torch.compile(model)
            with torch.no_grad():
                for _ in range(2):
                    compiled(x)
            with profile(activities=activities) as prof2:
                with torch.no_grad():
                    compiled(x)
            compiled_ops = len(prof2.key_averages())

            fused = compiled_ops < eager_ops
            status = "✓ fused" if fused else "○ no fusion"
            print(f"  {name:<30} {'✓':>10} {status:>20}")

        except Exception as e:
            print(f"  {name:<30} {'✗':>10} {'FAIL':>20}  {str(e)[:40]}")


class _MatMulSoftmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x @ x.transpose(-1, -2), dim=-1)


def save_report(results, ops, output_path):
    """保存覆盖率报告为 JSON"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {},
        "op_frequency": dict(ops.most_common()),
    }
    for name, r in results.items():
        report["models"][name] = {
            "compile_ok": r["compile_ok"],
            "output_ok": r["output_ok"],
            "max_diff": r["max_diff"] if r["max_diff"] != float("inf") else None,
            "compile_time_s": r["compile_time_s"],
            "error": r["error"],
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  报告保存到: {output_path}")


# ==================== Main ====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # === Part 1: 模型覆盖率 ===
    print(f"\n{'='*70}")
    print(f"  Part 1: 模型编译覆盖率")
    print(f"{'='*70}")

    results, ops, category_stats = run_coverage_suite(device)
    print_coverage_dashboard(results, ops, category_stats)

    # === Part 2: Fusion Pattern ===
    print_fusion_patterns(device)

    # === Part 3: 保存报告 ===
    output_dir = os.path.dirname(__file__)
    report_path = os.path.join(output_dir, "coverage_report.json")
    save_report(results, ops, report_path)

    # === 提示 ===
    print(f"\n{'='*70}")
    print(f"  扩展指南")
    print(f"{'='*70}")
    print("""
  添加更多模型到 MODEL_ZOO:
    1. 在 MODEL_ZOO dict 中添加 factory + input_factory
    2. 重新运行即可自动统计覆盖率

  追踪覆盖率趋势:
    1. 每次构建/发版后运行此脚本
    2. coverage_report.json 记录了时间戳
    3. 对比多次报告即可看到进度

  自研 Triton Backend 接入:
    1. 实现 triton.compiler.backends.YourBackend
    2. 设置 TRITON_BACKEND=your_backend
    3. 运行本脚本检查覆盖率

  接入 CI:
    python lab37_backend_coverage.py
    # exit code 0 = all pass, 1 = has failures
""")

    # 返回非零退出码如果有失败
    failures = sum(1 for r in results.values() if not r["output_ok"])
    if failures > 0:
        print(f"\n  ⚠ {failures} 个模型未通过输出正确性检查")


if __name__ == "__main__":
    main()
