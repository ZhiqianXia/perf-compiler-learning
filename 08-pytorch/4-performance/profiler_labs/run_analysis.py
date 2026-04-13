"""自动化方法轮运行器 — 一键跑完全链路分析并生成报告

将 lab32-lab38 的分析能力编排成自动化 pipeline:
  - 注册模型一次，所有分析自动跑完
  - 统一 JSON 报告 + 终端摘要
  - 支持按芯片阶段选择运行哪些分析
  - 支持与上次结果自动 diff (回归检测)
  - 可直接集成到 CI/CD

用法:
  # 全量运行
  python run_analysis.py

  # 只跑某个阶段
  python run_analysis.py --stage arch
  python run_analysis.py --stage presilicon
  python run_analysis.py --stage bringup
  python run_analysis.py --stage perf
  python run_analysis.py --stage ecosystem

  # 指定模型
  python run_analysis.py --models transformer_lm,vision_cnn

  # 与上次结果对比
  python run_analysis.py --diff reports/report_20260412.json

  # CI 模式 (失败返回非零退出码)
  python run_analysis.py --ci
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


# ============================================================
# 1. Model Registry — 注册一次，全部分析自动跑
# ============================================================

@dataclass
class ModelSpec:
    """模型注册规格"""
    name: str
    category: str  # transformer / vision / mlp / rnn / ...
    factory: Callable  # -> nn.Module
    input_factory: Callable  # (device) -> Tensor
    description: str = ""
    training: bool = False  # 是否也测训练


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name, category, factory, input_factory, description="", training=False):
    """注册模型到全局 registry"""
    _MODEL_REGISTRY[name] = ModelSpec(
        name=name, category=category, factory=factory,
        input_factory=input_factory, description=description, training=training,
    )


def get_registry():
    return _MODEL_REGISTRY


# ---------- 内置模型 ----------
class _TransformerLM(nn.Module):
    def __init__(self, vocab=2000, d=512, nhead=8, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        enc = nn.TransformerEncoderLayer(d, nhead, d * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, vocab)
    def forward(self, x):
        return self.head(self.encoder(self.embed(x)))

class _VisionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, 100)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))

class _MLP(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d), nn.LayerNorm(d),
        )
    def forward(self, x):
        return self.net(x)

class _AttentionBlock(nn.Module):
    def __init__(self, d=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm(x + h)
        return x + self.ff(x)


def _register_builtins():
    register_model("transformer_lm", "transformer",
                    lambda: _TransformerLM(), lambda dev: torch.randint(0, 2000, (8, 128), device=dev),
                    "4-layer Transformer LM")
    register_model("vision_cnn", "vision",
                    lambda: _VisionCNN(), lambda dev: torch.randn(8, 3, 224, 224, device=dev),
                    "3-block CNN")
    register_model("mlp", "mlp",
                    lambda: _MLP(), lambda dev: torch.randn(8, 512, device=dev),
                    "Simple MLP")
    register_model("attention", "attention",
                    lambda: _AttentionBlock(), lambda dev: torch.randn(4, 64, 256, device=dev),
                    "Single attention block")

_register_builtins()

# 自动加载 model_registry.py 中的自定义模型
try:
    import model_registry  # noqa: F401
except ImportError:
    pass


# ============================================================
# 2. Analysis Passes — 每个 pass 独立可组合
# ============================================================

@dataclass
class AnalysisResult:
    """单个分析 pass 的结果"""
    pass_name: str
    model_name: str
    status: str  # "pass" / "fail" / "warn" / "skip"
    metrics: dict = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0


class AnalysisPass:
    """分析 pass 基类"""
    name: str = "base"
    stage: str = "all"  # arch / presilicon / bringup / perf / ecosystem

    def run(self, model, inputs, device, spec) -> AnalysisResult:
        raise NotImplementedError


class CompileCorrectnessPass(AnalysisPass):
    """编译正确性: eager vs compiled 输出对比"""
    name = "compile_correctness"
    stage = "bringup"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        try:
            with torch.no_grad():
                eager_out = model(inputs)

            torch._dynamo.reset()
            compiled = torch.compile(model, mode="default", fullgraph=False)
            with torch.no_grad():
                for _ in range(2):
                    compiled(inputs)
                compiled_out = compiled(inputs)

            if isinstance(eager_out, torch.Tensor) and isinstance(compiled_out, torch.Tensor):
                max_diff = (eager_out.float() - compiled_out.float()).abs().max().item()
                result.metrics["max_diff"] = max_diff
                result.metrics["output_match"] = max_diff < 1e-2
                if max_diff >= 1e-2:
                    result.status = "fail"
                    result.error = f"output mismatch: max_diff={max_diff:.2e}"
            else:
                result.metrics["output_match"] = True  # non-tensor, skip

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class GraphBreakPass(AnalysisPass):
    """Graph break 诊断"""
    name = "graph_break"
    stage = "bringup"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        try:
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(model)(inputs)
            explanation_str = str(explanation)
            result.metrics["explanation_length"] = len(explanation_str)
            # 尝试提取 break 数量
            break_count = explanation_str.lower().count("break")
            result.metrics["break_mentions"] = break_count
            if break_count > 3:
                result.status = "warn"
        except Exception as e:
            result.status = "warn"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class CompileSpeedupPass(AnalysisPass):
    """编译加速比: eager vs 3 种 compile 模式"""
    name = "compile_speedup"
    stage = "perf"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        time_key = f"self_{device}_time_total"

        def measure(m, tag):
            with torch.no_grad():
                for _ in range(5):
                    m(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            with profile(activities=activities) as prof:
                with torch.no_grad():
                    for _ in range(10):
                        m(inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
            total = sum(getattr(e, time_key, 0) for e in prof.key_averages())
            n_kernels = len(prof.key_averages())
            return total, n_kernels

        try:
            eager_time, eager_kernels = measure(model, "eager")
            result.metrics["eager_time_us"] = eager_time
            result.metrics["eager_kernels"] = eager_kernels

            for mode in ["default", "reduce-overhead", "max-autotune"]:
                torch._dynamo.reset()
                try:
                    compiled = torch.compile(model, mode=mode)
                    with torch.no_grad():
                        for _ in range(3):
                            compiled(inputs)
                    ct, ck = measure(compiled, mode)
                    speedup = eager_time / ct if ct > 0 else 0
                    fusion = 1 - ck / eager_kernels if eager_kernels > 0 else 0
                    result.metrics[f"{mode}_time_us"] = ct
                    result.metrics[f"{mode}_kernels"] = ck
                    result.metrics[f"{mode}_speedup"] = round(speedup, 3)
                    result.metrics[f"{mode}_fusion_ratio"] = round(fusion, 3)
                except Exception as e:
                    result.metrics[f"{mode}_error"] = str(e)[:100]

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class RooflinePass(AnalysisPass):
    """Roofline 分析: AI / bound type / 效率"""
    name = "roofline"
    stage = "perf"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        try:
            with torch.no_grad():
                for _ in range(5):
                    model(inputs)

            with profile(activities=activities, with_flops=True, profile_memory=True) as prof:
                with torch.no_grad():
                    model(inputs)

            ai_values = []
            total_flops = 0
            total_time = 0
            for evt in prof.key_averages():
                flops = evt.flops or 0
                total_flops += flops
                t = evt.self_cuda_time_total if device == "cuda" else evt.self_cpu_time_total
                total_time += t
                if flops > 0:
                    mem = abs(evt.self_cpu_memory_usage) + abs(getattr(evt, "self_cuda_memory_usage", 0))
                    if mem > 0:
                        ai_values.append(flops / mem)

            result.metrics["total_gflops"] = round(total_flops / 1e9, 2)
            result.metrics["total_time_ms"] = round(total_time / 1e3, 3)
            if total_time > 0:
                result.metrics["throughput_tflops"] = round(total_flops / (total_time / 1e6) / 1e12, 4)
            if ai_values:
                ai_sorted = sorted(ai_values)
                result.metrics["ai_p25"] = round(ai_sorted[len(ai_sorted) // 4], 2)
                result.metrics["ai_p50"] = round(ai_sorted[len(ai_sorted) // 2], 2)
                result.metrics["ai_p75"] = round(ai_sorted[3 * len(ai_sorted) // 4], 2)
                result.metrics["n_kernels_with_flops"] = len(ai_values)

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class OpFrequencyPass(AnalysisPass):
    """算子频率分布 — 架构设计用"""
    name = "op_frequency"
    stage = "arch"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        try:
            with torch.no_grad():
                for _ in range(3):
                    model(inputs)
            with profile(activities=activities, record_shapes=True) as prof:
                with torch.no_grad():
                    model(inputs)

            time_key = f"self_{device}_time_total"
            op_time = {}
            total = 0
            for evt in prof.key_averages():
                t = getattr(evt, time_key, 0)
                op_time[evt.key] = t
                total += t

            # Top-10 by time
            sorted_ops = sorted(op_time.items(), key=lambda x: -x[1])[:10]
            top10 = {}
            for op, t in sorted_ops:
                top10[op] = {"time_us": t, "pct": round(t / total * 100, 1) if total > 0 else 0}
            result.metrics["top10_ops"] = top10
            result.metrics["total_ops"] = len(op_time)

            # Shape distribution
            shapes = Counter()
            for evt in prof.key_averages():
                if evt.input_shapes:
                    for s in evt.input_shapes:
                        if s and len(s) >= 2:
                            shapes[str(tuple(s))] += 1
            result.metrics["top_shapes"] = dict(shapes.most_common(10))

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class FusionBenefitPass(AnalysisPass):
    """Fusion 收益: eager vs compiled kernel 数量和内存"""
    name = "fusion_benefit"
    stage = "arch"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)

        try:
            with torch.no_grad():
                model(inputs)
            with profile(activities=activities, profile_memory=True) as prof_e:
                with torch.no_grad():
                    model(inputs)
            eager_k = len(prof_e.key_averages())
            eager_m = sum(abs(e.self_cpu_memory_usage) for e in prof_e.key_averages())

            torch._dynamo.reset()
            compiled = torch.compile(model, mode="default")
            with torch.no_grad():
                for _ in range(3):
                    compiled(inputs)
            with profile(activities=activities, profile_memory=True) as prof_c:
                with torch.no_grad():
                    compiled(inputs)
            comp_k = len(prof_c.key_averages())
            comp_m = sum(abs(e.self_cpu_memory_usage) for e in prof_c.key_averages())

            result.metrics["eager_kernels"] = eager_k
            result.metrics["compiled_kernels"] = comp_k
            result.metrics["kernel_reduction"] = round(1 - comp_k / eager_k, 3) if eager_k > 0 else 0
            result.metrics["mem_saved_kb"] = round(max(eager_m - comp_m, 0) / 1024, 1)

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class GoldenExtractPass(AnalysisPass):
    """Golden 提取 — 硅前验证用"""
    name = "golden_extract"
    stage = "presilicon"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        golden_dir = os.path.join(REPORT_DIR, "golden", spec.name)
        os.makedirs(golden_dir, exist_ok=True)

        try:
            goldens = {}
            hooks = []
            def make_hook(name):
                def hook(mod, inp, out):
                    def to_cpu(x):
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu()
                        elif isinstance(x, (tuple, list)):
                            return type(x)(to_cpu(i) for i in x)
                        return None
                    goldens[name] = {"input": to_cpu(inp), "output": to_cpu(out)}
                return hook

            for name, mod in model.named_modules():
                n = name or "root"
                hooks.append(mod.register_forward_hook(make_hook(n)))

            with torch.no_grad():
                model(inputs)

            for h in hooks:
                h.remove()

            # Save
            manifest = {}
            for name, data in goldens.items():
                safe = name.replace(".", "_")
                for key in ("input", "output"):
                    path = os.path.join(golden_dir, f"{safe}_{key}.pt")
                    torch.save(data[key], path)
                    manifest.setdefault(name, {})[key] = path

            manifest_path = os.path.join(golden_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            result.metrics["n_modules"] = len(goldens)
            result.metrics["golden_dir"] = golden_dir
            result.metrics["manifest"] = manifest_path

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


class BisectPass(AnalysisPass):
    """子模块 bisect — 定位 eager vs compiled 差异"""
    name = "bisect"
    stage = "bringup"

    def run(self, model, inputs, device, spec):
        t0 = time.time()
        result = AnalysisResult(self.name, spec.name, "pass")
        issues = []

        try:
            for name, submod in model.named_children():
                captured = {}
                def make_hook(n):
                    def hook(mod, inp, out):
                        captured[n] = tuple(
                            i.detach().clone() if isinstance(i, torch.Tensor) else i for i in inp
                        )
                    return hook
                h = submod.register_forward_hook(make_hook(name))
                with torch.no_grad():
                    model(inputs)
                h.remove()

                if name not in captured:
                    continue
                sub_input = captured[name]

                try:
                    with torch.no_grad():
                        eager_out = submod(*sub_input)
                    torch._dynamo.reset()
                    compiled = torch.compile(submod, mode="default")
                    with torch.no_grad():
                        for _ in range(2):
                            compiled(*sub_input)
                        compiled_out = compiled(*sub_input)

                    def max_diff(a, b):
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            return (a.float() - b.float()).abs().max().item()
                        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                            diffs = [max_diff(ai, bi) for ai, bi in zip(a, b) if ai is not None and bi is not None]
                            return max(diffs) if diffs else 0
                        return 0

                    diff = max_diff(eager_out, compiled_out)
                    if diff > 1e-4:
                        severity = "WRONG" if diff > 1.0 else "IMPRECISE"
                        issues.append({"module": name, "severity": severity, "max_diff": diff})
                except Exception as e:
                    issues.append({"module": name, "severity": "CRASH", "error": str(e)[:100]})

            result.metrics["modules_tested"] = len(list(model.named_children()))
            result.metrics["issues"] = issues
            result.metrics["n_issues"] = len(issues)
            if any(i["severity"] in ("WRONG", "CRASH") for i in issues):
                result.status = "fail"
            elif issues:
                result.status = "warn"

        except Exception as e:
            result.status = "fail"
            result.error = str(e)[:200]
        result.duration_s = time.time() - t0
        return result


# ============================================================
# 3. Pipeline — 编排 pass 执行顺序
# ============================================================

STAGE_PASSES = {
    "arch":       [OpFrequencyPass, FusionBenefitPass, RooflinePass],
    "presilicon":  [GoldenExtractPass],
    "bringup":    [CompileCorrectnessPass, GraphBreakPass, BisectPass],
    "perf":       [CompileSpeedupPass, RooflinePass],
    "ecosystem":  [CompileCorrectnessPass, GraphBreakPass, FusionBenefitPass],
    "all":        [OpFrequencyPass, FusionBenefitPass, CompileCorrectnessPass,
                   GraphBreakPass, CompileSpeedupPass, RooflinePass,
                   GoldenExtractPass, BisectPass],
}

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def run_pipeline(stage="all", model_filter=None, device=None):
    """运行指定阶段的全部分析 pass"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    passes = [P() for P in STAGE_PASSES.get(stage, STAGE_PASSES["all"])]
    models = get_registry()
    if model_filter:
        names = [n.strip() for n in model_filter.split(",")]
        models = {k: v for k, v in models.items() if k in names}

    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_results: List[AnalysisResult] = []
    total_models = len(models)
    total_passes = len(passes)

    print(f"\n{'='*70}")
    print(f"  方法轮自动化分析 — stage={stage}")
    print(f"  Models: {total_models}  Passes: {total_passes}  Device: {device}")
    print(f"{'='*70}")

    for mi, (model_name, spec) in enumerate(models.items(), 1):
        print(f"\n  [{mi}/{total_models}] {model_name} ({spec.category})")

        model = spec.factory().eval().to(device)
        inputs = spec.input_factory(device)

        for pi, p in enumerate(passes, 1):
            tag = f"    ({pi}/{total_passes}) {p.name}"
            try:
                r = p.run(model, inputs, device, spec)
                icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "○"}[r.status]
                detail = ""
                # 挑关键 metric 显示
                if "max_diff" in r.metrics:
                    detail = f" max_diff={r.metrics['max_diff']:.2e}"
                elif "default_speedup" in r.metrics:
                    detail = f" speedup={r.metrics['default_speedup']:.2f}x"
                elif "kernel_reduction" in r.metrics:
                    detail = f" fusion={r.metrics['kernel_reduction']:.1%}"
                elif "total_gflops" in r.metrics:
                    detail = f" {r.metrics['total_gflops']}GFLOPS"
                elif "n_modules" in r.metrics:
                    detail = f" {r.metrics['n_modules']} modules"
                print(f"{tag:40} {icon} {r.status:6}{detail}  ({r.duration_s:.1f}s)")
                if r.error:
                    print(f"{'':44} └─ {r.error[:80]}")
            except Exception as e:
                r = AnalysisResult(p.name, model_name, "fail", error=str(e)[:200])
                print(f"{tag:40} 💥 exception: {str(e)[:60]}")
            all_results.append(r)

    return all_results, timestamp


# ============================================================
# 4. Report — JSON 报告 + 终端摘要 + Diff
# ============================================================

def generate_report(results, timestamp, stage):
    """生成 JSON 报告"""
    report = {
        "timestamp": timestamp,
        "pytorch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stage": stage,
        "summary": {"total": 0, "pass": 0, "fail": 0, "warn": 0, "skip": 0},
        "results": [],
    }

    for r in results:
        report["summary"]["total"] += 1
        report["summary"][r.status] += 1
        entry = asdict(r)
        # 清理不可序列化的字段
        for k, v in list(entry.get("metrics", {}).items()):
            if isinstance(v, float) and (v != v):  # NaN
                entry["metrics"][k] = None
        report["results"].append(entry)

    report_path = os.path.join(REPORT_DIR, f"report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report, report_path


def print_summary(report, report_path):
    """终端打印摘要"""
    s = report["summary"]
    print(f"\n{'='*70}")
    print(f"  摘要")
    print(f"{'='*70}")
    print(f"  Total: {s['total']}  ✓ Pass: {s['pass']}  ✗ Fail: {s['fail']}  "
          f"⚠ Warn: {s['warn']}  ○ Skip: {s['skip']}")
    print(f"  报告: {report_path}")

    # 按模型汇总
    model_status = defaultdict(lambda: {"pass": 0, "fail": 0, "warn": 0})
    for r in report["results"]:
        model_status[r["model_name"]][r["status"]] += 1

    print(f"\n  {'Model':<25} {'Pass':>6} {'Fail':>6} {'Warn':>6} {'Status':>8}")
    print("  " + "-" * 55)
    for model, counts in model_status.items():
        status = "✓" if counts["fail"] == 0 else "✗"
        print(f"  {model:<25} {counts['pass']:>6} {counts['fail']:>6} {counts['warn']:>6} {status:>8}")


def diff_reports(current_path, baseline_path):
    """对比两次报告，检测回归"""
    with open(current_path) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\n{'='*70}")
    print(f"  回归检测: {os.path.basename(baseline_path)} → {os.path.basename(current_path)}")
    print(f"{'='*70}")

    # 索引 baseline
    bl_index = {}
    for r in baseline.get("results", []):
        key = (r["model_name"], r["pass_name"])
        bl_index[key] = r

    regressions = []
    improvements = []

    for r in current.get("results", []):
        key = (r["model_name"], r["pass_name"])
        if key not in bl_index:
            continue
        bl = bl_index[key]

        # 状态回归
        if bl["status"] == "pass" and r["status"] == "fail":
            regressions.append(f"  ✗ {key[0]}/{key[1]}: pass → fail")
        elif bl["status"] == "fail" and r["status"] == "pass":
            improvements.append(f"  ✓ {key[0]}/{key[1]}: fail → pass")

        # Speedup 回归 (> 10%)
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            bl_sp = bl.get("metrics", {}).get(f"{mode}_speedup")
            cur_sp = r.get("metrics", {}).get(f"{mode}_speedup")
            if bl_sp and cur_sp and cur_sp < bl_sp * 0.9:
                regressions.append(
                    f"  ⚠ {key[0]}/{key[1]}: {mode} speedup {bl_sp:.2f}x → {cur_sp:.2f}x (-{(1-cur_sp/bl_sp)*100:.0f}%)")
            elif bl_sp and cur_sp and cur_sp > bl_sp * 1.1:
                improvements.append(
                    f"  ↑ {key[0]}/{key[1]}: {mode} speedup {bl_sp:.2f}x → {cur_sp:.2f}x (+{(cur_sp/bl_sp-1)*100:.0f}%)")

    if regressions:
        print(f"\n  🔴 回归 ({len(regressions)}):")
        for r in regressions:
            print(r)
    else:
        print(f"\n  ✅ 无回归")

    if improvements:
        print(f"\n  🟢 改善 ({len(improvements)}):")
        for i in improvements:
            print(i)

    return len(regressions)


# ============================================================
# 5. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="方法轮自动化分析运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_analysis.py                          # 全量
  python run_analysis.py --stage perf             # 只跑性能分析
  python run_analysis.py --stage bringup --ci     # CI 模式
  python run_analysis.py --models transformer_lm  # 指定模型
  python run_analysis.py --diff reports/report_20260412_120000.json
        """)
    parser.add_argument("--stage", default="all",
                        choices=["all", "arch", "presilicon", "bringup", "perf", "ecosystem"],
                        help="芯片阶段 (default: all)")
    parser.add_argument("--models", default=None,
                        help="逗号分隔的模型名 (default: 全部)")
    parser.add_argument("--diff", default=None,
                        help="与指定报告 JSON 对比，检测回归")
    parser.add_argument("--ci", action="store_true",
                        help="CI 模式: 有 fail 返回非零退出码")
    parser.add_argument("--device", default=None,
                        help="强制指定设备 (cuda / cpu)")
    parser.add_argument("--list-models", action="store_true",
                        help="列出所有注册的模型")
    parser.add_argument("--list-passes", action="store_true",
                        help="列出所有分析 pass")

    args = parser.parse_args()

    if args.list_models:
        print("注册的模型:")
        for name, spec in get_registry().items():
            print(f"  {name:<25} [{spec.category}]  {spec.description}")
        return

    if args.list_passes:
        print("分析 Passes:")
        for stage, passes in STAGE_PASSES.items():
            print(f"\n  stage={stage}:")
            for P in passes:
                p = P()
                print(f"    {p.name}")
        return

    # 运行
    results, timestamp = run_pipeline(args.stage, args.models, args.device)
    report, report_path = generate_report(results, timestamp, args.stage)
    print_summary(report, report_path)

    # Diff
    n_regressions = 0
    if args.diff:
        n_regressions = diff_reports(report_path, args.diff)

    # CI 退出码
    if args.ci:
        n_fail = report["summary"]["fail"]
        if n_fail > 0 or n_regressions > 0:
            print(f"\n  CI FAIL: {n_fail} failures, {n_regressions} regressions")
            sys.exit(1)
        else:
            print(f"\n  CI PASS")
            sys.exit(0)


if __name__ == "__main__":
    main()
