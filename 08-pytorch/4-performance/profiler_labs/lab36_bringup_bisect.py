"""实验 36：Bring-up Bisect — 自动定位硬件/编译 Bug

Day 0 硬件到手后的核心问题: 跑一个完整模型失败了，bug 在哪？

本实验实现自动化 bisect:
  1. 逐算子对比 eager vs compiled 输出 → 定位第一个出错的算子
  2. 逐子模块隔离测试 → 缩小范围到具体 kernel
  3. 精度分级报告 → 区分 "完全错误" vs "精度略差"
  4. 导出失败 kernel 的独立复现脚本
"""

import os
import sys
import traceback
import torch
import torch.nn as nn
from collections import OrderedDict


# ==================== 测试模型 ====================
class ResidualMLP(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        return self.norm(self.fc2(h) + x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm(x)
        out, _ = self.attn(h, h, h)
        return x + out


class TestModel(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=6):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(AttentionBlock(d_model, nhead))
            else:
                self.layers.append(ResidualMLP(d_model))
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ==================== Bisect Engine ====================
class BisectResult:
    """单个模块的测试结果"""
    def __init__(self, name, passed, max_diff, atol, rtol, error=None):
        self.name = name
        self.passed = passed
        self.max_diff = max_diff
        self.atol = atol
        self.rtol = rtol
        self.error = error

    @property
    def severity(self):
        if self.error:
            return "CRASH"
        if self.max_diff > 1.0:
            return "WRONG"
        if not self.passed:
            return "IMPRECISE"
        return "OK"


def bisect_submodules(model, example_input, device, atol=1e-4, rtol=1e-4):
    """逐子模块测试: eager vs compiled 输出对比"""
    results = []

    for name, submod in model.named_children():
        print(f"\n  Testing: {name} ({submod.__class__.__name__})")

        # 准备该子模块的输入
        # 通过 hook 截获
        captured_input = {}

        def make_hook(n):
            def hook(mod, inp, out):
                captured_input[n] = tuple(i.detach().clone() if isinstance(i, torch.Tensor) else i for i in inp)
            return hook

        h = submod.register_forward_hook(make_hook(name))
        with torch.no_grad():
            model(example_input)
        h.remove()

        if name not in captured_input:
            print(f"    跳过 (无法捕获输入)")
            continue

        sub_input = captured_input[name]

        # Eager 输出
        try:
            with torch.no_grad():
                eager_out = submod(*sub_input)
        except Exception as e:
            print(f"    Eager 执行失败: {e}")
            results.append(BisectResult(name, False, float("inf"), atol, rtol, error=str(e)))
            continue

        # Compiled 输出
        try:
            torch._dynamo.reset()
            compiled_sub = torch.compile(submod, mode="default")
            with torch.no_grad():
                # Warmup
                for _ in range(2):
                    compiled_sub(*sub_input)
                compiled_out = compiled_sub(*sub_input)
        except Exception as e:
            print(f"    Compile 执行失败: {e}")
            results.append(BisectResult(name, False, float("inf"), atol, rtol, error=str(e)))
            continue

        # 对比
        def compare_tensors(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                diff = (a.float() - b.float()).abs().max().item()
                ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
                return ok, diff
            elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                max_diff = 0
                all_ok = True
                for ai, bi in zip(a, b):
                    if ai is None or bi is None:
                        continue
                    ok_i, diff_i = compare_tensors(ai, bi)
                    max_diff = max(max_diff, diff_i)
                    all_ok = all_ok and ok_i
                return all_ok, max_diff
            return True, 0

        ok, max_diff = compare_tensors(eager_out, compiled_out)
        results.append(BisectResult(name, ok, max_diff, atol, rtol))
        status = "PASS" if ok else "FAIL"
        print(f"    {status}  max_diff={max_diff:.2e}")

    return results


def bisect_aten_ops(model, example_input, device):
    """通过 FX 追踪，逐 aten 算子检查"""
    print("\n--- aten 算子级检查 ---")

    try:
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model)
    except Exception as e:
        print(f"  symbolic_trace 失败: {e}")
        return []

    results = []
    for node in traced.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "__name__"):
            op_name = node.target.__name__
        elif node.op == "call_method":
            op_name = node.target
        else:
            continue
        results.append(op_name)

    from collections import Counter
    op_counts = Counter(results)
    print(f"  aten op 统计:")
    for op, count in op_counts.most_common():
        print(f"    {op}: {count}")

    return results


# ==================== 全模型 Smoke Test ====================
def full_model_smoke_test(model, example_input, device, atol=1e-3):
    """全模型编译冒烟测试"""
    print("\n--- 全模型冒烟测试 ---")

    # Eager
    with torch.no_grad():
        eager_out = model(example_input)

    # Compiled
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, mode="default", fullgraph=False)
        with torch.no_grad():
            for _ in range(2):
                compiled(example_input)
            compiled_out = compiled(example_input)
    except Exception as e:
        print(f"  FAIL: compile 执行失败")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False, float("inf")

    max_diff = (eager_out.float() - compiled_out.float()).abs().max().item()
    passed = torch.allclose(eager_out.float(), compiled_out.float(), atol=atol, rtol=atol)
    status = "PASS" if passed else "FAIL"
    print(f"  {status}  max_diff={max_diff:.2e}  atol={atol}")
    return passed, max_diff


# ==================== 报告生成 ====================
def generate_report(results, model_name):
    """生成 bring-up 报告"""
    print(f"\n{'='*70}")
    print(f"  BRING-UP BISECT REPORT: {model_name}")
    print(f"{'='*70}")

    print(f"\n  {'Module':<35} {'Class':<20} {'Severity':>10} {'MaxDiff':>12} {'Status':>8}")
    print("  " + "-" * 88)

    severity_counts = {"OK": 0, "IMPRECISE": 0, "WRONG": 0, "CRASH": 0}
    for r in results:
        severity_counts[r.severity] += 1
        icon = {"OK": "✓", "IMPRECISE": "⚠", "WRONG": "✗", "CRASH": "💥"}.get(r.severity, "?")
        diff_str = f"{r.max_diff:.2e}" if r.max_diff != float("inf") else "N/A"
        print(f"  {r.name:<35} {r.severity:>10} {diff_str:>12}   {icon}")

    total = len(results)
    print(f"\n  汇总: {total} 个子模块测试")
    print(f"    ✓ OK:         {severity_counts['OK']}")
    print(f"    ⚠ IMPRECISE:  {severity_counts['IMPRECISE']}  (atol/rtol 超标但量级正确)")
    print(f"    ✗ WRONG:      {severity_counts['WRONG']}  (结果完全错误)")
    print(f"    💥 CRASH:      {severity_counts['CRASH']}  (执行崩溃)")

    # 失败的排在最前面
    failures = [r for r in results if r.severity in ("WRONG", "CRASH")]
    if failures:
        print(f"\n  ⚠ 需要优先排查:")
        for r in failures:
            print(f"    - {r.name}: {r.severity}", end="")
            if r.error:
                print(f" ({r.error[:80]})")
            else:
                print(f" (max_diff={r.max_diff:.2e})")

    return severity_counts


def export_repro_script(model, submod_name, example_input, output_dir):
    """导出失败 kernel 的最小复现脚本"""
    path = os.path.join(output_dir, f"repro_{submod_name}.py")
    script = f'''"""自动生成的复现脚本: {submod_name}

运行此脚本复现 eager vs compiled 差异:
  python {os.path.basename(path)}
  TORCH_LOGS="output_code" python {os.path.basename(path)}
"""
import torch

# 加载子模块和输入 (需要从 lab35 golden 目录加载)
# input_data = torch.load("presilicon_golden/{submod_name}_input.pt")

# 或者手动构造:
input_data = torch.randn(2, 32, 256)  # 根据实际 shape 修改

# TODO: 在此处定义或加载子模块
# submod = ...

# eager_out = submod(input_data)
# compiled_out = torch.compile(submod)(input_data)
# print("max_diff:", (eager_out - compiled_out).abs().max().item())
'''
    with open(path, "w") as f:
        f.write(script)
    print(f"  复现脚本: {path}")


# ==================== Main ====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    model = TestModel(d_model=256, nhead=4, num_layers=6).eval().to(device)
    x = torch.randn(2, 32, 256, device=device)
    output_dir = os.path.dirname(__file__)

    # === Step 1: 全模型冒烟测试 ===
    print(f"\n{'='*60}")
    print("  Step 1: 全模型冒烟测试")
    print(f"{'='*60}")

    smoke_ok, smoke_diff = full_model_smoke_test(model, x, device)

    # === Step 2: Graph Break 检查 ===
    print(f"\n{'='*60}")
    print("  Step 2: Graph Break 检查")
    print(f"{'='*60}")

    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(model)(x)
        print(explanation)
    except Exception as e:
        print(f"  explain() 失败: {e}")

    # === Step 3: 子模块 Bisect ===
    print(f"\n{'='*60}")
    print("  Step 3: 子模块逐个 Bisect (eager vs compiled)")
    print(f"{'='*60}")

    results = bisect_submodules(model, x, device, atol=1e-4, rtol=1e-4)

    # === Step 4: 报告 ===
    severity = generate_report(results, "TestModel(6 layers)")

    # === Step 5: 导出失败复现脚本 ===
    failures = [r for r in results if r.severity in ("WRONG", "CRASH")]
    if failures:
        print(f"\n{'='*60}")
        print("  Step 5: 导出失败算子的复现脚本")
        print(f"{'='*60}")
        for r in failures:
            export_repro_script(model, r.name, x, output_dir)

    # === 提示 ===
    print(f"\n{'='*60}")
    print("  Bring-up 下一步")
    print(f"{'='*60}")
    print("""
  如果全模型通过:
    → 进入 lab34 Roofline 分析，开始性能优化
    → 运行 lab37 统计更大模型的 op 覆盖率

  如果有子模块失败:
    1. 查看失败子模块的 Inductor 生成代码:
       TORCH_LOGS="output_code" python lab36_bringup_bisect.py

    2. 用 lab35 的 golden 对比模拟器输出

    3. 用 Nsight Compute 检查失败 kernel:
       ncu --set full python repro_<module>.py

    4. 常见硬件 bug 模式:
       - shared memory barrier 不正确 → attention kernel 出错
       - atomic 操作异常 → reduce kernel 出错
       - 浮点精度 (TF32 vs FP32) → matmul 精度偏差
""")


if __name__ == "__main__":
    main()
