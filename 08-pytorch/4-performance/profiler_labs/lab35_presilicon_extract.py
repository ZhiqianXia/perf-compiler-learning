"""实验 35：硅前验证 — 从真实模型提取 Golden & 编译到虚拟 ISA

芯片 Pre-silicon 阶段的核心需求:
  1. 从 PyTorch 模型提取每个算子/子图的 golden 输入输出
  2. 通过 torch.compile / Triton 生成 kernel 源码
  3. 编译到 LLVM IR（可进一步 lower 到你的虚拟 ISA）
  4. 在模拟器上验证正确性

本实验实现步骤 1-3，步骤 4 需要接入你的硬件模拟器。
"""

import os
import json
import torch
import torch.nn as nn
import torch.fx


# ==================== 模型定义 ====================
class MLP(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        return self.norm(self.fc2(h) + x)


class SmallTransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = MLP(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        return self.mlp(x)


# ==================== Step 1: FX 图提取与 Golden 收集 ====================
class GoldenCollector:
    """Hook-based golden 收集器: 记录每个子模块的输入输出"""

    def __init__(self):
        self.goldens = {}
        self._hooks = []

    def _make_hook(self, name):
        def hook(module, inp, out):
            # 只保存 tensor (跳过 None 等)
            def to_cpu(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu()
                elif isinstance(x, (tuple, list)):
                    return type(x)(to_cpu(i) for i in x)
                return None

            self.goldens[name] = {
                "input": to_cpu(inp),
                "output": to_cpu(out),
            }
        return hook

    def register(self, model, prefix=""):
        for name, mod in model.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name == "":
                full_name = "root"
            h = mod.register_forward_hook(self._make_hook(full_name))
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def save(self, path):
        """保存 golden 到目录 (每个子模块一对 input/output .pt 文件)"""
        os.makedirs(path, exist_ok=True)
        manifest = {}
        for name, data in self.goldens.items():
            safe_name = name.replace(".", "_")
            inp_path = os.path.join(path, f"{safe_name}_input.pt")
            out_path = os.path.join(path, f"{safe_name}_output.pt")
            torch.save(data["input"], inp_path)
            torch.save(data["output"], out_path)
            manifest[name] = {"input": inp_path, "output": out_path}
        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path


# ==================== Step 2: 算子级 FX 图分解 ====================
def extract_aten_ops(model, example_input):
    """通过 torch.export 或 fx.symbolic_trace 提取 aten 算子列表"""
    print("\n--- FX Symbolic Trace ---")
    try:
        traced = torch.fx.symbolic_trace(model)
        ops = []
        for node in traced.graph.nodes:
            if node.op == "call_function":
                ops.append(str(node.target))
            elif node.op == "call_module":
                ops.append(f"module:{node.target}")
        print(f"  节点总数: {len(list(traced.graph.nodes))}")
        print(f"  算子/模块调用: {len(ops)}")
        for op in ops:
            print(f"    {op}")
        return traced, ops
    except Exception as e:
        print(f"  symbolic_trace 失败: {e}")
        return None, []


# ==================== Step 3: Inductor 生成代码提取 ====================
def extract_inductor_code(model, example_input, device):
    """编译模型并提示如何获取 Triton / LLVM 代码"""
    print("\n--- Inductor Code Extraction ---")

    # 编译
    torch._dynamo.reset()
    compiled = torch.compile(model, backend="inductor", mode="default")

    with torch.no_grad():
        out = compiled(example_input)
    print(f"  Compiled 输出 shape: {out.shape}, dtype: {out.dtype}")

    # 提示: 查看生成代码
    print(f"""
  要查看 Inductor 生成的完整 Triton kernel:
    TORCH_LOGS="output_code" python {os.path.basename(__file__)}

  要将 Triton kernel 编译到 LLVM IR:
    1. 从上面输出中提取 @triton.jit 函数
    2. 使用 triton.compiler.compile() 编译到 LLVM IR:

       import triton
       from triton.compiler import ASTSource
       # src = ASTSource(fn=kernel_fn, ...)
       # compiled = triton.compile(src, target=("llvm", 64))

  要查看 Triton cache 中的 .llir / .ptx:
    ls ~/.triton/cache/**/*.llir
    ls ~/.triton/cache/**/*.ptx
""")
    return compiled


# ==================== Step 4: 正确性对比框架 ====================
def compare_golden(golden_dir, model, example_input, device, atol=1e-4, rtol=1e-4):
    """加载 golden 并与当前模型输出对比"""
    print("\n--- Golden 正确性对比 ---")
    manifest_path = os.path.join(golden_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  manifest 不存在: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # 重新收集当前输出
    collector = GoldenCollector()
    collector.register(model)
    with torch.no_grad():
        model(example_input)
    collector.remove()

    print(f"\n  {'Module':<40} {'Status':>10} {'MaxDiff':>12}")
    print("  " + "-" * 65)

    pass_count = 0
    fail_count = 0
    for name, current_data in collector.goldens.items():
        if name not in manifest:
            continue

        golden_out = torch.load(manifest[name]["output"], weights_only=True)
        current_out = current_data["output"]

        # 处理 tuple 输出
        if isinstance(golden_out, tuple) and isinstance(current_out, tuple):
            golden_out = golden_out[0] if golden_out[0] is not None else golden_out[1]
            current_out = current_out[0] if current_out[0] is not None else current_out[1]

        if not isinstance(golden_out, torch.Tensor) or not isinstance(current_out, torch.Tensor):
            continue

        current_out_cpu = current_out.detach().cpu().float()
        golden_out_cpu = golden_out.float()

        max_diff = (current_out_cpu - golden_out_cpu).abs().max().item()
        passed = torch.allclose(current_out_cpu, golden_out_cpu, atol=atol, rtol=rtol)

        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  {name:<40} {status:>10} {max_diff:>12.2e}")

    total = pass_count + fail_count
    print(f"\n  结果: {pass_count}/{total} PASS, {fail_count}/{total} FAIL")
    print(f"        atol={atol}, rtol={rtol}")
    return fail_count == 0


# ==================== Main ====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    output_dir = os.path.join(os.path.dirname(__file__), "presilicon_golden")

    # === 模型 ===
    model = SmallTransformerBlock(d_model=256, nhead=4).eval().to(device)
    x = torch.randn(2, 32, 256, device=device)

    # === Step 1: 收集 Golden ===
    print(f"\n{'='*60}")
    print("  Step 1: 收集 Golden (每个子模块的输入/输出)")
    print(f"{'='*60}")

    collector = GoldenCollector()
    collector.register(model)
    with torch.no_grad():
        out = model(x)
    collector.remove()

    manifest_path = collector.save(output_dir)
    print(f"  Golden 保存到: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  子模块数: {len(collector.goldens)}")
    for name in sorted(collector.goldens.keys()):
        g = collector.goldens[name]
        out_shape = g["output"].shape if isinstance(g["output"], torch.Tensor) else "tuple"
        print(f"    {name}: output={out_shape}")

    # === Step 2: 提取算子列表 ===
    print(f"\n{'='*60}")
    print("  Step 2: FX 图提取 aten 算子")
    print(f"{'='*60}")

    traced, ops = extract_aten_ops(model.cpu(), x.cpu())

    # === Step 3: Inductor 代码提取 ===
    print(f"\n{'='*60}")
    print("  Step 3: Inductor 生成代码 (Triton kernel)")
    print(f"{'='*60}")

    model_gpu = model.to(device)
    x_gpu = x.to(device)
    extract_inductor_code(model_gpu, x_gpu, device)

    # === Step 4: 正确性对比 (self-test) ===
    print(f"\n{'='*60}")
    print("  Step 4: Golden 正确性对比 (self-test)")
    print(f"{'='*60}")

    compare_golden(output_dir, model_gpu, x_gpu, device)

    # === 提示 ===
    print(f"\n{'='*60}")
    print("  Pre-silicon 对接指南")
    print(f"{'='*60}")
    print(f"""
  Golden 文件在: {output_dir}/
  每个子模块有 *_input.pt 和 *_output.pt

  对接你的硬件模拟器:
    1. 用 torch.load() 加载 input tensor
    2. 将 tensor 数据转为模拟器输入格式
    3. 在模拟器上运行对应的 kernel
    4. 将模拟器输出转回 tensor
    5. 用 torch.allclose() 与 output golden 对比

  自动化正确性回归:
    golden = torch.load("{output_dir}/norm_output.pt")
    sim_out = run_on_simulator(kernel, input_data)
    assert torch.allclose(golden, sim_out, atol=1e-3)
""")


if __name__ == "__main__":
    main()
