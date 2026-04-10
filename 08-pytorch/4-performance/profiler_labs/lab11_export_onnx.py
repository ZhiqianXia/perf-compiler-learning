"""实验 11：导出 ONNX 文件并查看基础结构

目标：
1) 学会用 torch.onnx.export 导出 .onnx
2) 理解 dynamic_axes 的作用（支持动态 batch）
3) 可选地用 onnx 包读取并打印图结构

依赖：
  python3 -m pip install --user onnx
"""

import os
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    model = TinyMLP().eval()

    # 导出示例输入：batch=4, feature=16
    sample = torch.randn(4, 16)

    output_dir = os.path.dirname(__file__)
    onnx_path = os.path.join(output_dir, "tiny_mlp.onnx")

    export_kwargs = dict(
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    try:
        # 新导出路径在部分环境会依赖 onnxscript。
        torch.onnx.export(model, sample, onnx_path, **export_kwargs)
    except ModuleNotFoundError as err:
        if err.name != "onnxscript":
            raise
        print("检测到缺少 onnxscript，回退到 classic exporter (dynamo=False)。")
        try:
            torch.onnx.export(model, sample, onnx_path, dynamo=False, **export_kwargs)
        except Exception as fallback_err:
            msg = str(fallback_err)
            if "Module onnx is not installed" in msg or "No module named 'onnx'" in msg:
                print("\n导出失败：当前环境缺少 onnx。")
                print("请先安装后重试：python3 -m pip install --user onnx")
                return
            raise
    except Exception as err:
        msg = str(err)
        if "Module onnx is not installed" in msg or "No module named 'onnx'" in msg:
            print("\n导出失败：当前环境缺少 onnx。")
            print("请先安装后重试：python3 -m pip install --user onnx")
            return
        raise

    print(f"ONNX 导出完成: {onnx_path}")
    print("提示: input 的第 0 维是动态 batch_size")

    try:
        import onnx

        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        print("ONNX 校验通过。")
        print(f"graph name: {model_onnx.graph.name}")
        print(f"node count: {len(model_onnx.graph.node)}")

        print("\n前 5 个节点:")
        for i, node in enumerate(model_onnx.graph.node[:5]):
            print(f"  {i}: op_type={node.op_type}, name={node.name}")

    except ImportError:
        print("未安装 onnx，跳过模型校验与结构打印。")
        print("可执行: python3 -m pip install --user onnx")


if __name__ == "__main__":
    main()
