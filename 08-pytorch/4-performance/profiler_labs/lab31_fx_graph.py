"""实验 31：torch.fx 符号追踪 — 查看计算图结构

torch.fx 把模型变成可编程的图 IR，可以：
  - 打印算子图
  - 分析模型结构
  - 做图变换（融合、剪枝等）
"""

import torch
import torch.nn as nn
import torch.fx


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def main():
    model = SmallNet().eval()

    # ---- 符号追踪 ----
    traced = torch.fx.symbolic_trace(model)

    # 1. 打印 graph
    print("=== torch.fx Graph ===")
    traced.graph.print_tabular()

    # 2. 生成等价 Python 代码
    print("\n=== 生成的 Python 代码 ===")
    print(traced.code)

    # 3. 遍历节点
    print("=== 节点列表 ===")
    for node in traced.graph.nodes:
        print(f"  op={node.op:15s} name={node.name:20s} target={node.target}")

    # 4. 统计各类型节点数量
    from collections import Counter
    op_counts = Counter(n.op for n in traced.graph.nodes)
    print(f"\n节点统计: {dict(op_counts)}")

    # 5. 验证追踪后的模型与原模型输出一致
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out_orig = model(x)
        out_traced = traced(x)
    diff = (out_orig - out_traced).abs().max().item()
    print(f"\n原始 vs 追踪后模型最大差异: {diff:.2e}")

    print("\n提示:")
    print("  torch.fx 适合编写自动化图变换工具")
    print("  局限: 不支持数据依赖的控制流 (if tensor.item() > 0)")
    print("  替代: torch.export / torch.compile 可处理更复杂情况")


if __name__ == "__main__":
    main()
