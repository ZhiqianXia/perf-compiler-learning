"""实验 30：模型 Hooks — 用 forward / backward hook 调试中间层

register_forward_hook   → 查看每层的输出 shape、数值范围
register_backward_hook  → 查看每层的梯度
"""

import torch
import torch.nn as nn


class DebugNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def make_forward_hook(name):
    def hook(module, input, output):
        print(f"  [FWD] {name:15s} | out shape={tuple(output.shape)} "
              f"| mean={output.mean().item():.4f} | std={output.std().item():.4f} "
              f"| min={output.min().item():.4f} | max={output.max().item():.4f}")
    return hook


def make_backward_hook(name):
    def hook(module, grad_input, grad_output):
        go = grad_output[0]
        print(f"  [BWD] {name:15s} | grad_out shape={tuple(go.shape)} "
              f"| mean={go.mean().item():.4f} | norm={go.norm().item():.4f}")
    return hook


def main():
    model = DebugNet()

    # 注册 hooks
    handles = []
    for name, mod in model.named_modules():
        if name:  # 跳过根模块
            handles.append(mod.register_forward_hook(make_forward_hook(name)))
            handles.append(mod.register_full_backward_hook(make_backward_hook(name)))

    x = torch.randn(16, 64)

    print("=== Forward pass ===")
    out = model(x)

    print("\n=== Backward pass ===")
    loss = out.sum()
    loss.backward()

    # 移除 hooks（生产环境记得清理）
    for h in handles:
        h.remove()

    print("\n提示:")
    print("  1. forward_hook 可以检查中间层输出是否有 NaN / 爆炸")
    print("  2. backward_hook 可以检查梯度是否消失 / 爆炸")
    print("  3. 用完后调用 handle.remove() 清理")
    print("  4. register_forward_pre_hook: 可以在前向之前修改输入")


if __name__ == "__main__":
    main()
