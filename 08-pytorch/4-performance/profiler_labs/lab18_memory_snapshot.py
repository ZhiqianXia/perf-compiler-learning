"""实验 18：CUDA 显存快照可视化

使用 torch.cuda.memory._record_memory_history 记录分配历史，
导出为 pickle，可用 PyTorch 官方工具 https://pytorch.org/memory_viz 查看。
"""

import os
import pickle
import torch
import torch.nn as nn


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    if not torch.cuda.is_available():
        print("此实验需要 CUDA GPU。")
        print("功能介绍：torch.cuda.memory._record_memory_history()")
        print("  → 记录每次 alloc / free 的调用栈")
        print("  → 导出后可在 https://pytorch.org/memory_viz 查看")
        return

    output_dir = os.path.dirname(__file__)
    snapshot_path = os.path.join(output_dir, "memory_snapshot.pickle")

    # 开始记录
    torch.cuda.memory._record_memory_history(max_entries=100000)

    model = SmallNet().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(5):
        x = torch.randn(256, 1024, device="cuda")
        out = model(x)
        loss = out.sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # 停止记录 & 导出
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)

    print(f"显存快照已保存: {snapshot_path}")
    print("可视化方法:")
    print("  1. 打开 https://pytorch.org/memory_viz")
    print("  2. 上传 memory_snapshot.pickle")
    print("  3. 查看 allocation timeline 和调用栈")


if __name__ == "__main__":
    main()
