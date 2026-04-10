"""实验 15：DataLoader 瓶颈定位

模拟数据加载 vs 计算的时间占比，学习如何发现 I/O 瓶颈。
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity, record_function


class SlowDataset(Dataset):
    """模拟慢速数据集：每次 __getitem__ 人为 sleep"""

    def __init__(self, size=200, sleep_sec=0.01):
        self.size = size
        self.sleep_sec = sleep_sec

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        time.sleep(self.sleep_sec)  # 模拟磁盘 I/O
        return torch.randn(3, 32, 32), torch.randint(0, 10, ())


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def main():
    print("=== num_workers=0 (主进程加载) ===")
    run_experiment(num_workers=0)
    print("\n=== num_workers=2 (多进程加载) ===")
    run_experiment(num_workers=2)


def run_experiment(num_workers: int):
    ds = SlowDataset(size=40, sleep_sec=0.01)
    loader = DataLoader(ds, batch_size=8, num_workers=num_workers)

    model = TinyCNN()
    criterion = nn.CrossEntropyLoss()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for batch_idx, (imgs, labels) in enumerate(loader):
            with record_function("FORWARD"):
                out = model(imgs)
                loss = criterion(out, labels)
            with record_function("BACKWARD"):
                loss.backward()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(f"提示: 对比 enumerate(DataLoader) 和 FORWARD/BACKWARD 耗时占比")


if __name__ == "__main__":
    main()
