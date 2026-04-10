"""实验 24：pin_memory & non_blocking 传输

pin_memory 锁页内存 + non_blocking=True 实现 CPU→GPU 异步传输。
"""

import time
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    if not torch.cuda.is_available():
        print("此实验需要 CUDA GPU。")
        print("核心概念:")
        print("  pin_memory=True:  DataLoader 分配锁页内存，DMA 传输更快")
        print("  non_blocking=True: .to(device) 不等 GPU 拷贝完成就返回")
        return

    # 构造数据
    data = torch.randn(10000, 3, 64, 64)
    labels = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(data, labels)

    configs = [
        {"pin_memory": False, "non_blocking": False, "tag": "普通传输"},
        {"pin_memory": True,  "non_blocking": False, "tag": "pin_memory"},
        {"pin_memory": True,  "non_blocking": True,  "tag": "pin_memory + non_blocking"},
    ]

    for cfg in configs:
        loader = DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=2, pin_memory=cfg["pin_memory"],
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for imgs, lbls in loader:
            imgs = imgs.to("cuda", non_blocking=cfg["non_blocking"])
            lbls = lbls.to("cuda", non_blocking=cfg["non_blocking"])

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"{cfg['tag']:35s} → {elapsed * 1000:.1f} ms")

    print("\n提示:")
    print("  pin_memory 对小 batch 效果不明显，大 batch / 大 tensor 差异更大。")
    print("  non_blocking 需要配合 pin_memory 使用才有意义。")


if __name__ == "__main__":
    main()
