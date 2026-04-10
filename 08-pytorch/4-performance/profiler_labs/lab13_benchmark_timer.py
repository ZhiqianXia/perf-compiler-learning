"""实验 13：torch.utils.benchmark — 微基准测试

比 time.time() 更精准：自动 warmup、多次采样、统计 median/IQR。
"""

import torch
from torch.utils.benchmark import Timer


def main():
    x = torch.randn(512, 512)
    y = torch.randn(512, 512)

    # ---------- 基本用法 ----------
    t = Timer(
        stmt="torch.mm(x, y)",
        globals={"x": x, "y": y, "torch": torch},
        label="matmul",
        sub_label="512x512",
        description="CPU",
    )
    result = t.blocked_autorange(min_run_time=1.0)
    print(result)

    # ---------- 对比不同尺寸 ----------
    results = []
    for n in [128, 256, 512, 1024]:
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        r = Timer(
            stmt="torch.mm(a, b)",
            globals={"a": a, "b": b, "torch": torch},
            label="matmul",
            sub_label=f"{n}x{n}",
            description="CPU",
        ).blocked_autorange(min_run_time=0.5)
        results.append(r)

    compare = torch.utils.benchmark.Compare(results)
    compare.print()

    # ---------- GPU (如果可用) ----------
    if torch.cuda.is_available():
        xg = torch.randn(1024, 1024, device="cuda")
        yg = torch.randn(1024, 1024, device="cuda")
        tg = Timer(
            stmt="torch.mm(xg, yg)",
            globals={"xg": xg, "yg": yg, "torch": torch},
            label="matmul",
            sub_label="1024x1024",
            description="CUDA",
        )
        print(tg.blocked_autorange(min_run_time=1.0))


if __name__ == "__main__":
    main()
