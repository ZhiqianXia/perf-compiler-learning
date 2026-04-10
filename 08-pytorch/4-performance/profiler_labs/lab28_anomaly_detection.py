"""实验 28：anomaly detection — 自动检测 NaN / Inf 的来源

torch.autograd.detect_anomaly() 会在 backward 出现异常时
打印出产生问题的前向操作位置。
"""

import torch
import torch.nn as nn


def main():
    print("=== 正常运行 ===")
    try:
        run_normal()
        print("  正常完成，无异常。\n")
    except RuntimeError as e:
        print(f"  异常: {e}\n")

    print("=== 故意制造 NaN + detect_anomaly ===")
    try:
        run_with_nan()
    except RuntimeError as e:
        print(f"  detect_anomaly 捕获到异常:")
        # 只打印第一行
        for line in str(e).split("\n")[:5]:
            print(f"    {line}")

    print("\n关键 API:")
    print("  torch.autograd.set_detect_anomaly(True)")
    print("  或 with torch.autograd.detect_anomaly():")
    print("  → 会在 backward 遇到 NaN 时抛异常并显示前向代码位置")


def run_normal():
    model = nn.Linear(10, 10)
    x = torch.randn(4, 10, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()


def run_with_nan():
    with torch.autograd.detect_anomaly():
        x = torch.randn(4, 10, requires_grad=True)
        # 人为制造 NaN：log(负数)
        y = torch.log(x)  # 负数部分产生 NaN
        loss = y.sum()
        loss.backward()  # 这里会触发 anomaly detection


if __name__ == "__main__":
    main()
