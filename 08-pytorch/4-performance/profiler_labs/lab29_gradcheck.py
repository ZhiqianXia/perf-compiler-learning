"""实验 29：gradcheck — 验证自定义 autograd 函数的梯度正确性

当你写了自定义 backward，gradcheck 用数值微分来验证它是否正确。
"""

import torch
from torch.autograd import Function, gradcheck


class MyReLU(Function):
    """自定义 ReLU（正确实现）"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0).float()


class BuggyReLU(Function):
    """故意写错的 ReLU — backward 有 bug"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # BUG: 应该是 (x > 0)，这里错写成 (x > 0.5)
        return grad_output * (x > 0.5).float()


def main():
    x = torch.randn(8, requires_grad=True, dtype=torch.double)

    # ---- 正确的实现 ----
    print("=== 检查 MyReLU (正确实现) ===")
    try:
        ok = gradcheck(MyReLU.apply, (x,), eps=1e-6, atol=1e-4)
        print(f"  gradcheck 通过: {ok}")
    except RuntimeError as e:
        print(f"  gradcheck 失败: {e}")

    # ---- 有 bug 的实现 ----
    print("\n=== 检查 BuggyReLU (有 bug) ===")
    try:
        ok = gradcheck(BuggyReLU.apply, (x,), eps=1e-6, atol=1e-4)
        print(f"  gradcheck 通过: {ok}")
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        print(f"  gradcheck 失败: {msg}")

    print("\n关键 API:")
    print("  torch.autograd.gradcheck(func, inputs, eps, atol)")
    print("  torch.autograd.gradgradcheck(func, inputs)  — 二阶梯度")
    print("  注意: 输入需要 dtype=torch.double 以保证数值精度")


if __name__ == "__main__":
    main()
