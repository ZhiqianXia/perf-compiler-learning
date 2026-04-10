"""实验 21：contiguous vs non-contiguous — 内存布局对性能的影响

转置后的 tensor 不是 contiguous，某些算子会隐式拷贝。
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = f"self_{device}_time_total"

    a = torch.randn(1024, 1024, device=device)

    # contiguous tensor
    a_contig = a.contiguous()
    assert a_contig.is_contiguous()

    # non-contiguous tensor（转置后）
    a_noncontig = a.t()
    assert not a_noncontig.is_contiguous()

    # warmup
    for _ in range(5):
        torch.mm(a_contig, a_contig)
        torch.mm(a_noncontig.contiguous(), a_noncontig.contiguous())

    with profile(activities=activities) as prof:
        with record_function("CONTIG_MATMUL"):
            for _ in range(50):
                torch.mm(a_contig, a_contig)

    print("=== contiguous matmul ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=5))

    with profile(activities=activities) as prof:
        with record_function("NON_CONTIG_MATMUL"):
            for _ in range(50):
                # 非 contiguous 的 tensor 做 matmul，会先隐式 contiguous()
                torch.mm(a_noncontig, a_noncontig)

    print("\n=== non-contiguous matmul (隐式 copy) ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=5))

    print("\n提示: 观察是否多出了 aten::contiguous / aten::copy_ 算子")
    print(f"  a_contig.is_contiguous()    = {a_contig.is_contiguous()}")
    print(f"  a_noncontig.is_contiguous() = {a_noncontig.is_contiguous()}")
    print(f"  a_noncontig.stride()        = {a_noncontig.stride()}")


if __name__ == "__main__":
    main()
