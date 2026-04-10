"""实验 10：将 PyTorch Profiler 输出为 TensorBoard 可读日志

运行后会在当前目录生成 tb_logs/，然后可用以下命令查看：
  tensorboard --logdir tb_logs --port 6006
"""

import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model = SimpleMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 256
    x = torch.randn(batch_size, 1024, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    output_dir = os.path.dirname(__file__)
    logdir = os.path.join(output_dir, "tb_logs")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    print(f"Profiling device: {device}")
    print(f"TensorBoard logdir: {logdir}")

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(6):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            writer.add_scalar("train/loss", loss.item(), step)
            print(f"step={step}, loss={loss.item():.4f}")
            prof.step()

    writer.flush()
    writer.close()

    print("\nDone. 使用以下命令打开 TensorBoard:")
    print(f"tensorboard --logdir {logdir} --port 6006")


if __name__ == "__main__":
    main()
