# 5. 分布式与多卡训练

- DataParallel vs DDP
  - `nn.DataParallel` 适合快速实验；`DistributedDataParallel` 是主流生产模式。
  - `torch.distributed.init_process_group`、`local_rank`、`world_size`。
- 通信后端
  - NCCL（GPU 高速）、Gloo（CPU/GPU）、MPI。
  - `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`。
- 并行策略
  - 数据并行、模型并行、张量并行、pipeline 并行。
  - ZeRO（DeepSpeed）、FSDP。
- BatchNorm 与同步
  - `nn.SyncBatchNorm`、`torch.nn.parallel.DistributedDataParallel(gradient_as_bucket_view=True)`。
- 容错与恢复
  - checkpoint 方案、训练重启、超时检测、断点续训。