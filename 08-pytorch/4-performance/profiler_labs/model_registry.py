"""模型注册表 — 在这里添加你的自定义模型

run_analysis.py 会自动加载本文件中注册的所有模型。
只需添加 register_model() 调用即可，无需修改其他文件。

用法:
  1. 定义你的模型类
  2. 调用 register_model(name, category, factory, input_factory)
  3. 运行 python run_analysis.py 即可自动包含
"""

# 如果直接运行 run_analysis.py，它已经有内置模型。
# 本文件用于添加额外的自定义模型。
# 在 run_analysis.py 中 import 本模块即可注册。

import torch
import torch.nn as nn

# 导入注册函数
try:
    from run_analysis import register_model
except ImportError:
    # 独立运行时提供空实现
    def register_model(*args, **kwargs):
        print(f"  (独立模式) 注册: {args[0]}")


# ============================================================
# 在下面添加你的自定义模型
# ============================================================

# --- 示例: LLaMA-like (缩小版) ---
class MiniLLaMA(nn.Module):
    """缩小版 LLaMA 结构: RMSNorm + RoPE-style attention + SwiGLU"""
    def __init__(self, vocab=4000, d=256, nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d, nhead, d * 4, batch_first=True, norm_first=True,
            activation="gelu",  # 近似 SwiGLU
        )
        self.layers = nn.TransformerEncoder(encoder_layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.embed(x)
        h = self.layers(h)
        return self.head(self.norm(h))


# --- 示例: U-Net 风格 (缩小版, 用于 diffusion model 分析) ---
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.mid = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 64, 2, stride=2), nn.ReLU(), nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m = self.mid(d2)
        u1 = self.up1(m)
        return self.up2(u1)


# --- 注册 ---
register_model(
    "mini_llama", "transformer",
    lambda: MiniLLaMA(vocab=4000, d=256, nhead=4, layers=2),
    lambda dev: torch.randint(0, 4000, (4, 256), device=dev),
    "LLaMA-like (mini)",
)

register_model(
    "mini_unet", "vision",
    lambda: MiniUNet(),
    lambda dev: torch.randn(2, 3, 64, 64, device=dev),
    "U-Net style (mini, for diffusion)",
)


# ============================================================
# 添加模型模板 (取消注释并修改即可)
# ============================================================

# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # ... 你的模型
#
#     def forward(self, x):
#         # ...
#         return x
#
# register_model(
#     "my_model",           # 唯一名称
#     "transformer",         # 类别: transformer / vision / mlp / rnn / ...
#     lambda: MyModel(),     # 工厂函数
#     lambda dev: torch.randn(8, 512, device=dev),  # 输入工厂
#     "My custom model",     # 描述
# )
