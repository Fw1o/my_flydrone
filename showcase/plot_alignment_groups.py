"""Alignment check chart: how many neurons each configured group actually matches.

Reads the full MaleCNS brain and renders one horizontal bar chart, so the alignment
work (and its remaining gaps) is visible at a glance.

Usage: python plot_groups.py <brain.npz> <out.png>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flydrones.brain import load_connectome  # noqa: E402
from flydrones.brain.connectome import GroupSpec  # noqa: E402
from flydrones.config import load_config  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = load_config()
c = load_connectome(src)

inputs = {k: GroupSpec.from_dict(k, v) for k, v in cfg["inputs"].items()}
outputs = {k: GroupSpec.from_dict(k, v) for k, v in cfg["outputs"].items()}
c.resolve_groups({**inputs, **outputs})

# populations whose L/R labels are *inferred* from the wiring, not annotated upstream
INFERRED = {"R16_L", "R16_R", "HAL_L", "HAL_R"}

rows = [(k, c.group(k).size, "input") for k in inputs] + [
    (k, c.group(k).size, "output") for k in outputs
]
rows.sort(key=lambda r: (r[2], r[1]))

fig, ax = plt.subplots(figsize=(11, 8))
y = np.arange(len(rows))
vals = [max(r[1], 0.6) for r in rows]          # keep zeros visible on a log axis
colors = []
for k, n, role in rows:
    if n == 0:
        colors.append("#d62728")
    elif k in INFERRED:
        colors.append("#17becf")               # input group, side recovered by inference
    else:
        colors.append("#1f77b4" if role == "input" else "#2ca02c")
ax.barh(y, vals, color=colors, height=0.72)
ax.set_yticks(y)
ax.set_yticklabels([f"{k}({role})" for k, n, role in rows], fontsize=9.5)
ax.set_xscale("log")
ax.set_xlim(0.5, 4000)
ax.set_xlabel("匹配到的神经元数量(对数刻度)")
ax.set_title(
    "全量连接组神经元组对齐检查 · malecns-v1.0 · 166,700 神经元 / 25,582,837 连接\n"
    "蓝 = 输入组 · 青 = 侧向由推断补全的输入组 · 绿 = 输出组 · 红 = 未匹配",
    fontsize=12,
)
for i, (_name, n, _role) in enumerate(rows):
    ax.text(max(n, 0.6) * 1.12, i, f"{n:,}", va="center", fontsize=8.5, color="#333333")
ax.grid(axis="x", alpha=0.3)
fig.text(
    0.5, 0.005,
    "侧向补全:原始数据里光电感受器(3,377 个 R1-R6)只有 13 个带 L/R 标注,平衡棒 205 个全部没有;\n"
    "现用连接结构推断补全(对那 13 个已知样本校验 13/13 正确)—— R16 由 4/9 变为 1,112/2,246,"
    "HAL 由“左右不分”变为 98/94。",
    ha="center", va="bottom", fontsize=9, color="#555555",
)
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(dst, dpi=150)
print("saved:", dst, f"({dst.stat().st_size // 1024} KB)")

print("\nnumbers:")
for k, n, role in rows:
    print(f"  {role:6s} {k:10s} {n:,}")
