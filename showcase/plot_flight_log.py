"""Report-ready flight plot: altitude + behaviour annotations, lift DNs, escape DNs, commands.

Usage: python plot_flight2.py <demo_log.csv> <out.png>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_csv(src)
t = df["t"].to_numpy()
esc = df["escape"].astype(str).str.lower().isin(["true", "1"]).to_numpy()

# behaviour timeline printed by `flydrones demo` (label changes)
EVENTS = [
    (2.5, "张手\n下沉错觉"),
    (9.5, "场景旋转"),
    (13.5, "手逼近\n逃逸反射"),
    (15.5, "手放下\n上升错觉"),
]

fig, axes = plt.subplots(
    4, 1, figsize=(11.5, 11.5), sharex=True, gridspec_kw={"height_ratios": [1.15, 0.9, 1.0, 0.9]}
)

# ---------------------------------------------------------------- altitude
ax = axes[0]
ax.plot(t, df["alt"], color="#1f77b4", lw=2.4, zorder=3)
if esc.any():
    ax.fill_between(t, 0, df["alt"].max() * 1.2, where=esc, color="#d62728", alpha=0.16,
                    label="逃逸反射(安全层进入爬升模式)", zorder=1)
for x, label in EVENTS:
    ax.axvline(x, color="#888888", ls=":", lw=1.1, zorder=2)
    ax.annotate(label, xy=(x, df["alt"].max() * 1.19), xytext=(x, df["alt"].max() * 1.22),
                ha="center", va="bottom", fontsize=8.5, color="#444444")
ax.set_ylim(0, df["alt"].max() * 1.42)
ax.set_ylabel("高度 (m)")
ax.legend(loc="lower left", fontsize=9)
ax.set_title(
    "全量果蝇脑闭环飞行 · 166,700 神经元 / 25,582,837 连接 / 124,176,976 突触 · CPU 后端 (bench 0.24x 实时)",
    fontsize=12.5, pad=26,
)
ax.grid(alpha=0.3)

# ------------------------------------------------------ lift DNs (ongoing)
ax = axes[1]
ax.plot(t, df["hz_DNg02_L"], lw=1.8, color="#1f77b4", label="DNg02_L(左升力)")
ax.plot(t, df["hz_DNg02_R"], lw=1.8, color="#ff7f0e", label="DNg02_R(右升力)")
ax.set_ylabel("放电率 (Hz)")
ax.set_title("① 升降控制神经元 —— 全程持续工作的控制信号", fontsize=10.5, loc="left")
ax.legend(ncol=2, fontsize=9)
ax.grid(alpha=0.3)

# --------------------------------------------------- escape DNs (events)
ax = axes[2]
ax.plot(t, df["hz_DNp03_L"], lw=1.8, color="#2ca02c", label="DNp03_L(躲避扫视)")
ax.plot(t, df["hz_DNp03_R"], lw=1.8, color="#98df8a", label="DNp03_R(躲避扫视)")
ax.plot(t, df["hz_DNp01_L"], lw=1.8, color="#d62728", label="DNp01_L(巨纤维逃逸)")
ax.plot(t, df["hz_DNp01_R"], lw=1.8, color="#fcae91", label="DNp01_R(巨纤维逃逸)")
ax.annotate("手逼近 → 逃逸神经元爆发",
            xy=(13.5, df["hz_DNp01_L"].max()), xytext=(16.0, df["hz_DNp01_L"].max() * 0.82),
            arrowprops={"arrowstyle": "->", "color": "#d62728"}, fontsize=10, color="#d62728")
ax.set_ylabel("放电率 (Hz)")
ax.set_title("② 事件响应神经元 —— 视觉逼近触发逃逸反射", fontsize=10.5, loc="left")
ax.legend(ncol=4, fontsize=9)
ax.grid(alpha=0.3)

# ---------------------------------------------------------------- commands
ax = axes[3]
for col, c, lbl in [("cmd_throttle", "#1f77b4", "油门"), ("cmd_yaw", "#ff7f0e", "偏航")]:
    ax.plot(t, df[col], lw=1.8, color=c, label=lbl)
ax.set_ylabel("飞行指令 (-1..1)")
ax.set_xlabel("时间 (s)")
ax.set_title("③ 解码 + 安全层之后的最终指令(逃逸时刻油门被推至上限)", fontsize=10.5, loc="left")
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(dst, dpi=150)
print("saved:", dst, f"({dst.stat().st_size // 1024} KB)")
