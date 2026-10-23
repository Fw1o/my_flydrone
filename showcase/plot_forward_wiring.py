"""Why the demo looks like 'spin in place': compare forward-axis wiring on/off.

Left: top-down flight path. Right: the `forward` command over time.

Usage: python plot_forward_compare.py <default_log.csv> <cruise_log.csv> <out.png>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

base_csv, cru_csv, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
base, cru = pd.read_csv(base_csv), pd.read_csv(cru_csv)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2))

# ------------------------------------------------------------- left: path
for lo, hi, label in [
    ((1.2, -0.25), (1.7, 0.25), "椅子"),
    ((-2.9, 1.2), (-1.3, 2.9), "床"),
    ((2.4, -2.9), (2.9, -1.9), "衣柜"),
]:
    ax1.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1],
                            facecolor="#dddddd", edgecolor="#bbbbbb", zorder=1))
ax1.add_patch(Rectangle((-3, -3), 6, 6, facecolor="none", edgecolor="#888888", ls="--", lw=1.2))

ax1.plot(base["x"], base["y"], color="#1f77b4", lw=2, label="默认配置(前进轴未接线)")
ax1.plot(cru["x"], cru["y"], color="#d62728", lw=2, label="cruise = 0.35(打开前向巡航)")
ax1.scatter([base["x"].iloc[0]], [base["y"].iloc[0]], color="black", zorder=5, s=45, marker="o",
            label="起飞点")
ax1.scatter([cru["x"].iloc[-1]], [cru["y"].iloc[-1]], color="#d62728", zorder=5, s=55, marker="x")
ax1.scatter([base["x"].iloc[-1]], [base["y"].iloc[-1]], color="#1f77b4", zorder=5, s=55, marker="x")
ax1.set_xlim(-3.2, 3.2)
ax1.set_ylim(-3.2, 3.2)
ax1.set_aspect("equal")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_title("俯视轨迹:默认配置几乎原地打转(位移 %.2f m),\n打开 cruise 后正常前进(位移 %.2f m)"
              % (max(base["x"].max() - base["x"].min(), base["y"].max() - base["y"].min()),
                 max(cru["x"].max() - cru["x"].min(), cru["y"].max() - cru["y"].min())),
              fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)

# ------------------------------------------------------- right: forward cmd
ax2.plot(base["t"], base["cmd_forward"], color="#1f77b4", lw=2,
         label="默认配置:gain=0、cruise=0 → 恒为 0")
ax2.plot(cru["t"], cru["cmd_forward"], color="#d62728", lw=2, label="cruise = 0.35")
loom = (cru["cmd_forward"] < 0.34) & (cru["t"] > 2.0)
if loom.any():
    ax2.fill_between(cru["t"], 0, 0.45, where=loom, color="#ff7f0e", alpha=0.15)
    ax2.annotate("looming 出现时被刹车项压低\n(cruise 会自动减速避险)",
                 xy=(float(cru["t"][loom].iloc[len(cru["t"][loom]) // 2]), 0.30),
                 xytext=(float(cru["t"].max()) * 0.34, 0.12),
                 arrowprops={"arrowstyle": "->", "color": "#ff7f0e"}, fontsize=9.5, color="#b35900")
ax2.set_ylim(-0.05, 0.45)
ax2.set_xlabel("时间 (s)")
ax2.set_ylabel("forward 指令 (-1..1)")
ax2.set_title("前向指令:唯一让飞机水平移动的通道", fontsize=11)
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(alpha=0.3)

fig.suptitle("demo 为什么看起来“原地旋转”:decoder 的 forward / lateral 两轴默认未标定(gain = 0、cruise = 0)",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(dst, dpi=150)
print("saved:", dst, f"({dst.stat().st_size // 1024} KB)")

print("\ncruise 版位移:", f"{cru['x'].max()-cru['x'].min():.2f} m x {cru['y'].max()-cru['y'].min():.2f} m")
print("默认版位移:", f"{base['x'].max()-base['x'].min():.2f} m x {base['y'].max()-base['y'].min():.2f} m")
print("cruise 版 forward 取值范围:", f"{cru['cmd_forward'].min():.3f} .. {cru['cmd_forward'].max():.3f}")
print("cruise 版逃逸标志:",
      int(cru["escape"].astype(str).str.lower().isin(["true", "1"]).sum()), "行 True")
