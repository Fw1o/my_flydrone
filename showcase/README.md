# 展示素材 · 全量 MaleCNS 果蝇脑仿真

给汇报/演示用的一套素材,全部由 **全量** 果蝇脑
(**166,700 神经元 / 25,582,837 连接 / 124,176,976 突触**)跑出。

| 文件 | 展示什么 |
|---|---|
| `malecns_full_demo.gif` | 闭环演示动图(22 秒):复眼画面 → **实时脉冲栅格** → 高度/下行神经元/指令 |
| `malecns_flight_plot.png` | 一次完整飞行的四联图 —— 报告里最适合直接引用的图 |
| `malecns_groups.png` | 神经元组对齐检查:22 个输入/输出组各匹配到多少神经元 |
| `malecns_forward_wiring.png` | 解释 demo 为何"原地旋转":前进轴默认未接线,以及打开 `cruise` 后的对照 |
| `malecns_flight_log.csv` | 四联图背后的原始数据(440 行,每 50 ms 一行) |
| `cruise_example.yaml` | 打开前向巡航的最小配置(`decoder.cruise: 0.35`) |
| `plot_*.py` | 生成上述三张图的脚本(可复现) |

---

## 一、看图要点

### 1. 飞行四联图 `malecns_flight_plot.png`

| 面板 | 看点 |
|---|---|
| 高度 | 起飞 → 爬升 → 稳定控制 → 四段行为依次出现,0 碰撞 |
| ① 升降神经元 `DNg02_L/R` | 10–40 Hz 的**持续控制信号**,说明全脑一直在输出有效指令 |
| ② 事件响应神经元 | t=13.5 s "手逼近"时 `DNp01_L`(巨纤维逃逸)**冲到 ~360 Hz**,`DNp03` 同步响应 |
| ③ 指令 | 同一时刻油门被推到 **+0.6(安全层上限)**,反射直接变成动作 |

> 注:该图录制于侧向映射修复**之前**。修复后行为序列相同,高度略有变化(bias 现在真正生效)。

### 2. 闭环演示动图 `malecns_full_demo.gif`

22 秒内依次出现:

```
t= 2.5s  张手        → 下沉错觉(大脑加大升力)
t= 9.5s  场景旋转     → 视觉旋转输入
t=13.5s  手逼近       → ESCAPE 逃逸反射(巨纤维爆发,油门推满)
t=15.5s  手放下       → 上升错觉
collisions: 0
```

### 3. 对齐检查 `malecns_groups.png`

- 8 个运动视觉输入组(T4a/b/c/d 左右)各匹配到 **1,661–1,757 个神经元**,数量对称。
- 光电感受器 `R16` 与平衡棒 `HAL` 原来几乎匹配不到(4 / 9 个、左右不分),
  现已用**连接结构推断**补全侧向:校验 **13/13 正确**,`R16` 1,112 / 2,246,`HAL` 98 / 94。
- 6 个输出组(DNg02 15/14、DNp03、DNp01)全部命中。

### 4. 前进 / 横向:前进已接线,横向不可解 `malecns_forward_wiring.png`

demo 看起来"原地旋转"**不是仿真出错**,而是 decoder 的四个轴里,原来只有两个接了线:

| 轴 | 原状态 | 现状态 |
|---|---|---|
| `throttle`(升降) | ✅ DNg02_L/R 各 0.5 | 不变 |
| `yaw`(偏航) | ✅ DNg02 差模 + DNp03 | 不变 |
| **`forward`(前进)** | ❌ `gain=0`、`terms={}`、`cruise=0` → 恒 0 | ✅ **已接线**(见下) |
| **`lateral`(横向)** | ❌ 同上 | ❌ **不可解**(见下) |

#### 前进:找到了对"前向平移流"特异的神经元

上一轮调研的缺口在于**刺激选错了** —— 当时用"双侧 loom(扩张)"当前进代理,而 `loom = min(水平拉伸, 垂直拉伸)`
会被垂直线索污染。改成直接刺激方向性运动检测器后(T4a = front-to-back,Maisak 2013),扫描数据集里
**全部 484 种下行神经元**,得到:

| 候选 | Δ前向流 | Δ其他方向(最大) | 判断 |
|---|---|---|---|
| `DNae010` → 组名 `DNfwd_a` | **+71 Hz** | **0.0 Hz** | ✅ 完美特异 |
| `DNp72` → 组名 `DNfwd_b` | **+39.5 Hz** | **0.0 Hz** | ✅ 完美特异 |
| `DNbe001` | +234 Hz | +104 Hz(旋转) | ❌ 会被转向误触发 |
| `DNa10`(文献提名的 thrust/pitch) | +62 Hz | +56.5 Hz(旋转) | ❌ 不特异 |
| `DNp15`(HS 下游) | +163 Hz | **+185.5 Hz(后退)** | ❌ 是运动能量检测器,不是前进 |

> 这两个类型的功能**未在已发表文献中详述** —— 它们对前向光流的选择性是**在本连接组上实测出来的**,
> 不是文献既有的"前进速度神经元"。这一点必须说清楚。

**接线方式**(`defaults.yaml`):

```yaml
outputs:
  DNfwd_a: {types: ["^DNae010$"]}
  DNfwd_b: {types: ["^DNp72$"]}
decoder:
  axes:
    forward:
      gain: 0.002
      terms: {DNfwd_a: 1.0, DNfwd_b: 1.0}
  cruise: 0.0     # 基础巡航速度,由飞行配置设成 0.2
```

**实测行为**(core-hops 2,`cruise: 0.2`,16 秒闭环):

| 视觉情境 | `DNfwd_a` | `forward` 指令 | 判定 |
|---|---|---|---|
| 前向平移流 | 70.5 Hz | **+0.216** | ✅ 驱动力 |
| 后退流 | 0 Hz | 0.000 | ✅ |
| 向右旋转 | 1.0 Hz | 0.000 | ✅ |
| 向左旋转 | 24 Hz | +0.048 | ✅ 残余可接受(左转时左眼确实看到前向流) |
| looming | 0 Hz | 0.000(但 throttle +0.8 爬升) | ✅ |

闭环飞行:水平位移从 **0.12 m → 0.52 m**,0 碰撞。

> **诚实说明**:真实场景的光流远弱于人工满幅刺激,所以神经读出在飞行中只贡献
> **约 +0.02~0.04(巡航速度的 10–20%)**。主要驱动力仍是 `cruise`;
> 神经部分的作用是"随视觉流**温和调制**"。想要更激进可把 `gain` 提到 0.005,
> 但那是正反馈(前进→更多前向流→更快),要靠 `safety.max_forward` 限幅。

#### 横向:确认不可解

侧向平移与旋转在**两只侧置复眼的光流上无法区分**(都是"一眼向后流、一眼向前流")。
文献里 `DNa02/DNa03` 只区分"转向",不分 yaw 与平移。因此 `lateral` 保持未接线,
并在 `defaults.yaml` 里写明原因 —— 这比硬接一个没有依据的映射更诚实。

```powershell
# 想手动巡航(不依赖读数)时:
& $py -m flydrones demo --brain data\malecns_brain_v2.npz `
      --config showcase\cruise_example.yaml --seconds 16
```

| | 默认配置 | `cruise: 0.35` |
|---|---|---|
| 水平位移 | 0.12 m | **0.94 × 1.00 m**(同时持续转弯,走成圆弧) |
| forward 指令 | 0(现在会被前向光流轻微推高) | 0.35 平台;遇 looming 被刹车到 0 |

---

## 二、关键数字

### 性能

| 规模 | 数据文件 | 神经元 | CPU | GPU(graph) | GPU 推荐容量 |
|---|---|---|---|---|---|
| **full** | `malecns_brain_v2.npz` | 166,700 | 0.24x | **1.02x** | 4096 / 540,672(默认) |
| core-hops 2 | `malecns_core2_v2.npz` | 107,064 | — | **1.72x** | 1024 / 262,144 |
| core-hops 1 | `malecns_core1_v2.npz` | 18,726 | — | **3.65x** | 512 / 16,384 |

> **容量必须跟着网络规模缩小** —— CUDA Graph 的每步工作量正比于 `max_events`,
> 小网络沿用 540,672 会让 core-hops 1 从 3.65x 掉到 1.25x。
> 用法:把表中的两个数字写进配置的 `brain.torch` 段(`max_spikes` / `max_events`)。

### 压测(200 个控制 tick = 10 s 脑时间)

| 指标 | 结果 |
|---|---|
| 实时比 | 1.00x → 1.02x(漂移 +1.9%,无衰减) |
| GPU 显存 | 307.7 MiB allocated / 348.0 MiB reserved,**零增长** |

### 与"起点"的对照

| 文件 | 内容 | 显示的实时比 |
|---|---|---|
| `my_demo.gif` | MiniFly 合成电路(850 神经元) | 15.1x |
| `full_demo.gif` / `my_slow_demo.gif` | 全量连接组(**优化前**) | 0.1x |
| `showcase/malecns_full_demo.gif` | 全量连接组(**优化后**) | 0.2x(CPU) |

---

## 三、神经元映射调研(B 段结论)

文献给了两个候选方向,实测都没有直接可用的"前进"信号:

| 候选 | 文献依据 | 实测(6 种视觉刺激) |
|---|---|---|
| `DNg02` 的 a–g 亚群 | Namiki 2022:作者假设亚群分管 roll / pitch / **forward thrust** | 只有 `DNg02_a`(升降主力)与 `DNg02_c` 活跃;b/d/e/f/g **全静默**(缺视觉驱动) |
| `DNa10` | Cheong 2024(eLife):双侧输入到转向运动神经元 → thrust/lift/pitch | 对"场景下移"响应 **225 Hz**、对"右转"44 Hz,对**前向扩张光流 0 Hz** |
| `DNa05` | Cheong 2024:侧化为 yaw/roll | 对**双侧扩张光流 73 Hz**(其余全 0)—— 更像"逼近"而非"前进" |
| `DNa04` | Cheong 2024:yaw/roll | 全刺激 0 Hz |

**结论**:没有单个下行神经元被证明专门控制前进速度(文献与实测一致)。
前进在果蝇里由**身体俯仰**实现,而俯仰在本连接组里没有可读出的"速度指令"通道。
因此 `forward`/`lateral` 保持未接线,由 `cruise` 常数(或使用者自己的标定)提供。

---

## 四、复现方式

```powershell
$py = "E:\pycharm_pro\pro_project\FlyDrone\venv\Scripts\python.exe"
cd "E:\pycharm_pro\pro_project\FlyDrone\FlyDrones"

# 1) 录演示动图并导出飞行日志
& $py -m flydrones demo --config configs\malecns.yaml --seconds 22 `
      --record showcase\malecns_full_demo.gif --log showcase\malecns_flight_log.csv

# 2) 飞行四联图 / 对齐图
& $py showcase\plot_flight_log.py showcase\malecns_flight_log.csv showcase\malecns_flight_plot.png
& $py showcase\plot_alignment_groups.py data\malecns_brain_v2.npz showcase\malecns_groups.png

# 3) 性能(三档规模;GPU 需要独立 venv-gpu)
& $py -m flydrones bench --brain data\malecns_brain_v2.npz --ms 1000
$pyg = "E:\pycharm_pro\pro_project\FlyDrone\venv-gpu\Scripts\python.exe"
& $pyg -m flydrones bench --brain data\malecns_brain_v2.npz --backend torch --ms 1000

# 4) 输入→输出 响应核查(文本表,16 组 × 500 ms)
& $py -m flydrones inspect --config configs\malecns.yaml
```

依赖:主环境 `venv`(numpy / scipy / matplotlib / pandas / opencv);GPU 用独立 `venv-gpu`(PyTorch CUDA)。
