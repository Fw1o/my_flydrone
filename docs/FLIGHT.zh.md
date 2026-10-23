# 真机飞行接入指南(中文)

> 这份文档回答一件事:**怎样把已经跑通的全量果蝇脑仿真,接到一架真实无人机上让它飞起来。**
> 英文版的平台细节见 [`HARDWARE.md`](HARDWARE.md);性能与数据细节见仓库外的
> `brain_optimization_full.md` / `showcase/README.md`。

---

## 0. 先读这一段(安全 + 现状)

1. **上游明确声明:4 个硬件适配器都是按官方 SDK 写的,但作者尚未实飞验证过。**
   仿真路径是端到端测过的,真机路径需要你自己做第一次验证。
2. **这不是玩具级的失控风险**:一架 80 g 的 Tello 旋翼转速上万转,近距离足以割伤眼睛。
   请严格按 §4 的四步顺序走,**桨叶拆除那一步不要跳**。
3. 本项目的控制哲学是"大脑输出摇杆指令、安全层有最终否决权":
   `safety` 段里的高度、油门、地理围栏、脑超时随时可以接管。**不要削弱这些限制去"让它飞得更猛"。**

### 0.1 Ctrl+C 会发生什么

收到 Ctrl+C 时,`runtime.py` 依次做三件事:**发悬停 → 调各平台的 `land()` → 断开连接**。
各平台 `land()` 的真实行为(代码实测):

| 平台 | `land()` 具体做什么 | 会不会砸下来 |
|---|---|---|
| **Tello** | `send_rc_control(0,0,0,0)` + SDK 的 `land` 命令 | ❌ **不会** —— Tello 自己以约 0.5–1 m/s 缓降 |
| **Crazyflie** | 20 次降高 setpoint(0.1 s/次,2 秒降到 5 cm)+ 停桨 | ❌ 基本不会,有 2 秒缓降 |
| **ArduPilot / PX4** | `LAND` 模式 / `MAV_CMD_NAV_LAND` | ❌ 不会,飞控自己降落 |
| **ESP32 + Betaflight** | 连发 1.5 秒下降指令,然后**停发** | ⚠️ 之后完全依赖飞控的 RX-loss failsafe |

**两条铁律:**

1. **只按一次 Ctrl+C。** 第一次才进入降落流程;如果在 `land()` 执行期间(2–5 秒)再按一次,
   会打断它 —— 降落命令没发完,程序却退出了。**按完立刻把手拿开,盯着它落地。**
2. **不要碰 `emergency_stop()`。** Tello 的 `emergency()` 是**立即切断电机**,飞机会直接掉下来。
   这是唯一一条"真·停桨"的路径,代码里只在显式调用时触发,不要顺手写进脚本。

### 0.2 软件救不了的四件事

| 场景 | 会发生什么 |
|---|---|
| 笔记本死机 / 休眠 / 进程被强杀 | `finally` 不执行 → **没有任何主动降落**。Tello 的 SDK 有"长时间无指令自动降落"的保护,但不同固件版本不一定一致;Crazyflie / Betaflight 则取决于各自固件的 failsafe 设置 |
| WiFi / radio 链路断开 | 飞机在等指令,等不到;后续行为同上 |
| 连按两次 Ctrl+C | 降落流程被中断(见上) |
| 低电量强制降落 | Tello 会自己降落,**可能不顾当前高度** |

→ 所以物理防护不是可选项。

### 0.3 起飞前的准备清单

| 准备 | 为什么 |
|---|---|
| **软垫 / 瑜伽垫(≥2×2 m)** | 铺满起降区。**不是为 Ctrl+C**(那是软着陆),而是为 §0.2 那四种"软件救不了"的情况,以及最常见的"漂移撞墙" |
| **护桨(prop guards)** | Tello 自带,务必装上 —— 它挡的是撞墙,不是掉落 |
| **护笼(约 50 cm 见方)** | 第一次飞强烈建议,能兜住绝大多数事故 |
| **清空场地** | 收好杯子、显示器、线缆;**别让飞机朝向任何人的脸** |
| **笔记本插电源 + 关休眠 / 自动锁屏** | "死机坠落"最常见的真实原因 |
| **电量 > 50%** | 低电量时 Tello 会强制降落 |
| **手放在键盘上** | Tello 没有实体遥控器,你唯一的接管手段就是 Ctrl+C |

第一次的目标是"确认链路和指令范围",不是"飞得好看":把高度窗口压到 **0.4–0.6 m**
(`safety.max_alt_m`),万一下落冲击也更小。

---

## 1. 选脑:三档对比

仓库里已经准备好三档脑,都在 `data/` 下(含完整左右侧向标注):

| 档位 | 数据文件 | 神经元 | 连接 | GPU 实时比 | CPU 实时比 | 20 Hz 周期占用 | 显存 | 适用 |
|---|---|---|---|---|---|---|---|---|
| **core-hops 1** | `malecns_core1_v2.npz` | 18,726 | 397,452 | **3.65x** | **2.41x** | 13.7 ms(GPU) | 约 20 MB | 首次试飞 / 没有 GPU |
| **core-hops 2** ⭐ | `malecns_core2_v2.npz` | 107,064 | 17,386,387 | **1.72x** | 0.61x | 29 ms | 约 210 MB | **推荐** |
| **full** | `malecns_brain_v2.npz` | 166,700 | 25,582,837 | 1.03x | 0.24x | 48.5 ms | 约 310 MB | 演示 / 离线分析 |

**为什么要看"20 Hz 周期占用"这一列**:控制循环是 20 Hz,也就是**每 50 ms 必须交出一个新指令**。
脑仿真占掉的时间之外,还要留给:摄像头解码、光流编码、手势识别(MediaPipe)、日志写盘。
- `core-hops 1`:占 13.7 ms → 剩 36 ms,很宽裕
- `core-hops 2`:占 29 ms → 剩 21 ms,合理
- `full`:占 48.5 ms → **只剩 1.5 ms**,任何一次卡顿都会超过 `safety.brain_timeout_s`(0.5 s)让飞机进入悬停

> **结论**:真机用 **`--input camera`(Tello 自己的视频)+ `core-hops 1` 或 `2`**,都能跑满 20 Hz;
> `full` 留作演示。手势模式(`--input gesture`)需要额外调参,见 §1.1。

### 1.1 真机每 tick 的实测耗时(20 Hz 预算 = 50 ms)

在本机(RTX 3060 Laptop / i7-12700H)实测,单个控制周期各阶段:

| 阶段 | 耗时 |
|---|---|
| `brain.tick(50 ms)` — core-hops 1 | **19.0 ms** |
| `brain.tick(50 ms)` — core-hops 2 | 34.1 ms |
| `retina.encode`(真实 640×480 帧) | 5.0 ms |
| `encoder.encode` | 0.1 ms |
| **笔记本摄像头 `webcam.read()`** | **35.1 ms** ⚠️ |
| Tello 视频流 `drone.frame()` | ~0 ms(后台线程解码,取最新帧) |

组合起来:

| 场景 | 每 tick | 20 Hz(50 ms) |
|---|---|---|
| **core1 + `--input camera`** | **25.4 ms** | ✅ 余量 24 ms |
| **core2 + `--input camera`** | **37.5 ms** | ✅ 余量 12 ms |
| core1 + `--input gesture` | 60.5 ms | ❌ 超 10 ms |
| core2 + `--input gesture` | 72.6 ms | ❌ 超 23 ms |

> **关键**:笔记本摄像头读一帧要 **35 ms**,是手势模式超预算的元凶;而 Tello 自己的视频
> 由后台线程解码、取最新帧几乎零成本 —— **所以真机优先用 `--input camera`**。
> 一定要用手势时,任选其一:把 `control.hz` 降到 10–12 / 摄像头降到 320×240 /
> 改用 `--gestures opencv`(实测仅 0.6 ms,但精度不如 MediaPipe)。

⚠️ **容量参数必须跟着档位走**。三份配置里已经写好了:

| 档位 | `brain.torch.max_spikes` / `max_events` |
|---|---|
| core-hops 1 | 512 / **16,384** |
| core-hops 2 | 1024 / **262,144** |
| full | 4096 / **540,672** |

如果给 core-hops 1 用 full 的容量,每步都会白遍历 53 万个空槽位 —— 实测实时比从 **3.65x 掉到 1.25x**。

---

## 2. 选平台:四种接入方式

| 平台 | 难度 | 定高 | 视觉输入 | 需要的硬件 | 适合 |
|---|---|---|---|---|---|
| **DJI / Ryze Tello** | ★ | 自带 | 内置 720p 视频 | 无(笔记本直连 WiFi) | **推荐首选** |
| **Crazyflie 2.1 + Flow deck** | ★★ | 靠 Flow deck | 无(只能用手势) | Crazyradio 遥控器 | 室内最安全,27 g |
| **ArduPilot / PX4 (MAVLink)** | ★★★ | 飞控负责 | 需另接相机 | 飞控 + 数传/串口 | 户外,先跑 SITL |
| **ESP32 + Betaflight** | ★★★★ | **没有** | 需另接相机 | ESP32 + 飞控 + 焊接 | 不建议入门 |

**第一次飞,选 Tello**:自带定高和光流定位,漂了会自己稳住;有护桨;WiFi 直连不需要额外硬件。

---

## 3. 环境准备

### 3.1 两个虚拟环境的分工

本机有两个独立的 venv,不要混用:

| 环境 | 路径 | 用来做什么 |
|---|---|---|
| **主环境** | `E:\pycharm_pro\pro_project\FlyDrone\venv` | CPU 脑后端、数据分析、画图 |
| **GPU 环境** | `E:\pycharm_pro\pro_project\FlyDrone\venv-gpu` | PyTorch + CUDA 后端(core-hops 2 / full 必需) |

> 为什么分开:torch 的 CUDA 依赖很大,放进主环境会拖累 CPU 路径的基线;两个环境各自独立,互不影响。

### 3.2 装依赖

```powershell
# 主环境:CPU 脑 + 无人机 SDK + 手势
& "E:\pycharm_pro\pro_project\FlyDrone\venv\Scripts\pip.exe" install -e ".[tello,gestures]"

# GPU 环境:在已有 torch 的基础上补上无人机 SDK 与手势
& "E:\pycharm_pro\pro_project\FlyDrone\venv-gpu\Scripts\pip.exe" install -e ".[tello,gestures]"
```

各平台要装的 extras:`tello`(Tello)、`crazyflie`(魔方)、`mavlink`(ArduPilot/PX4)、`gestures`(手势,建议都装)。

### 3.3 确认脑能跑

```powershell
$pyg = "E:\pycharm_pro\pro_project\FlyDrone\venv-gpu\Scripts\python.exe"
cd "E:\pycharm_pro\pro_project\FlyDrone\FlyDrones"

# 应该看到 "backend: TorchLIFNetwork on cuda" 和 ~1.7x real time
& $pyg -m flydrones bench --brain data/malecns_core2_v2.npz --backend torch --ms 1000
```

看到 `1.7x` 左右才算准备好。如果只有 `0.6x`,说明在用 CPU —— 检查 torch 是否装好、CUDA 是否可用。

### 3.4 确认飞行依赖装齐了(**最容易漏的一步**)

GPU 环境默认只有 torch,而真机飞行还需要 `opencv`(摄像头/手势)和 `djitellopy`(Tello)。
**这两样没装的话,`--send` 之前就会崩**,报 `OpenCV missing: pip install 'flydrones[vision]'`。

```powershell
& $pyg -c "import importlib
for m in ('torch','cv2','djitellopy','mediapipe'):
    try:
        importlib.import_module(m); print(f'{m:12s} OK')
    except Exception as e:
        print(f'{m:12s} MISSING  <-- 需要装')"
```

| 模块 | 缺了会怎样 | 怎么补 |
|---|---|---|
| `torch` | 跑 CPU(0.6x),达不到实时 | 见 §3.2 的 GPU 安装 |
| `cv2` | `--input camera` / `--input gesture` 直接退出 | `pip install -e ".[vision]"` |
| `djitellopy` | `--drone tello` 抛 `SystemExit` | `pip install -e ".[tello]"` |
| `mediapipe` | 手势降级成 OpenCV 肤色检测(能用但更抖) | `pip install -e ".[gestures]"` |

> `mediapipe` 可选;`cv2` 和 `djitellopy` 是飞 Tello 的**硬依赖**。

---

## 4. 起飞前必须走完的四步

> 每步都有明确的判断标准。**前一步不过,不要进下一步。**

### 步骤 0:仿真干跑(不需要无人机)

```powershell
& $pyg -m flydrones fly --drone sim --config configs/fly_core2.yaml --input camera --seconds 10
```

**判断标准**:10 秒内跑完、终端不报错、结尾能看到飞行摘要。这验证"脑 + 配置"本身没问题。

### 步骤 1:DRY RUN(完全不碰硬件)

```powershell
# 注意:不要加 --send
& $pyg -m flydrones fly --drone tello --config configs/fly_core2.yaml `
      --config configs/tello.yaml --input gesture --seconds 10
```

**这一步不会连接无人机。** 不加 `--send` 时程序根本不会创建 Tello 对象
(`cli.py`:`drone = make_drone(...) if args.send or args.drone == "sim" else None`),
所以它只打印 `[dry-run] would connect to tello` —— 不碰网络、不读遥测、不发指令。

**判断标准**:
- 终端出现 `DRY RUN: nothing will fly` 和 `[dry-run] would connect to tello`
- 实时比在 **1.5x 以上**(低于这个值,后面真飞会频繁脑超时)
- 10 秒内正常结束,没有异常堆栈
- 若报 `OpenCV missing` / `djitellopy missing` → 回到 §3.4 补依赖

> 它验证的是"脑 + 配置 + 指令范围";**WiFi 链路和真机遥测只能在步骤 2(拆桨 + `--send`)里验证**。

### 步骤 2:桨叶拆掉,加 `--send`

**把四个桨叶都拆下来。** 然后:

```powershell
& $pyg -m flydrones fly --drone tello --config configs/fly_core2.yaml `
      --config configs/tello.yaml --input gesture --send --seconds 20
```

**判断标准**:
- 电机随输入变化(能看到转速改变),但飞机不会飞走
- 你在镜头前张手/握拳/挥手时,电机声音明显变化
- **没有出现"油门突然打满不动"的僵死状态**(那是脑超时,见 §8)
- 按**一次** `Ctrl+C` 后,终端打印 `Ctrl+C -> landing`,电机会缓慢减速停下(不是骤停);
  若骤停或被"挂住不落",说明你按了第二次,或落到了 ESP32 那种"停发即失控"的路径上

### 步骤 3:空旷场地 + 护笼,低高度

- 场地:至少 3×3 m 空旷,或者干脆套一个护笼
- 配置里把高度压低(见 §7),先只用手势输入(`--input gesture`),不上摄像头光流
- 第一次飞 **10–15 秒就按一次 Ctrl+C**,然后**手离开键盘**,盯着它缓降落地,再复盘日志
- 起步前把 §0.3 的清单走一遍(软垫、护桨、护笼、电源、电量)

**判断标准**:起飞、悬停、缓降都正常,`--log` 里没有 `escape` 误触发,落地后机体无擦伤。

---

## 5. 各平台接入细则

### 5.1 DJI / Ryze Tello(推荐)

```powershell
# 1) 打开 Tello,笔记本连上它的 WiFi(TELLO-XXXXXX)
# 2) 跑(先不加 --send)
& $pyg -m flydrones fly --drone tello --config configs/fly_core2.yaml `
      --config configs/tello.yaml --input both --live
# 3) 确认无误后加 --send
```

- 摄像头画面喂给复眼(光流 + 逼近检测);笔记本摄像头识别手势叠加错觉
- 视频延迟 100–200 ms,逼近反射会变慢 —— 这是物理限制,不是 bug
- `Ctrl+C` 会执行缓降(只按一次,见 §0.1);**`emergency_stop()` 会立即停桨、飞机直接掉落** —— 只在真正失控时才用,平时不要碰
- Tello 需要**光照 + 有纹理的地面**才能定住位置,纯白地板会漂

### 5.2 Crazyflie 2.1 + Flow deck

```powershell
& $pyg -m flydrones fly --drone crazyflie --uri radio://0/80/2M/E7E7E7E7E7 `
      --config configs/fly_core2.yaml --config configs/crazyflie.yaml `
      --input gesture --send
```

- 没有摄像头,只能用笔记本摄像头做手势输入
- 脑输出的油门会被积分成**高度目标**,由飞控自己保持
- 新固件需要 arming 请求,适配器会自己发

### 5.3 ArduPilot / PX4 (MAVLink)

**先在 SITL 里跑通,再上真机。**

```powershell
# SITL:ArduPilot 用 sim_vehicle.py -v ArduCopter --console --map
& $pyg -m flydrones fly --drone mavlink --mavlink udpin:0.0.0.0:14550 `
      --autopilot ardupilot --config configs/fly_core2.yaml `
      --config configs/mavlink_sitl.yaml --input gesture --send
```

- 真机走串口数传:`--mavlink COM5,57600`(Windows)或 `/dev/ttyUSB0,57600`
- ArduPilot 用 GUIDED 模式,PX4 用 OFFBOARD(会先发一段 setpoint 再切)
- **户外飞,手里必须有能随时切回 LOITER/LAND 的真遥控器**

### 5.4 ESP32 + Betaflight(最难,不建议入门)

这条路径**没有定高**:油门是围绕 `HOVER_PWM` 的摇杆偏移,你必须自己标定这个值。
流程见 [`HARDWARE.md`](HARDWARE.md) 的 "Betaflight / INAV quad via ESP32 bridge";
务必拆桨标定 → 护笼 → 再飞。

---

## 6. 配置怎么叠加

`--config` 现在**可以写多次**,后面的文件覆盖前面的同名项。这是为了把"脑"和"机型"分开维护:

```powershell
& $pyg -m flydrones fly --drone tello `
      --config configs/fly_core2.yaml `      # 脑:数据文件 + 容量 + backend
      --config configs/tello.yaml `          # 机型:vision 增益 + 收紧的 safety
      --input both --send
```

合并结果(实测):

| 项 | 来自 | 值 |
|---|---|---|
| `brain.source` | fly_core2 | `data/malecns_core2_v2.npz` |
| `brain.torch.max_events` | fly_core2 | 262,144 |
| `safety.max_throttle` | tello | 0.5 |
| `vision.flow_gain` | tello | 2.0 |

> 只写一个 `--config` 也可以;写不存在的文件会**直接报错**(不会静默忽略)。

---

## 7. 关键参数调优

| 参数 | 默认 | 真机建议 | 说明 |
|---|---|---|---|
| `safety.max_throttle` | 0.6 | **0.4–0.5** | 垂直速度上限;室内低一点更安全 |
| `safety.max_yaw` | 0.6 | 0.4–0.5 | 偏航速率上限 |
| `safety.max_forward` | 0.4 | **0.2–0.3** | 前向速度上限(配合 `cruise` 用) |
| `safety.min_alt_m` / `max_alt_m` | 0.3 / 2.2 | 0.4 / 1.5 | 高度窗口,室内压低 |
| `safety.brain_timeout_s` | 0.5 | 0.5 | 脑超过这么久没输出就悬停;**别调大** |
| `safety.geofence_radius_m` | 3.0 | 2.0–3.0 | 只在无人机能报位置时生效 |
| `safety.min_battery_pct` | 20 | 25 | 低于此值自动降落 |
| `vision.flow_gain` | 1.2 | **2.0**(Tello) | 真机画面比仿真噪,要放大光流增益 |
| `decoder.cruise` | 0.0 | **0.2**(飞行配置已设) | 基础巡航速度;`forward` 轴还会被脑内的前向光流读出调制 |
| `decoder.axes.forward.gain` | 0.002 | 0.002–0.005 | 前向光流读出的强度;调大会变成正反馈(靠 `safety.max_forward` 限幅) |
| `brain.torch.max_events` | 540,672 | **按档位** | 见 §1;设错会掉 3 倍性能 |

> **前进轴现在真的接上脑了**:`decoder.axes.forward` 读出两个对"双侧 front-to-back 光流"
> 特异的神经元(`DNae010` / `DNp72`,实测对前向流 +71 / +39.5 Hz,对后退/旋转/loom 零响应),
> 所以飞机前进时会被视觉流**温和地推得更快**,转弯和后退不会误触发。
> 但真实场景的光流远弱于人工满幅刺激,神经部分只贡献约 **+0.02~0.04(巡航速度的 10–20%)**,
> 主要驱动力仍是 `cruise`。横向平移则**确认不可解**(与旋转在光流上不可区分)。
> 细节见 `showcase/README.md` 第三节。

---

## 8. 故障排查

| 症状 | 最可能的原因 | 处理 |
|---|---|---|
| 一起飞就定住不动,电机没反应 | 脑输出超时(`safety.brain_timeout_s`) | 看终端实时比;低于 1.2x 就换 `core-hops 1` |
| `groups with no matching neurons: HAL_L, HAL_R` | 用了侧向未补全的旧 npz | 换 `*_v2.npz`,或跑 `flydrones fill-sides` |
| 实时比只有 0.6x | 用 CPU 跑 core-hops 2 / full | 装 torch 用 GPU,或改 `fly_core1.yaml` |
| 只会原地打转,不往前飞 | `cruise` 还是 0(默认不巡航) | 用 `configs/fly_*.yaml`(已设 `cruise: 0.2`),或自己加 `--config` 设 cruise |
| 前飞时速度忽快忽慢 | 前向光流读出在调制(`gain: 0.002`) | 正常现象;嫌太敏感就把 `forward.gain` 调小或设 0 |
| 一直往一边漂 | Tello 缺纹理/光照,光流失效 | 换场地、加照明,或改 `--input gesture` |
| 看到东西靠近却不躲 | 视频延迟 100–200 ms 削弱了 looming | 降低 `safety.max_forward`;属于物理限制 |
| 频繁触发 `ESCAPE` 乱爬升 | 光流增益过高,被当成逼近 | 调低 `vision.loom_gain` |
| 启动报 `unknown brain.backend` | 配置写错 | 只能是 `cpu` / `torch` / `auto` |

---

## 9. 日志与复盘

一定要加 `--log`,出问题全靠它:

```powershell
& $pyg -m flydrones fly --drone tello --config configs/fly_core2.yaml `
      --config configs/tello.yaml --input both --send --log flight.csv
```

CSV 字段(每 50 ms 一行):

| 字段 | 含义 |
|---|---|
| `t`, `alt`, `x`, `y`, `yaw` | 时间 / 高度 / 位置 / 偏航 |
| `cmd_throttle`, `cmd_yaw`, `cmd_forward` | 安全层之后的最终指令(−1..1) |
| `escape` | 是否处于巨纤维逃逸反射 |
| `hz_DNg02_L/R` | 升降神经元放电率(全脑是否在工作) |
| `hz_DNp03_L/R`, `hz_DNp01_L/R` | 躲避扫视 / 逃逸神经元 |

**复盘要点**:`cmd_throttle` 长期贴在 ±0.5/0.6 上限 → 说明脑在拼命推,应该降增益或降高度目标;
`hz_DNg02_*` 长期为 0 → 全脑没在输出,检查是否脑超时。

---

## 10. 已知限制

| 限制 | 说明 |
|---|---|
| **适配器未实飞验证** | 上游声明;真机首次飞行需要人在现场判断 |
| **前进的神经驱动是"温和调制"** | 已接线(前向光流读出),但真实场景光流弱,只贡献巡航速度的 10–20%;主驱动仍是 `cruise` |
| **横向平移不可解** | 与旋转在两侧复眼的光流上无法区分,`lateral` 保持未接线 |
| **全量不适合真机** | 1.03x 只剩 1.5 ms/周期余量,任何抖动都会超时 |
| **数值一致性边界** | 保证统计量一致;数十秒后轨迹会与旧版本分叉,严格复现请用 CPU 路径 |
| **视频延迟** | Tello 100–200 ms,急停类反射不可能做到毫秒级 |
| **ESP32 无定高** | 需要自己标定 `HOVER_PWM`,是最危险的路径 |
