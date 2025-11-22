# Teleoperation Recording 使用文档

## 概述

`teleop_record.py` 是一个用于在 Isaac Lab 环境中进行遥操作并录制机器人演示数据的脚本。它支持多种输入设备（键盘、SO101 Leader），可以录制单臂或双臂机器人的操作数据，并将数据保存为 LeRobot 格式的数据集。

## 主要特性

- ✅ **多种输入设备支持**
  - `keyboard`: 单臂键盘控制
  - `bi-keyboard`: 双臂键盘控制
  - `so101leader`: SO101 Leader 单臂设备
  - `bi-so101leader`: 双臂 SO101 Leader 设备

- ✅ **数据录制功能**
  - 自动保存为 LeRobot 数据集格式
  - 支持图像（RGB）、深度图、关节状态、动作数据
  - 视频编码压缩存储
  - 自动增量保存到 `Datasets/record/XXX/`

- ✅ **录制控制**
  - **S键**: 开始录制
  - **N键**: 保存当前 episode
  - **D键**: 丢弃当前 episode（重新录制）
  - **ESC键**: 中止录制并清空 buffer（安全退出）
  - **Ctrl+C**: 强制退出（会自动清空 buffer）

- ✅ **灵活配置**
  - 可配置录制频率 (step_hz)
  - 可选择是否录制深度图
  - 支持设备校准
  - 自动任务类型验证

## 命令行参数

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | 必需 | 任务名称，如 `LeHome-SO101-Direct-Grament-v0` |
| `--num_envs` | int | 1 | 并行环境数量 |
| `--seed` | int | 42 | 随机种子 |

### 输入设备参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--teleop_device` | str | keyboard | 输入设备类型，可选：<br>- `keyboard`: 单臂键盘<br>- `bi-keyboard`: 双臂键盘<br>- `so101leader`: 单臂 SO101<br>- `bi-so101leader`: 双臂 SO101 |
| `--port` | str | /dev/ttyACM0 | 单臂 SO101 Leader 串口 |
| `--left_arm_port` | str | /dev/ttyACM0 | 双臂左臂 SO101 串口 |
| `--right_arm_port` | str | /dev/ttyACM1 | 双臂右臂 SO101 串口 |
| `--sensitivity` | float | 1.0 | 键盘控制灵敏度 |
| `--recalibrate` | flag | False | 是否重新校准 SO101 设备 |

### 录制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record` | flag | False | **必须添加此参数才能启用录制** |
| `--num_episode` | int | 100 | 最大录制 episode 数量 |
| `--step_hz` | int | 60 | 环境步进频率（Hz） |
| `--disable_depth` | flag | False | 禁用深度图录制 |

### AppLauncher 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--headless` | flag | False | 无头模式运行 |
| `--device` | str | cuda:0 | 运行设备 |
| `--enable_cameras` | flag | False | 启用相机渲染 |

## 按键说明

### 键盘设备 (keyboard)

#### 控制键
| 按键 | 功能 |
|------|------|
| **T/G** | Joint 1 (shoulder_pan) 正/负 |
| **Y/H** | Joint 2 (shoulder_lift) 正/负 |
| **U/J** | Joint 3 (elbow_flex) 正/负 |
| **I/K** | Joint 4 (wrist_flex) 正/负 |
| **O/L** | Joint 5 (wrist_roll) 正/负 |
| **Q/A** | Joint 6 (gripper) 开/关 |

#### 功能键
| 按键 | 功能 |
|------|------|
| **B** | 开始遥操作控制 |
| **S** | 开始录制 |
| **N** | 保存当前 episode |
| **D** | 丢弃当前 episode |
| **R** | 重置环境 |
| **ESC** | 中止录制，清空 buffer |
| **Ctrl+C** | 退出程序 |

### 双臂键盘设备 (bi-keyboard)

#### 左臂控制键（字母键）
| 按键 | 功能 |
|------|------|
| **T/G** | 左臂 Joint 1 正/负 |
| **Y/H** | 左臂 Joint 2 正/负 |
| **U/J** | 左臂 Joint 3 正/负 |
| **I/K** | 左臂 Joint 4 正/负 |
| **O/L** | 左臂 Joint 5 正/负 |
| **Q/A** | 左臂 Joint 6 正/负 |

#### 右臂控制键（数字键）
| 按键 | 功能 |
|------|------|
| **1/7** | 右臂 Joint 1 正/负 |
| **2/8** | 右臂 Joint 2 正/负 |
| **3/9** | 右臂 Joint 3 正/负 |
| **4/0** | 右臂 Joint 4 正/负 |
| **5/-** | 右臂 Joint 5 正/负 |
| **6/=** | 右臂 Joint 6 正/负 |

功能键与单臂键盘相同。

### SO101 Leader 设备

物理移动 Leader 臂来控制 Follower 机器人。

| 按键 | 功能 |
|------|------|
| **b** | 开始遥操作 |
| **s** | 开始录制 |
| **n** | 保存当前 episode |
| **d** | 丢弃当前 episode |
| **ESC** | 中止录制，清空 buffer |
| **Ctrl+C** | 退出程序 |

## 使用流程

### 1. 准备阶段

```bash
# 进入项目目录
cd /home/glzn/project/lehome

# 确保环境已激活
```

### 2. 单臂任务 - 键盘录制

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-SO101-Direct-v0 \
    --teleop_device keyboard \
    --record \
    --num_episode 10 \
    --step_hz 60
```

**操作步骤：**
1. 程序启动后，环境会初始化
2. 按 **B** 键开始遥操作（机器人开始响应控制）
3. 按 **S** 键开始录制
4. 使用控制键操作机器人完成任务
5. 完成后按 **N** 键保存此 episode
6. 重复步骤 4-5，直到录制完成指定数量
7. 如果某条 episode 失败，按 **D** 键丢弃重录
8. 如果要提前结束，按 **ESC** 键安全退出

### 3. 单臂任务 - SO101 Leader 录制

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-SO101-Direct-v0 \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --record \
    --num_episode 10
```

**首次使用需要校准：**
```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-SO101-Direct-v0 \
    --teleop_device so101leader \
    --port /dev/ttyACM0 \
    --recalibrate \
    --record \
    --num_episode 10
```

### 4. 双臂任务 - 键盘录制

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-v0 \
    --teleop_device bi-keyboard \
    --record \
    --num_episode 10
```

### 5. 双臂任务 - SO101 Leader 录制

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --record \
    --num_episode 10
```

### 6. 无深度图录制（加快速度）

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-SO101-Direct-v0 \
    --teleop_device keyboard \
    --record \
    --disable_depth \
    --num_episode 10
```

## 数据集格式

录制的数据会保存在 `Datasets/record/XXX/` 目录下，格式如下：

```
Datasets/record/024/
├── data/
│   └── chunk-000/
│       └── file-000.parquet          # 状态和动作数据
├── videos/
│   ├── observation.images.top_rgb/   # 视频编码
│   ├── observation.images.left_rgb/
│   └── observation.images.right_rgb/
└── meta/
    ├── info.json                      # 数据集元信息
    ├── object_initial_pose.jsonl      # 每个 episode 的物体初始位姿
    ├── episodes/
    │   └── chunk-000/
    │       └── file-000.parquet       # Episode 元数据
    └── tasks/
        └── file-000.parquet           # 任务元数据
```

### 数据内容

每个 frame 包含：

- **observation.state**: 关节状态（6 或 12 维，取决于单/双臂）
- **action**: 关节动作（6 或 12 维）
- **observation.images.xxx_rgb**: RGB 图像 (480x640x3)
- **observation.top_depth**: 深度图 (480x640)，可选
- **task**: 任务标签

## 常见问题

### Q1: 按 ESC 键没反应？

**A**: 确保你更新了所有 device 类文件：
- `source/lehome/lehome/devices/keyboard/se3_keyboard.py`
- `source/lehome/lehome/devices/keyboard/bi_keyboard.py`
- `source/lehome/lehome/devices/lerobot/so101_leader.py`

这些文件已经修改，支持 ESC 键功能。

### Q2: 录制到一半 Ctrl+C 退出，数据会损坏吗？

**A**: 不会！代码已添加 `KeyboardInterrupt` 异常处理，会自动清空当前 episode 的 buffer，保持已录制数据的完整性。

### Q3: 如何查看已录制的 episode 数量？

**A**: 录制过程中会在终端显示进度：
```
Episode 0 录制完成，进度: 1/10
Episode 1 录制完成，进度: 2/10
...
```

### Q4: SO101 Leader 设备连接失败？

**A**: 检查：
1. 串口设备是否正确：`ls /dev/ttyACM*`
2. 是否有权限：`sudo chmod 666 /dev/ttyACM0`
3. 是否已被其他程序占用
4. 尝试重新插拔 USB

### Q5: 任务和设备不匹配？

**A**: 脚本会自动验证：
- 单臂任务（不含 `Bi`）必须使用：`keyboard` 或 `so101leader`
- 双臂任务（含 `Bi`）必须使用：`bi-keyboard` 或 `bi-so101leader`

### Q6: 如何调整录制频率？

**A**: 使用 `--step_hz` 参数：
```bash
# 30 Hz 录制
python scripts/teleoperation/teleop_record.py --step_hz 30 ...

# 60 Hz 录制（默认）
python scripts/teleoperation/teleop_record.py --step_hz 60 ...
```

注意：频率越高，数据量越大，但动作会更平滑。

### Q7: 录制数据在哪里？

**A**: 默认保存在 `Datasets/record/` 目录，自动按编号递增（001, 002, ...）。会跳过已存在的编号。

### Q8: 如何重放录制的数据？

**A**: 使用 `replay.py` 脚本：
```bash
python scripts/tool/replay.py --dataset_path Datasets/record/024
```

详见 `scripts/tool/README_REPLAY.md`。

## 录制最佳实践

### 1. **录制前准备**
- ✅ 确保环境稳定运行
- ✅ 熟悉按键操作
- ✅ 先不开启 `--record` 练习几次

### 2. **录制中**
- ✅ 动作平滑，避免突然加速
- ✅ 失败的尝试用 **D** 键丢弃，不要保存
- ✅ 每条 episode 尽量完成任务目标
- ✅ 保持一致的操作风格

### 3. **数据质量**
- ✅ 录制 50-100 条高质量数据比 1000 条低质量数据更有用
- ✅ 多样化场景和初始状态
- ✅ 包含成功和部分失败的轨迹

### 4. **安全退出**
- ✅ 优先使用 **ESC** 键安全退出
- ✅ 避免在录制过程中强制关闭终端
- ✅ 录制完成后检查数据完整性

## 技术细节

### 数据采集流程

1. **初始化阶段**：按 S 之前
   - 环境正常运行
   - 可以遥操作但不录制
   - 用于调整初始状态

2. **录制阶段**：按 S 之后
   - 每个 step 调用 `dataset.add_frame(frame)`
   - 缓存在 episode buffer 中
   - 按 N 时调用 `dataset.save_episode()` 持久化
   - 按 D 或超时时调用 `dataset.clear_episode_buffer()`
   - 按 ESC 清空 buffer 并退出录制循环

3. **数据保存**：
   - 图像异步编码为视频
   - 状态和动作保存为 Parquet 格式
   - 物体初始位姿保存为 JSONL

### 坐标系说明

- **Joint State**: 归一化关节角度 [0, 1] 或 [-1, 1]
- **Action**: 关节增量或目标位置
- **相机**: OpenGL 坐标系

## 相关文档

- [Replay 工具文档](../tool/README_REPLAY.md)
- [LeRobot 数据集格式](https://github.com/huggingface/lerobot)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)

## 更新日志



**Happy Recording! 🎉**

