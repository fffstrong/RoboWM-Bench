# Replay Script - 重构说明

## 概述

`replay.py` 已根据 LeRobotDataset 3.0 格式重构，参照 `teleop_record.py` 的代码结构。

## 主要改进

### 1. 代码结构优化
- **模块化函数**：将代码分解为清晰的函数模块
  - `validate_args()`: 参数验证
  - `load_dataset()`: 加载数据集
  - `load_initial_pose()`: 加载初始姿态
  - `create_replay_dataset()`: 创建输出数据集
  - `replay_episode()`: 回放单个 episode
  - `main()`: 主循环控制

### 2. 新增功能
- ✅ **可选保存**：通过 `--output_root` 控制是否保存回放结果
- ✅ **选择性保存**：`--save_successful_only` 只保存成功的 episode
- ✅ **多次回放**：`--num_replays` 支持每个 episode 回放多次
- ✅ **部分回放**：`--start_episode` 和 `--end_episode` 指定回放范围
- ✅ **统计信息**：显示成功率、回放次数等统计数据
- ✅ **更好的错误处理**：完善的参数验证和错误提示

### 3. 与 teleop_record.py 保持一致
- 相同的导入结构和依赖
- 相同的 features 处理逻辑
- 相同的辅助函数（RateLimiter, _ndarray_to_list）
- 相同的 LeRobotDataset 3.0 格式

## 使用示例

### 基本回放（不保存）
```bash
python scripts/tool/replay.py \
    --task LeIsaac-BiSO101-Direct-loftburger-v0 \
    --dataset_root Datasets/record/023
```

### 回放并保存所有 episode
```bash
python scripts/tool/replay.py \
    --task LeIsaac-BiSO101-Direct-loftburger-v0 \
    --dataset_root Datasets/record/023 \
    --output_root Datasets/replay
```

### 只保存成功的 episode
```bash
python scripts/tool/replay.py \
    --task LeIsaac-BiSO101-Direct-loftburger-v0 \
    --dataset_root Datasets/record/023 \
    --output_root Datasets/replay \
    --save_successful_only
```

### 多次回放每个 episode
```bash
python scripts/tool/replay.py \
    --task LeIsaac-BiSO101-Direct-loftburger-v0 \
    --dataset_root Datasets/record/023 \
    --num_replays 5 \
    --save_successful_only
```

### 回放指定范围
```bash
python scripts/tool/replay.py \
    --task LeIsaac-BiSO101-Direct-loftburger-v0 \
    --dataset_root Datasets/record/023 \
    --start_episode 5 \
    --end_episode 10
```

## 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务环境名称 | `LeIsaac-BiSO101-Direct-loftburger-v0` |
| `--dataset_root` | 输入数据集路径 | `Datasets/record/023` |
| `--output_root` | 输出数据集路径 | `None`（不保存） |
| `--num_replays` | 每个 episode 回放次数 | `1` |
| `--save_successful_only` | 只保存成功的 episode | `False` |
| `--disable_depth` | 禁用深度观测 | `False` |
| `--start_episode` | 起始 episode 索引 | `0` |
| `--end_episode` | 结束 episode 索引 | `None`（全部） |
| `--step_hz` | 步进频率（Hz） | `60` |

## 输出统计

运行结束后会显示：
```
============================================================
[Statistics]
  Total attempts: 50
  Total successes: 42
  Success rate: 84.0%
  Saved episodes: 42
============================================================
```

## 注意事项

1. 确保数据集是 LeRobotDataset 3.0 格式
2. `--task` 参数应与录制时使用的任务一致
3. 如果数据集没有 `object_initial_pose.jsonl`，将使用环境默认初始状态
4. 成功判断由环境的 `_get_success()` 方法决定





