# Evaluation for Imitation Learning Policy

This script evaluates a trained imitation learning policy in LeHome manipulation environments using Isaac Lab.

## Usage

```bash
python scripts/eval/eval_il.py [OPTIONS]
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_envs` | int | 1 | Number of environments to simulate. |
| `--max_steps` | int | 600 | Maximum number of steps per evaluation episode. |
| `--task` | str | "LeIsaac-BiSO101-Direct-Garment-v0" | Name of the task. |
| `--num_episodes` | int | 5 | Total number of evaluation episodes to run. |
| `--seed` | int | 42 | Seed for the environment. |
| `--step_hz` | int | 60 | Environment stepping rate in Hz. |
| `--sensitivity` | float | 1.0 | Sensitivity factor. |
| `--save_video` | flag | False | If set, save evaluation episodes as video. |
| `--video_dir` | str | "outputs/eval_videos" | Directory to save evaluation videos. |
| `--save_datasets` | flag | False | If set, save evaluation episodes dataset (only success). |
| `--eval_dataset_path` | str | "datasets/eval" | Path to save the evaluation dataset. |
| `--eval_task` | str | "eval" | Dataset task name when eval. |
| `--policy_path` | str | "outputs/train/..." | Path to the pretrained policy checkpoint. |
| `--dataset_root` | str | "datasets/fold" | Path of the train dataset (for metadata). |

## Examples

**Basic Evaluation:**
```bash
python scripts/eval/eval_il.py --task LeHome-BiSO101-Direct-Garment-v0 --policy_path outputs/train/act_so101_test/checkpoints/100000/pretrained_model --enable_cameras --device=cuda
```

**Evaluation with Video Saving:**
```bash
python scripts/eval/eval_il.py --task LeHome-BiSO101-Direct-Garment-v0 --policy_path /path/to/checkpoint --save_video --enable_cameras
```

**Evaluation generating a dataset of successful episodes:**
```bash
python scripts/eval/eval_il.py --task LeIsaac-BiSO101-Direct-Garment-v0 --policy_path /path/to/checkpoint --save_datasets  --eval_dataset_path datasets/my_eval --enable_cameras
```

## Notes
- Ensure you have the correct `policy_path` pointing to a trained model checkpoint.
- The script uses `lehome.utils.record` for shared utilities like `RateLimiter` and `get_next_experiment_path_with_gap`.




