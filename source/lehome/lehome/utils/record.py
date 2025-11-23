from pathlib import Path
import time
import json
import numpy as np


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_next_experiment_path_with_gap(base_path: Path) -> Path:
    """查找第一个可用的编号（包括空缺位置）"""
    base_path.mkdir(parents=True, exist_ok=True)

    # 收集现有索引
    indices = set()
    for folder in base_path.iterdir():
        if folder.is_dir():
            try:
                indices.add(int(folder.name))
            except ValueError:
                continue

    # 找到第一个可用索引
    folder_index = 1
    while folder_index in indices:
        folder_index += 1

    return base_path / f"{folder_index:03d}"


def _ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    else:
        return obj


def append_episode_initial_pose(jsonl_path, episode_idx, object_initial_pose):
    object_initial_pose = _ndarray_to_list(object_initial_pose)
    rec = {"episode_idx": episode_idx, "object_initial_pose": object_initial_pose}
    with open(jsonl_path, "a") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
