#!/usr/bin/env python3
"""PKL (real robot transitions) -> LeRobot dataset v3.0.

observation.state: 9-dim float32, equals previous timestep's action within each episode;
  first frame of each episode is all zeros (no previous action).
action: 9-dim like tools/pkl_split.py (7 joints + gripper_pose*0.04 x2).
Video: same side image discovery as pkl_split; encode one concatenated mp4 (h264).
Task text: from CLI / JSON only, never from PKL.

Episode parquet and meta/stats.json: observation.state stats computed from the state column;
action stats from the action column.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Match outputs/auto_pipeline/011 meta/info.json (joint naming).
PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]
VEC_DIM = 9


def _write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _create_empty_dataset_info(
    *,
    codebase_version: str,
    fps: int,
    features: dict[str, Any],
    use_videos: bool,
    robot_type: str | None = None,
    chunks_size: int = 1000,
    data_files_size_in_mb: int = 100,
    video_files_size_in_mb: int = 200,
) -> dict[str, Any]:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "chunks_size": chunks_size,
        "data_files_size_in_mb": data_files_size_in_mb,
        "video_files_size_in_mb": video_files_size_in_mb,
        "fps": fps,
        "splits": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4" if use_videos else None,
        "features": features,
    }


def _write_info(info: dict[str, Any], out_root: Path) -> None:
    _write_json(info, out_root / "meta" / "info.json")


def _write_tasks(tasks_df: pd.DataFrame, out_root: Path) -> None:
    path = out_root / "meta" / "tasks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(path)


# ----- Same extraction logic as tools/pkl_split.py -----


def _get_gripper_pose(transition: dict, index: int) -> float:
    infos = transition.get("infos")
    if not isinstance(infos, dict):
        raise ValueError(f"Missing infos at index {index}")
    original_state_obs = infos.get("original_state_obs")
    if not isinstance(original_state_obs, dict):
        raise ValueError(f"Missing original_state_obs at index {index}")
    gp = original_state_obs.get("gripper_pose")
    if gp is None:
        raise ValueError(f"Missing gripper_pose at index {index}")
    if isinstance(gp, np.ndarray):
        gp = float(gp.reshape(-1)[0])
    else:
        gp = float(gp)
    return gp


def _get_joints(transition: dict, index: int) -> np.ndarray:
    if "joints" not in transition:
        raise ValueError(f"Missing joints at index {index}")
    joints = np.array(transition["joints"], dtype=np.float32).reshape(-1)
    if joints.shape[0] != 7:
        raise ValueError(f"Expected 7 joints at index {index}, got {joints.shape}")
    return joints


def _build_vec9(transition: dict, index: int) -> list[float]:
    """7 joints + [gripper_pose*0.04, gripper_pose*0.04] — same as pkl_split _build_action."""
    joints = _get_joints(transition, index)
    gripper_pose = _get_gripper_pose(transition, index)
    grip = gripper_pose * 0.04
    action = joints.tolist() + [grip, grip]
    return [float(x) for x in action]


def _find_side_image(obs: Any) -> np.ndarray | None:
    if isinstance(obs, dict):
        if "side" in obs and isinstance(obs["side"], np.ndarray):
            return obs["side"]
        if "images" in obs and isinstance(obs["images"], dict):
            if "side" in obs["images"] and isinstance(obs["images"]["side"], np.ndarray):
                return obs["images"]["side"]
            for key, value in obs["images"].items():
                if "side" in key.lower() and isinstance(value, np.ndarray):
                    return value
        for key, value in obs.items():
            if "side" in key.lower() and isinstance(value, np.ndarray):
                return value
    return None


def _extract_side_rgb_uint8(transition: dict) -> np.ndarray:
    """RGB uint8 (H,W,3) for ffmpeg; discovery order matches pkl_split._extract_frames."""
    side_img = None
    if "observations" in transition:
        side_img = _find_side_image(transition["observations"])
    if side_img is None and "next_observations" in transition:
        side_img = _find_side_image(transition["next_observations"])
    if side_img is None:
        raise ValueError("Missing side image in transition")
    out = side_img
    if out.ndim == 4 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dtype != np.uint8:
        if out.max() <= 1.0:
            out = (out * 255).astype(np.uint8)
        else:
            out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"Expected RGB (H,W,3), got {out.shape} {out.dtype}")
    return out


@dataclass(frozen=True)
class VideoSpec:
    feature_key: str
    height: int
    width: int
    channels: int = 3


def _iter_episodes(transitions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    buf: list[dict[str, Any]] = []
    for tr in transitions:
        buf.append(tr)
        if bool(tr.get("dones", False)):
            episodes.append(buf)
            buf = []
    if buf:
        episodes.append(buf)
    return episodes


def _stats_vector(x: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "count": [int(x.shape[0])],
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q10": np.quantile(x, 0.10, axis=0).tolist(),
        "q50": np.quantile(x, 0.50, axis=0).tolist(),
        "q90": np.quantile(x, 0.90, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def _stats_scalar(x: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "min": [float(np.min(x))],
        "max": [float(np.max(x))],
        "mean": [float(np.mean(x))],
        "std": [float(np.std(x))],
        "count": [int(x.shape[0])],
        "q01": [float(np.quantile(x, 0.01))],
        "q10": [float(np.quantile(x, 0.10))],
        "q50": [float(np.quantile(x, 0.50))],
        "q90": [float(np.quantile(x, 0.90))],
        "q99": [float(np.quantile(x, 0.99))],
    }


def _stats_int64_scalar(x: np.ndarray) -> dict[str, list[Any]]:
    """
    Match LeRobot episodes parquet schema for integer features:
    - min/max are list<int64>
    - mean/std/quantiles are list<double>
    - count is list<int64>
    """
    xi = np.asarray(x, dtype=np.int64).reshape(-1)
    xf = xi.astype(np.float64)
    return {
        "min": [int(np.min(xi))],
        "max": [int(np.max(xi))],
        "mean": [float(np.mean(xf))],
        "std": [float(np.std(xf))],
        "count": [int(xi.shape[0])],
        "q01": [float(np.quantile(xf, 0.01))],
        "q10": [float(np.quantile(xf, 0.10))],
        "q50": [float(np.quantile(xf, 0.50))],
        "q90": [float(np.quantile(xf, 0.90))],
        "q99": [float(np.quantile(xf, 0.99))],
    }


def _stats_image_rgb(frames_uint8: np.ndarray, max_frames: int = 500) -> dict[str, Any]:
    frames = np.asarray(frames_uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3) uint8, got {frames.shape} {frames.dtype}.")

    n = frames.shape[0]
    if n > max_frames:
        idx = np.linspace(0, n - 1, num=max_frames).astype(int)
        frames = frames[idx]

    x = frames.astype(np.float64) / 255.0
    ch = x.reshape(-1, 3)
    s = {
        "min": np.min(ch, axis=0),
        "max": np.max(ch, axis=0),
        "mean": np.mean(ch, axis=0),
        "std": np.std(ch, axis=0),
        "q01": np.quantile(ch, 0.01, axis=0),
        "q10": np.quantile(ch, 0.10, axis=0),
        "q50": np.quantile(ch, 0.50, axis=0),
        "q90": np.quantile(ch, 0.90, axis=0),
        "q99": np.quantile(ch, 0.99, axis=0),
    }
    count = int(ch.shape[0])

    def nest(v: np.ndarray) -> list[list[list[float]]]:
        return [[[float(v[0])]], [[float(v[1])]], [[float(v[2])]]]

    return {
        "min": nest(s["min"]),
        "max": nest(s["max"]),
        "mean": nest(s["mean"]),
        "std": nest(s["std"]),
        "count": [count],
        "q01": nest(s["q01"]),
        "q10": nest(s["q10"]),
        "q50": nest(s["q50"]),
        "q90": nest(s["q90"]),
        "q99": nest(s["q99"]),
    }


def _encode_video_ffmpeg(
    out_mp4: Path,
    frames: list[np.ndarray],
    *,
    fps: int,
    width: int,
    height: int,
    codec: str,
    crf: int,
    preset: str,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames provided for video encoding.")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for fr in frames:
            if fr.shape != (height, width, 3) or fr.dtype != np.uint8:
                raise ValueError(
                    f"Bad frame: shape={fr.shape} dtype={fr.dtype}, expected {(height, width, 3)} uint8."
                )
            proc.stdin.write(fr.tobytes())
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}. Command: {' '.join(cmd)}")


def _build_global_stats(
    state_mat: np.ndarray,
    action_mat: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    index: np.ndarray,
    task_index: np.ndarray,
    video_key: str | None,
    video_frames_uint8: np.ndarray | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mat in [("observation.state", state_mat), ("action", action_mat)]:
        s = _stats_vector(mat)
        out[name] = s
    out["timestamp"] = _stats_scalar(timestamp.astype(np.float64))
    out["frame_index"] = _stats_scalar(frame_index.astype(np.float64))
    out["episode_index"] = _stats_scalar(episode_index.astype(np.float64))
    out["index"] = _stats_scalar(index.astype(np.float64))
    out["task_index"] = _stats_scalar(task_index.astype(np.float64))
    if video_key and video_frames_uint8 is not None and video_frames_uint8.size:
        out[video_key] = _stats_image_rgb(video_frames_uint8, max_frames=500)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True, help="Input PKL path.")
    ap.add_argument("--out", type=str, required=True, help="Output dataset root directory.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunks-size", type=int, default=1000)

    ap.add_argument(
        "--task-spec",
        type=str,
        default=None,
        help='JSON: {"tasks":["..."], "episode_task_indices":[0,0,...] (optional)}',
    )
    ap.add_argument("--task-description", type=str, default=None, help="Single task string if no --task-spec.")

    ap.add_argument("--video", action="store_true", help="Write videos/ (side camera, one stream).")
    ap.add_argument("--video-key", type=str, default="side_rgb", help="Feature suffix: observation.images.<key>")
    ap.add_argument("--video-codec", type=str, default="libx264")
    ap.add_argument("--video-crf", type=int, default=23)
    ap.add_argument("--video-preset", type=str, default="medium")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)
    if not isinstance(transitions, list) or not transitions or not isinstance(transitions[0], dict):
        raise ValueError("Expected PKL to be a list of transition dicts.")

    episodes = _iter_episodes(transitions)
    n_trans = len(transitions)

    if args.task_spec:
        spec = json.loads(Path(args.task_spec).read_text(encoding="utf-8"))
        tasks: list[str] = list(spec["tasks"])
        episode_task_indices = spec.get("episode_task_indices")
        if episode_task_indices is None:
            episode_task_indices = [0 for _ in range(len(episodes))]
        if len(episode_task_indices) != len(episodes):
            raise ValueError(
                f"episode_task_indices len={len(episode_task_indices)} must match episodes={len(episodes)}"
            )
    else:
        task = args.task_description or "task"
        tasks = [task]
        episode_task_indices = [0 for _ in range(len(episodes))]

    tasks_df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name=None))
    _write_tasks(tasks_df, out_root)

    ft_key = f"observation.images.{args.video_key}"
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (VEC_DIM,),
            "names": list(PANDA_JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (VEC_DIM,),
            "names": list(PANDA_JOINT_NAMES),
        },
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    video_spec: VideoSpec | None = None
    if args.video:
        sample_rgb = _extract_side_rgb_uint8(transitions[0])
        h, w, c = sample_rgb.shape
        if c != 3:
            raise ValueError("Expected 3 RGB channels.")
        features[ft_key] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "h264" if args.video_codec == "libx264" else args.video_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": args.fps,
                "video.channels": c,
                "has_audio": False,
            },
        }
        video_spec = VideoSpec(feature_key=ft_key, height=h, width=w, channels=c)

    info = _create_empty_dataset_info(
        codebase_version="v3.0",
        fps=int(args.fps),
        features={
            k: {**v, "shape": list(v["shape"]) if isinstance(v.get("shape"), tuple) else v.get("shape")}
            for k, v in features.items()
        },
        use_videos=bool(args.video),
        robot_type=None,
        chunks_size=int(args.chunks_size),
    )

    all_state: list[list[float]] = []
    all_action: list[list[float]] = []
    all_timestamp: list[float] = []
    all_frame_index: list[int] = []
    all_episode_index: list[int] = []
    all_index: list[int] = []
    all_task_index: list[int] = []

    global_i = 0
    for ep_i, ep in enumerate(episodes):
        prev_act = [0.0] * VEC_DIM
        for t_i, tr in enumerate(ep):
            act = _build_vec9(tr, global_i)
            all_state.append(list(prev_act))
            all_action.append(act)
            prev_act = act
            all_timestamp.append(np.float32(t_i / args.fps).item())
            all_frame_index.append(t_i)
            all_episode_index.append(ep_i)
            all_index.append(global_i)
            all_task_index.append(int(episode_task_indices[ep_i]))
            global_i += 1

    def fixed_list(arr: list[list[float]], length: int) -> pa.Array:
        values = pa.array(arr, type=pa.list_(pa.float32()))
        return pa.FixedSizeListArray.from_arrays(values.values, length)

    table = pa.table(
        {
            "observation.state": fixed_list(all_state, VEC_DIM),
            "action": fixed_list(all_action, VEC_DIM),
            "timestamp": pa.array(all_timestamp, type=pa.float32()),
            "frame_index": pa.array(all_frame_index, type=pa.int64()),
            "episode_index": pa.array(all_episode_index, type=pa.int64()),
            "index": pa.array(all_index, type=pa.int64()),
            "task_index": pa.array(all_task_index, type=pa.int64()),
        }
    )

    data_path = out_root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_path)

    all_frames_rgb: list[np.ndarray] | None = None
    if video_spec is not None:
        all_frames_rgb = [_extract_side_rgb_uint8(tr) for tr in transitions]
        for fr in all_frames_rgb:
            if fr.shape != (h, w, 3):
                raise ValueError(f"Inconsistent frame shape {fr.shape} vs {(h, w, 3)}")
        vpath = out_root / "videos" / video_spec.feature_key / "chunk-000" / "file-000.mp4"
        _encode_video_ffmpeg(
            vpath,
            all_frames_rgb,
            fps=args.fps,
            width=w,
            height=h,
            codec=args.video_codec,
            crf=args.video_crf,
            preset=args.video_preset,
        )

    # Episodes parquet: vector stats from actual state/action columns per episode.
    episode_rows: list[dict[str, Any]] = []
    cursor = 0
    for ep_i, ep in enumerate(episodes):
        length = len(ep)
        start = cursor
        end = cursor + length
        cursor = end

        row: dict[str, Any] = {
            "episode_index": ep_i,
            "tasks": [tasks[int(episode_task_indices[ep_i])]],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": end,
            # Required by lerobot aggregate/merge tooling to locate the consolidated
            # meta/episodes parquet file holding this row.
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        if video_spec is not None:
            from_ts = start / args.fps
            to_ts = (end - 1) / args.fps if length > 0 else start / args.fps
            row[f"videos/{video_spec.feature_key}/chunk_index"] = 0
            row[f"videos/{video_spec.feature_key}/file_index"] = 0
            row[f"videos/{video_spec.feature_key}/from_timestamp"] = float(from_ts)
            row[f"videos/{video_spec.feature_key}/to_timestamp"] = float(to_ts)

        ep_state = np.asarray(all_state[start:end], dtype=np.float64)
        ep_action = np.asarray(all_action[start:end], dtype=np.float64)
        for k, stat in _stats_vector(ep_state).items():
            row[f"stats/observation.state/{k}"] = stat
        for k, stat in _stats_vector(ep_action).items():
            row[f"stats/action/{k}"] = stat

        ep_ts = np.array([t / args.fps for t in range(length)], dtype=np.float32)
        for k, stat in _stats_scalar(ep_ts).items():
            row[f"stats/timestamp/{k}"] = stat
        for k, stat in _stats_int64_scalar(np.arange(length, dtype=np.int64)).items():
            row[f"stats/frame_index/{k}"] = stat
        for k, stat in _stats_int64_scalar(np.full((length,), ep_i, dtype=np.int64)).items():
            row[f"stats/episode_index/{k}"] = stat
        for k, stat in _stats_int64_scalar(np.arange(start, end, dtype=np.int64)).items():
            row[f"stats/index/{k}"] = stat
        for k, stat in _stats_int64_scalar(np.full((length,), int(episode_task_indices[ep_i]), dtype=np.int64)).items():
            row[f"stats/task_index/{k}"] = stat

        if video_spec is not None and all_frames_rgb is not None:
            ep_frames = np.stack([all_frames_rgb[start + j] for j in range(length)], axis=0)
            img_stats = _stats_image_rgb(ep_frames, max_frames=100)
            for k, stat in img_stats.items():
                row[f"stats/{video_spec.feature_key}/{k}"] = stat

        episode_rows.append(row)

    ep_df = pd.DataFrame(episode_rows)
    ep_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    ep_df.to_parquet(ep_path, index=False)

    # Global stats.json (like reference 011): from full frame table + sampled video pixels.
    state_mat = np.asarray(all_state, dtype=np.float64)
    action_mat = np.asarray(all_action, dtype=np.float64)
    ts_arr = np.asarray(all_timestamp, dtype=np.float64)
    fi_arr = np.asarray(all_frame_index, dtype=np.int64)
    ei_arr = np.asarray(all_episode_index, dtype=np.int64)
    ix_arr = np.asarray(all_index, dtype=np.int64)
    tk_arr = np.asarray(all_task_index, dtype=np.int64)
    vid_stack = None
    if all_frames_rgb is not None:
        vid_stack = np.stack(all_frames_rgb, axis=0)

    global_stats = _build_global_stats(
        state_mat,
        action_mat,
        ts_arr,
        fi_arr,
        ei_arr,
        ix_arr,
        tk_arr,
        video_key=video_spec.feature_key if video_spec else None,
        video_frames_uint8=vid_stack,
    )
    _write_json(global_stats, out_root / "meta" / "stats.json")

    info["total_episodes"] = int(len(episodes))
    info["total_frames"] = int(n_trans)
    info["total_tasks"] = int(len(tasks))
    info["splits"] = {"train": f"0:{len(episodes)}"}
    _write_info(info, out_root)

    (out_root / "meta" / "source.json").write_text(
        json.dumps(
            {"source_pkl": os.fspath(pkl_path), "num_transitions": n_trans, "num_episodes": len(episodes)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
