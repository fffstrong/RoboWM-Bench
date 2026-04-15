"""Convert a flat folder of numbered mp4 videos into a LeRobot-style dataset.

This is a simplified standalone tool intended for the workflow:
  - Input: a directory containing many videos named 0.mp4, 1.mp4, ...
  - Output: LeRobot dataset directory with:
      data/chunk-XXX/episode_XXXXXX.parquet
      videos/chunk-XXX/<view_key>/episode_XXXXXX.mp4
      meta/ with 5 files:
        tasks.jsonl, episodes.jsonl, info.json, modality.json, stats.json

Design notes:
- We DO NOT infer anything from the input/output folder *paths*.
- We may read metadata from the video files (ffprobe) to estimate fps/frames.
- We do not require labels/ instructions.

Example:
  python GR00T-Dreams/IDM_dump/video_folder_to_lerobot.py \
    --video_folder /path/to/videos \
    --output_dir /path/to/out.data \
    --view_key observation.images.video
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tqdm import tqdm


CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_int_stem(p: Path) -> int | None:
    try:
        return int(p.stem)
    except Exception:
        return None


def _write_json(path: Path, obj: Any, indent: int = 4) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _ffprobe_stream(path: Path) -> dict | None:
    """Return ffprobe stream info for v:0 or None."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,r_frame_rate,avg_frame_rate,height,width,codec_name,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        data = json.loads(out)
        streams = data.get("streams") or []
        return streams[0] if streams else None
    except Exception:
        return None


def _ffprobe_count_frames(path: Path) -> int | None:
    """Accurately count frames using ffprobe.

    This makes ffprobe read frames (effectively decoding timestamps), which is slower
    than metadata-only probing, but gives the correct frame count for most mp4 files.

    Returns:
        int frame count, or None if ffprobe fails.
    """
    # Prefer nb_read_frames as it's widely used.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        data = json.loads(out)
        streams = data.get("streams") or []
        if not streams:
            return None
        val = streams[0].get("nb_read_frames")
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _decode_count_frames_opencv(path: Path) -> int | None:
    """Count frames by fully decoding using OpenCV.

    Slowest option, but acts as a fallback if ffprobe can't provide nb_read_frames.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        return n
    except Exception:
        return None


def _parse_fps(rate_str: str | None) -> float | None:
    if not rate_str:
        return None
    try:
        num, den = rate_str.split("/")
        num_i = int(num)
        den_i = int(den)
        if den_i == 0:
            return None
        return num_i / den_i
    except Exception:
        return None


def _estimate_num_frames_and_fps(video_path: Path, fallback_fps: int) -> tuple[int, int]:
    """Estimate (num_frames, fps) from ffprobe.

    We don't decode the video. We only use container metadata.
    """
    stream = _ffprobe_stream(video_path)
    fps = None
    num_frames = None

    if stream:
        fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate"))
        try:
            duration = float(stream.get("duration")) if stream.get("duration") is not None else None
        except Exception:
            duration = None

        if fps and duration is not None:
            num_frames = int(max(1.0, duration * fps))

    fps_i = int(round(fps)) if fps else int(fallback_fps)
    frames_i = int(num_frames) if num_frames else 93
    return max(1, frames_i), max(1, fps_i)


def _get_num_frames_and_fps(
    video_path: Path,
    fallback_fps: int,
    mode: Literal["accurate", "estimate"] = "accurate",
) -> tuple[int, int]:
    """Get (num_frames, fps).

    - accurate: use ffprobe -count_frames; fallback to metadata estimate; last fallback decode with OpenCV.
    - estimate: metadata-only (fast, may be wrong).
    """
    if mode == "estimate":
        return _estimate_num_frames_and_fps(video_path, fallback_fps=fallback_fps)

    # accurate
    _, fps_i = _estimate_num_frames_and_fps(video_path, fallback_fps=fallback_fps)

    n = _ffprobe_count_frames(video_path)
    if n is not None and n > 0:
        return int(n), int(fps_i)

    # fallback to estimate
    n_est, fps_i2 = _estimate_num_frames_and_fps(video_path, fallback_fps=fps_i)
    if n_est is not None and n_est > 0:
        # If you want *strict* accuracy, keep going to OpenCV by passing --frame_count_mode decode.
        return int(n_est), int(fps_i2)

    n_cv = _decode_count_frames_opencv(video_path)
    if n_cv is not None and n_cv > 0:
        return int(n_cv), int(fps_i)

    return 93, int(fps_i)


def _get_video_feature_metadata(video_path: Path) -> dict | None:
    stream = _ffprobe_stream(video_path)
    if not stream:
        return None

    height = stream.get("height")
    width = stream.get("width")
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate"))

    if height is None or width is None:
        return None

    return {
        "dtype": "video",
        "shape": [int(height), int(width), 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(fps) if fps else None,
            "video.codec": stream.get("codec_name"),
            "video.pix_fmt": stream.get("pix_fmt"),
            "video.is_depth_map": False,
        },
    }


def convert(
    video_folder: Path,
    output_dir: Path,
    *,
    view_key: str,
    fps: int | None,
    fixed_num_frames: int | None,
    frame_count_mode: Literal["accurate", "estimate"],
) -> Path:
    if not video_folder.exists() or not video_folder.is_dir():
        raise ValueError(f"video_folder not found or not a directory: {video_folder}")

    _ensure_dir(output_dir)
    meta_dir = output_dir / "meta"
    _ensure_dir(meta_dir)

    mp4s = sorted(
        video_folder.glob("*.mp4"),
        key=lambda p: (_safe_int_stem(p) is None, _safe_int_stem(p) or 0, p.name),
    )
    if not mp4s:
        raise ValueError(f"No .mp4 files found in: {video_folder}")

    # One dummy task.
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": "task"}])

    episodes_jsonl: list[dict] = []
    total_frames = 0

    # Decide a dataset-level fps used for timestamps.
    if fps is None:
        # Try from the first video; else fallback to 16.
        _, fps0 = _estimate_num_frames_and_fps(mp4s[0], fallback_fps=16)
        fps_used = fps0
    else:
        fps_used = int(fps)

    # Copying tasks
    copy_tasks: list[tuple[Path, Path]] = []

    for episode_index, src in enumerate(tqdm(mp4s, desc="Converting")):
        episode_chunk = episode_index // CHUNKS_SIZE
        if fixed_num_frames is not None:
            num_frames = int(fixed_num_frames)
        else:
            num_frames, _ = _get_num_frames_and_fps(src, fallback_fps=fps_used, mode=frame_count_mode)

        num_frames = max(1, int(num_frames))

        episode_data = {
            "observation.state": [np.zeros(9, dtype=np.float32)] * num_frames,
            "action": [np.zeros(9, dtype=np.float32)] * num_frames,
            "timestamp": [i / float(fps_used) for i in range(num_frames)],
            "episode_index": [episode_index] * num_frames,
            "index": np.arange(total_frames, total_frames + num_frames),
            "task_index": [0] * num_frames,
            "annotation.human.action.task_description": [[0]] * num_frames,
        }

        parquet_path = output_dir / DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
        _ensure_dir(parquet_path.parent)

        # Lazy import to avoid hard dependency until runtime.
        import pandas as pd

        pd.DataFrame(episode_data).to_parquet(parquet_path)

        dst_video = output_dir / VIDEO_PATH.format(episode_chunk=episode_chunk, video_key=view_key, episode_index=episode_index)
        _ensure_dir(dst_video.parent)
        copy_tasks.append((src, dst_video))

        episodes_jsonl.append(
            {
                "episode_index": episode_index,
                "tasks": ["task"],
                "length": num_frames,
                "video_id": src.stem,
            }
        )

        total_frames += num_frames

    # Copy videos last.
    for src, dst in tqdm(copy_tasks, desc="Copying videos"):
        shutil.copy2(src, dst)

    _write_jsonl(meta_dir / "episodes.jsonl", episodes_jsonl)

    info = {
        "robot_type": "dream",
        "total_episodes": len(mp4s),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "chunks_size": CHUNKS_SIZE,
        "total_chunks": (len(mp4s) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "fps": fps_used,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32", "shape": (9,), "names": [f"motor_{i}" for i in range(9)]},
            "action": {"dtype": "float32", "shape": (9,), "names": [f"motor_{i}" for i in range(9)]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": (1,)},
        },
    }

    meta = _get_video_feature_metadata(mp4s[0])
    if meta is not None:
        # Drop None entries in video_info.
        vi = meta.get("video_info") or {}
        meta["video_info"] = {k: v for k, v in vi.items() if v is not None}
        info["features"][view_key] = meta

    _write_json(meta_dir / "info.json", info, indent=4)

    # Minimal modality & stats.
    # Extract video key name from view_key (e.g., "observation.images.top_rgb" -> "top_view")
    # Try to extract a meaningful name, fallback to a simple extraction
    video_key_name = "top_view"  # default
    if "top" in view_key.lower():
        video_key_name = "top_view"
    elif "front" in view_key.lower():
        video_key_name = "front_view"
    elif "side" in view_key.lower():
        video_key_name = "side_view"
    else:
        # Extract last part of the key path
        parts = view_key.split(".")
        if len(parts) > 0:
            last_part = parts[-1]
            # Remove common suffixes and convert to view name
            last_part = last_part.replace("_rgb", "").replace("_video", "")
            video_key_name = f"{last_part}_view" if last_part else "video"

    modality = {
        "state": {
            "lerobot": {
                "start": 0,
                "end": 9
            }
        },
        "action": {
            "lerobot": {
                "start": 0,
                "end": 9
            }
        },
        "video": {
            video_key_name: {
                "original_key": view_key
            }
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {}
        }
    }
    _write_json(meta_dir / "modality.json", modality, indent=4)

    stats = {
        "observation.state": {
            "mean": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "std": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "min": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "max": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "q01": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "q99": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
        "action": {
            "mean": [
                -0.007157385814934969,
                -0.04297269135713577,
                0.18464387953281403,
                -2.071974754333496,
                -0.08856634795665741,
                2.121952772140503,
                1.0351274013519287,
                0.016601143404841423,
                0.016601143404841423,
            ],
            "std": [
                0.055407263338565826,
                0.3608388602733612,
                0.32370725274086,
                0.4307419955730438,
                0.2884117662906647,
                0.3267955183982849,
                0.45014071464538574,
                0.018068447709083557,
                0.018068447709083557,
            ],
            "min": [
                -0.21057605743408203,
                -1.2948634624481201,
                -0.7374020218849182,
                -3.0715203285217285,
                -1.604585886001587,
                1.2275254726409912,
                -0.7712269425392151,
                8.56949991430156e-05,
                8.56949991430156e-05,
            ],
            "max": [
                0.23108357191085815,
                0.9625616073608398,
                1.0994420051574707,
                -0.7130497694015503,
                0.8721204996109009,
                3.751879930496216,
                2.6991124153137207,
                0.04009442403912544,
                0.04009442403912544,
            ],
            "q01": [
                -0.13043640464544295,
                -0.7830458009243011,
                -0.5378071713447571,
                -2.8936278104782103,
                -1.214939262866974,
                1.5591320896148682,
                -0.2676623982191086,
                8.602333400631323e-05,
                8.602333400631323e-05,
            ],
            "q99": [
                0.1470453494787216,
                0.7893036556243891,
                0.8564369809627531,
                -1.0925735569000243,
                0.4652929925918579,
                3.1386306285858154,
                2.186361250877379,
                0.039988044649362564,
                0.039988044649362564,
            ],
        },
        "timestamp": {
            "mean": [5.417086124420166],
            "std": [3.686272382736206],
            "min": [0.0],
            "max": [21.299999237060547],
            "q01": [0.05000000074505806],
            "q99": [15.699999809265137],
        },
        "frame_index": {
            "mean": [108.34171154997007],
            "std": [73.72545005535834],
            "min": [0.0],
            "max": [426.0],
            "q01": [1.0],
            "q99": [314.0],
        },
        "episode_index": {
            "mean": [150.96047034467142],
            "std": [81.82017348706363],
            "min": [0.0],
            "max": [309.0],
            "q01": [3.0],
            "q99": [306.0],
        },
        "index": {
            "mean": [30913.0],
            "std": [17847.91754425896],
            "min": [0.0],
            "max": [61826.0],
            "q01": [618.26],
            "q99": [61207.74],
        },
        "task_index": {
            "mean": [0.0],
            "std": [0.0],
            "min": [0.0],
            "max": [0.0],
            "q01": [0.0],
            "q99": [0.0],
        },
        "annotation.human.action.task_description": {
            "mean": [0.0],
            "std": [0.0],
            "min": [0.0],
            "max": [0.0],
            "q01": [0.0],
            "q99": [0.0],
        },
    }
    _write_json(meta_dir / "stats.json", stats, indent=4)

    print(f"Done. Wrote dataset to: {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a numbered mp4 folder into a LeRobot-style dataset")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing 0.mp4, 1.mp4, ...")
    parser.add_argument("--output_dir", type=str, required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--view_key", type=str, default="observation.images.video", help="View folder name under videos/")
    parser.add_argument("--fps", type=int, default=None, help="FPS for timestamps (default: infer from first video or use 16)")
    parser.add_argument("--fixed_num_frames", type=int, default=None, help="Use a fixed number of frames per episode (skip probing)")
    parser.add_argument(
        "--frame_count_mode",
        type=str,
        default="accurate",
        choices=["accurate", "estimate"],
        help="How to get num_frames. accurate=ffprobe -count_frames (slow, correct). estimate=duration*fps (fast, may be wrong).",
    )

    args = parser.parse_args()

    convert(
        video_folder=Path(args.video_folder),
        output_dir=Path(args.output_dir),
        view_key=args.view_key,
        fps=args.fps,
        fixed_num_frames=args.fixed_num_frames,
        frame_count_mode=args.frame_count_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
