import os
from typing import Optional

import cv2
import pandas as pd


def split_video_by_parquet(
    video_path: str,
    parquet_path: str,
    output_dir: str,
    n_episodes: Optional[int] = None,
    filename_format: str = "episode_{episode_index:06d}.mp4",
):
    """
    根据 parquet 文件中每个 episode_index 的 length，从一个长视频中截取多个小视频。

    参数：
        video_path: 原始长视频路径
        parquet_path: parquet 文件路径（需要包含 episode_index, length 两列）
        output_dir: 输出小视频目录
        n_episodes: 只导出前 n 个 episode；为 None 时导出全部
        filename_format: 输出文件命名格式（默认为 episode_index 零填充到 6 位）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取 parquet
    df = pd.read_parquet(parquet_path)
    # 只保留需要的字段，并按 episode_index 排序
    df = df[["episode_index", "length"]].dropna().copy()
    df = df.sort_values("episode_index").reset_index(drop=True)

    if n_episodes is not None:
        df = df.head(n_episodes)

    # 2. 计算每个 episode 的起始帧（按照顺序累加）
    # start_frame 为前面所有 length 的累积和
    lengths = df["length"].astype(int).tolist()
    start_frames = []
    cur = 0
    for L in lengths:
        start_frames.append(cur)
        cur += L
    df["start_frame"] = start_frames

    # 3. 打开视频，获取基础信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 4. 逐个 episode 截取
    for idx, row in df.iterrows():
        episode_index = int(row["episode_index"])
        start_frame = int(row["start_frame"])
        length = int(row["length"])

        # 设置起始帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out_name = filename_format.format(episode_index=episode_index)
        out_path = os.path.join(output_dir, out_name)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print(f"导出 episode_index={episode_index}, "
              f"start_frame={start_frame}, length={length}, "
              f"output={out_path}")

        frames_written = 0
        while frames_written < length:
            ret, frame = cap.read()
            if not ret:
                # 视频提前结束
                print(f"在帧 {start_frame + frames_written} 处读帧失败，提前结束该 episode")
                break
            writer.write(frame)
            frames_written += 1

        writer.release()

    cap.release()
    print("所有 episode 导出完成。")


if __name__ == "__main__":
    # 示例：你可以根据自己的真实路径修改
    video_path = "/home/feng/lehome_1/Datasets/001/videos/observation.images.top_rgb/chunk-000/file-000.mp4"
    parquet_path = "/home/feng/lehome_1/Datasets/001/meta/episodes/chunk-000/file-000.parquet"
    output_dir = "/home/feng/lehome_1/eval_videos/sim/pick"

    # 例如只导出前 100 个 episode；如果要全部导出，把 n_episodes 改为 None
    split_video_by_parquet(
        video_path=video_path,
        parquet_path=parquet_path,
        output_dir=output_dir,
        n_episodes=10,
    )