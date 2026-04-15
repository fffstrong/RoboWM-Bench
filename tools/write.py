#!/usr/bin/env python3
"""遍历文件夹下所有 json 文件，按顺序将文件名（不含 .json 后缀）写入 file_indices.txt，每行一个。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="将文件夹下所有 json 文件名写入 file_indices.txt")
    parser.add_argument("folder", type=str, help="目标文件夹路径")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"不是有效文件夹: {folder}")

    json_files = sorted(folder.glob("*.json"))
    names = [f.stem for f in json_files]  # stem = 文件名不含后缀

    out_path = folder / "file_indices.txt"
    out_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    print(f"已写入 {len(names)} 个文件名到 {out_path}")


if __name__ == "__main__":
    main()
