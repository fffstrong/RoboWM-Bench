from pathlib import Path


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
