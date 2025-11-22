import argparse
from pathlib import Path
import sys
import traceback
import os

# Try to import lerobot, handling the case where it might not be in the path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def check_directory_structure(dataset_root: Path):
    print(f"\n[Structure Check] Inspecting {dataset_root}...")
    
    if not dataset_root.exists():
        print(f"  [X] Root directory does not exist: {dataset_root}")
        return False
    
    # Check critical metadata files
    required_files = [
        "meta/info.json",
        "meta/stats.json",
    ]
    
    structure_issues = False
    for rel_path in required_files:
        p = dataset_root / rel_path
        if p.exists():
            print(f"  [OK] Found {rel_path}")
        else:
            print(f"  [X] Missing {rel_path}")
            structure_issues = True
            
    # Check meta/episodes directory (Specific to v3.0 structure issues we discussed)
    episodes_dir = dataset_root / "meta/episodes"
    if episodes_dir.exists():
        # Check if it has any content (chunks)
        chunks = list(episodes_dir.glob("chunk-*"))
        if chunks:
            print(f"  [OK] Found meta/episodes directory with {len(chunks)} chunks.")
        else:
            print(f"  [X] Found meta/episodes directory BUT it is empty.")
            structure_issues = True
    else:
        print(f"  [X] Missing meta/episodes directory. (This is likely the cause of failure for v3.0 datasets)")
        structure_issues = True
        
    # Check data directory
    data_dir = dataset_root / "data"
    if data_dir.exists():
        chunks = list(data_dir.glob("chunk-*"))
        if chunks:
            print(f"  [OK] Found data directory with {len(chunks)} chunks.")
        else:
            print(f"  [Warning] Found data directory BUT it is empty.")
    else:
        print(f"  [Warning] Missing data directory.")

    return not structure_issues

def main():
    parser = argparse.ArgumentParser(description="Debug LeRobotDataset loading.")
    # Default to the problematic dataset 026 mentioned by user
    parser.add_argument("--dataset_root", type=str, default="Datasets/record/027", help="Path to the dataset root directory.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).absolute()
    
    print(f"Debug script started for: {dataset_root}")
    
    # 1. Check Structure
    structure_ok = check_directory_structure(dataset_root)
    if not structure_ok:
        print("\n[Warning] The dataset structure seems incomplete. LeRobotDataset will likely try to download from HuggingFace as a fallback.")
    else:
        print("\n[Info] Dataset structure looks valid on surface.")

    # 2. Attempt Load
    print("\n[Action] Attempting to initialize LeRobotDataset...")
    print("Note: If local loading fails (e.g. due to missing metadata), LeRobotDataset automatically tries to download from Hugging Face.")
    print("      If you are offline, this will result in a ConnectionError.\n")
    

    # We use the same initialization as in replay.py
    dataset = LeRobotDataset(repo_id="replay_source", root=dataset_root)
    
    print("\n[Success] Dataset loaded successfully!")
    print(f"  - Repo ID: {dataset.repo_id}")
    print(f"  - Total Episodes: {dataset.num_episodes}")
    print(f"  - Total Frames: {dataset.num_frames}")
    print(f"  - Features: {list(dataset.features.keys())}")
    
    # Try to access one frame to ensure data is readable
    if dataset.num_frames > 0:
        print("  - Attempting to read frame 0...")
        frame = dataset[0]
        print("  - Frame 0 read successfully.")
        

if __name__ == "__main__":
    main()

