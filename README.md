<div align="center">
  <h1><b>RoboWM-Bench</b></h1>
  <p><a href="https://robowm-bench.github.io/RoboWM-Bench/">Project Page</a></p>
  <p>
    <a href="img/teaser.pdf">
      <img src="img/teaser.png" alt="RoboWM-Bench teaser" width="900" />
    </a>
  </p>
</div>

RoboWM-Bench provides Isaac Lab simulation tasks (with a LeHome-style layout) and tooling to:
- replay robot trajectories to generate masked RGB/depth data for IDM training
- run IDM inference and convert model outputs into action JSON trajectories
- evaluate Franka tasks in simulation and optionally record cameras / per-step scores

## Table of Contents
- [Installation](#installation)
- [Project Layout](#project-layout)
- [Replay: Generate IDM Training Data](#replay-generate-idm-training-data)
- [World Model Inputs](#world-model-inputs)
- [IDM](#idm)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)

## Installation

The recommended setup is to create a clean Conda environment (Python 3.11), install a CUDA-matched PyTorch build, install this repo in editable mode, then install `lerobot` and NVIDIA IsaacSim/IsaacLab (with IsaacLab pinned to a version compatible with IsaacSim 5.1). The commands below are intended to be run on Linux with NVIDIA drivers already working (i.e., `nvidia-smi` succeeds).

```bash
# Create and activate a Conda environment
conda create -n RWMBench python=3.11
conda activate RWMBench

# Install PyTorch (CUDA 12.8 build)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install RoboWM-Bench
git clone https://github.com/fffstrong/RoboWM-Bench.git
cd RoboWM-Bench
python -m pip install -e source/lehome

# Install lerobot==0.4.3
pip install "lerobot==0.4.3"
pip install "lerobot[all]==0.4.3"          # All available features

# Install IsaacSim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install IsaacLab (pinned for IsaacSim 5.1)
sudo apt install cmake build-essential
cd IsaacLab_5_1
git checkout v2.3.0
./isaaclab.sh --install

# Optional: extra utilities
pip install open3d
```

## Project Layout

We follow the LeHome project structure. Public tasks are organized under `source/lehome/lehome/tasks/`.

Each task follows the same pattern (example: `Task00_Pick/`):
- **`Pick.py`**: environment logic (reset, randomization, observations, success criteria)
- **`Pick_cfg.py`**: Isaac Lab configuration (robot, objects, scene, cameras)
- **`__init__.py`**: Gym task registration (you can look up the task name here)

## Replay: Generate IDM Training Data

We provide a replay script that can replay trajectories in IsaacLab for different robot arms and export camera data (e.g., masked RGB and depth) for IDM training.

- **Task code reference**: `source/lehome/lehome/tasks/franka_IDM`
- **Replay script**: `sh/replay_franka.sh`

Command:

```bash
python scripts/eval/replay_franka.py \
  --task Franka-IDM \
  --json_root /your_json_root \  # The input must be a folder path that contains many JSON files and an index TXT file; each JSON is one motion trajectory (see the `replay_json` folder for an example)
  --output_root /your_output_root \
  --enable_cameras \
```

## World Model Inputs

World model inputs (RGB frames and prompts) are under the `wm_inputs` folder.

## IDM

Please refer to NVIDIA DreamGen (GR00T-dreams) for the IDM section: `https://github.com/nvidia/GR00T-dreams`.

- Replace `data_config_idm.py` with `IDM/data_config_idm.py`.
- `IDM/discard_trash` is a reference input dataset. Make sure your dataset `meta` matches the reference, especially **`modality`** and **`stats`**.
- IDM weights (open-sourced): `https://huggingface.co/fffstrong/robowmbench-idm`.

IDM inference command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python IDM_dump/dump_idm_actions.py \
    --checkpoint "checkpoint_path" \
    --dataset "IDM/discard_trash" \
    --output_dir "your_output_path" \
    --num_gpus 8 \
    --video_indices "0 16"
```

## Evaluation

After IDM produces outputs, run `sh/parquet2action.sh` to convert the predicted actions into trajectory JSON files:

```bash
python tools/parquet_actions_to_json.py \
    --input_dir /your_input_dir \ # Folder path that contains parquet files
    --pose_dir ./GT/button \  # Select the GT subfolder for the current task
    --output_dir /your_output_dir
```

Then run `sh/eval_franka.sh`:

```bash
python scripts/robot/eval_franka.py \
  --task Franka-pick \  # Available tasks: Franka-pick, Franka-put_on_plate, Franka-discard_trash, Franka-put_in_drawer, Franka-press_button, Franka-close_drawer, Franka-pull_and_push
  --json_root your_json_path \  # The output folder produced by `sh/parquet2action.sh`
  --enable_cameras \
  --output_root your_output_path \ 
  --device "cpu" \ # Whether to run the simulation on CPU
  --part_scores \  # Whether to enable per-stage scoring; only Franka-put_on_plate, Franka-discard_trash, Franka-put_in_drawer have stage-score design
  # --episode_index 9  # Test a single JSON index only
  # --save_dataset  \   # Whether to save execution data
```

## Roadmap

- Open-source the pure-simulation tasks + evaluation code, and release the corresponding IDM weights (target: mid-May).