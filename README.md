## 1. Installation

  - Environment Setup
    ```bash
    # Anaconda env setup
    conda create -n lehome python=3.11
    conda activate lehome

    # Install PyTorch
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

    # Install lehome
    # This area is temporarily inaccessible from the outside; please use the provided zip file to extract the code.
    git clone http://git.lightwheel.ai/zeyi.li/lehome.git
    cd lehome
    python -m pip install -e source/lehome

   
    # Install lerobot==0.4.3 
    pip install "lerobot==0.4.3"
    pip install "lerobot[all]==0.4.3"         # All available features
    pip install "lerobot[aloha,pusht]==0.4.3"  # Specific features (Aloha & Pusht)
    pip install "lerobot[feetech]==0.4.3"      # Feetech motor support (use this one)

    # Install IsaacSim
    cd ..
    pip install --upgrade pip
    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

    # Install IsaacLab
    git clone https://github.com/isaac-sim/IsaacLab.git
    sudo apt install cmake build-essential

    # fix isaaclab version for isaacsim4.5
    cd IsaacLab
    git checkout v2.3.0 
    ./isaaclab.sh --install
    ```

    pip install open3d

## 2. Environment Preparation

### 2.1 Activate Conda Environment

```bash
# Enter project directory
cd /yourpath/lehome
# Activate environment
conda activate lehome
```

## 3. Leader-Follower Arm Hardware Setup

### 3.1 Hardware Connection

If using **Dual SO101 Leader Arms** for teleoperation, ensure the hardware is connected correctly:

#### Hardware List

  - **2 SO101 Leader Arms** (Master arms, for control)
  - **2 SO101 Follower Arms** (Slave arms, virtual robots in simulation)
  - **USB to Serial Cable** x2
  - **Power Adapter** x2

#### Connection Steps

1.  **Physical Connection**

      - Connect the Left Leader Arm to the computer via USB
      - Connect the Right Leader Arm to the computer via USB
      - Ensure both Leader Arms are powered on

2.  **Identify Serial Devices**

    ```bash
    # View connected serial devices
    ls /dev/ttyACM*

    # Usually displays (if two devices are connected):
    # /dev/ttyACM0
    # /dev/ttyACM1
    ```

3.  **Grant Serial Permissions**

    ```bash
    # Grant read/write permissions to serial devices
    sudo chmod 666 /dev/ttyACM0
    sudo chmod 666 /dev/ttyACM1
    ```

4.  **Determine Left/Right Mapping**

      - Remember which serial port corresponds to the left arm and which to the right arm
      - Example:
          - Left Arm: `/dev/ttyACM0`
          - Right Arm: `/dev/ttyACM1`

### 3.2 Leader Arm Calibration

**First-time use** or after **changing hardware**, the SO101 Leader Arms need to be calibrated.

#### Calibration Command

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --device cuda \
    --right_arm_port /dev/ttyACM1 \
    --recalibrate \
    --enable_cameras
```

#### Calibration Steps

1.  After running the command, the program will prompt for calibration.
2.  **Left Arm Calibration**:
      - Follow prompts to move the left arm to the maximum/minimum positions of each joint.
      - Record joint limits.
3.  **Right Arm Calibration**:
      - Similarly, move the right arm to the maximum/minimum positions of each joint.
      - Record joint limits.
4.  After calibration is complete, the program will save the calibration data locally.
5.  Press **Ctrl+C** to exit the calibration program.

#### Calibration File Location

Calibration data is usually saved in:

```
~/.cache/lerobot/calibration/
├── left_arm_calibration.json
└── right_arm_calibration.json
```

**Note**:

  - Calibration only needs to be done once; no need to add the `--recalibrate` parameter for subsequent use.
  - If hardware is replaced or control feels inaccurate, you can recalibrate.

### 3.3 Verify Hardware Connection

```bash
# Do not start recording, test control only
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --device cuda \
    --right_arm_port /dev/ttyACM1 \
    --enable_cameras
```

After running:

1.  The IsaacSim window will open, displaying the scene.
2.  Move the Leader Arm and observe if the Follower robot in the simulation follows.
3.  Confirm that the left/right arm mapping is correct.
4.  Press **Ctrl+C** to exit.

-----

## 4. Teleoperation Data Recording

### 4.1 Method 1: Keyboard Control (Recommended for Practice)

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-keyboard \
    --record \
    --num_episode 10 \
    --disable_depth \
    --device cuda \
    --task_description "fold the garment on the table" \
    --enable_cameras
```

#### Keyboard Controls

**Left Arm Control** (Letter Keys):

| Key | Function | Key | Function |
|------|------|------|------|
| **T** / **G** | Joint 1 (shoulder\_pan) + / - | **Y** / **H** | Joint 2 (shoulder\_lift) + / - |
| **U** / **J** | Joint 3 (elbow\_flex) + / - | **I** / **K** | Joint 4 (wrist\_flex) + / - |
| **O** / **L** | Joint 5 (wrist\_roll) + / - | **Q** / **A** | Joint 6 (gripper) Open / Close |

**Right Arm Control** (Number Keys):

| Key | Function | Key | Function |
|------|------|------|------|
| **1** / **7** | Joint 1 + / - | **2** / **8** | Joint 2 + / - |
| **3** / **9** | Joint 3 + / - | **4** / **0** | Joint 4 + / - |
| **5** / **-** | Joint 5 + / - | **6** / **=** | Joint 6 + / - |

**Function Keys**:

| Key | Function |
|------|------|
| **B** | Start Teleoperation (Activate Control) |
| **S** | Start Recording Current Episode |
| **N** | Save Current Episode (Mark as Success) |
| **D** | Discard Current Episode (Re-record) |
| **ESC** | Abort Recording, Clear Buffer |
| **Ctrl+C** | Exit Program |

### 4.2 Method 2: SO101 Leader Arms (Recommended for Production)

#### Basic Command

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --record \
    --num_episode 50 \
    --device=cuda \
    --disable_depth \
    --task_description "fold the garment on the table" \
    --enable_cameras
```

#### Parameter Description

| Parameter | Description | Value |
|------|------|-----|
| `--teleop_device` | Control Device | `bi-so101leader` (Dual SO101 Leader Arms) |
| `--left_arm_port` | Left Arm Serial Port | `/dev/ttyACM0` |
| `--right_arm_port` | Right Arm Serial Port | `/dev/ttyACM1` |
| Other Parameters | Same as Keyboard Control | - |

#### Leader Arm Control Method

**Physical Control**: Directly move the Leader Arms; the Follower robot in the simulation will follow in real-time.

**Function Keys**:
| Key | Function |
|------|------|
| **B** | Start Teleoperation (Activate Control) |
| **S** | Start Recording Current Episode |
| **N** | Save Current Episode (Mark as Success) |
| **D** | Discard Current Episode (Re-record) |
| **ESC** | Abort Recording, Clear Buffer |
| **Ctrl+C** | Exit Program |

#### Recording Flow

1.  **Start Program**

      - Start with `keyboard` or `bi_leader` arm.

2.  **Initialization**

      - After the program starts, the IsaacSim window opens.
      - The scene loads completely, displaying the dual-arm robot and the garment.

3.  **Start Control**

      - Press **B** to activate control.
      - Use keyboard keys or move the Leader arms to control the robot.

4.  **Start Recording**

      - When ready, press **S** to start recording.
      - Terminal displays: `[S] Recording started!`

5.  **Execute Task**

      - Use the keyboard to control the robot to complete the folding task (keyboard).
      - Use both hands to move the two Leader arms (leader).

6.  **Save/Discard**

      - **Success**: Press **N** to save this episode.
      - **Failure**: Press **D** to discard and re-record.
      - Terminal displays progress: `Episode 0 recording completed, progress: 1/50`

7.  **Repeat Recording**

      - The environment resets automatically.
      - Repeat steps 4-6 until 50 episodes are recorded.

8.  **End Early**

      - Press **ESC** to exit safely (saves recorded data).
      - Or press **Ctrl+C** to force exit.

## 5. Data Replay Verification

We have prepared two demo datasets for folding clothes, with the relative path "Datasets/record/001". You can replay them directly. Of course, you can also replay your own recorded datasets.
### 5.1 Basic Replay

```bash
python scripts/tool/replay.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --dataset_root Datasets/record/001 \
    --disable_depth \
    --enable_cameras \
    --device cuda
```

**Note**:

  - Loads dataset `Datasets/record/001`.
  - Plays back all episodes one by one.

### 5.2 Replay Parameters

| Parameter | Description | Default Value |
|------|------|--------|
| `--task` | Task Name | `LeHome-BiSO101-Direct-Garment-v0` |
| `--dataset_root` | Input Dataset Path | Required |
| `--output_root` | Output Save Path | `None` (Does not save) |
| `--save_successful_only` | Save Only Successful Episodes | `False` |
| `--start_episode` | Start Episode Index | `0` |
| `--end_episode` | End Episode Index | `None` (All) |
| `--num_replays` | Number of Replays per Episode | `1` |
| `--disable_depth` | Disable Depth Maps | `False` |
| `--step_hz` | Replay Frequency | `60` |
| `--enable_cameras` | Enable Camera Rendering | Required |
| `--device cuda` | Use GPU for Rendering

## 6. Model Training

Use **LeRobot Official Training Commands** to train imitation learning policies.

### Training Commands

#### ACT Policy Training

```bash
lerobot-train \
  --dataset.repo_id=abc \
  --dataset.root=Datasets/record/001 \
  --policy.type=act \
  --steps=600 \
  --output_dir=outputs/train/act_so101_test_2 \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false 
```

#### Diffusion Policy Training

```bash
lerobot-train \
  --dataset.repo_id=abc \
  --dataset.root=Datasets/record/001 \
  --policy.type=diffusion \
  --steps=600 \
  --output_dir=outputs/train/diffusion_so101_test \
  --job_name=diffusion_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false 
```

## 7. Model Evaluation

After training is complete, use the evaluation script to test policy performance.

### 7.1 Basic Evaluation

```bash
python scripts/eval/eval_il.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --policy_path outputs/train/act_so101_test/checkpoints/last/pretrained_model \
    --num_episodes 50 \
    --max_steps 600 \
    --step_hz 60 \
    --enable_cameras \
    --device cuda
```

### 7.2 Evaluation Parameters

| Parameter | Description | Recommended Value |
|------|------|--------|
| `--task` | Task Name | `LeHome-BiSO101-Direct-Garment-v0` |
| `--policy_path` | Model Checkpoint Path | Required |
| `--num_episodes` | Number of Episodes to Evaluate | 50 |
| `--max_steps` | Maximum Steps per Episode | 600 |
| `--step_hz` | Control Frequency | 60 |
| `--save_video` | Save Video | Optional |
| `--video_dir` | Video Save Directory | `outputs/eval_videos` |
| `--save_datasets` | Save Successful Trajectories as Dataset | Optional |
| `--eval_dataset_path` | Evaluation Dataset Save Path | Custom |
| `--device` | Inference Device | `cuda` |
| `--enable_cameras` | Enable Cameras | Required |

### 8. Quick Command Reference

#### Keyboard Recording

```bash
python scripts/teleoperation/teleop_record.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --teleop_device bi-keyboard \
    --record \
    --num_episode 10 \
    --disable_depth \
    --device cuda \
    --task_description "fold the garment on the table" \
    --enable_cameras
```

#### Leader Arm Recording

```bash
    python scripts/teleoperation/teleop_record.py \
        --task LeHome-BiSO101-Direct-Garment-v0 \
        --teleop_device bi-so101leader \
        --left_arm_port /dev/ttyACM0 \
        --right_arm_port /dev/ttyACM1 \
        --record \
        --num_episode 50 \
        --device=cuda \
        --disable_depth \
        --task_description "fold the garment on the table" \
        --enable_cameras
```

#### Replay

```bash
python scripts/tool/replay.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --dataset_root Datasets/record/001 \
    --disable_depth \
    --enable_cameras \
    --device cuda
```

#### Training

```bash
lerobot-train \
  --dataset.repo_id=abc \
  --dataset.root=Datasets/record/001 \
  --policy.type=act \
  --steps=600 \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false 
```

#### Evaluation

```bash
python scripts/eval/eval_il.py \
    --task LeHome-BiSO101-Direct-Garment-v0 \
    --policy_path outputs/train/act_so101_test/checkpoints/last/pretrained_model \
    --num_episodes 50 \
    --max_steps 600 \
    --step_hz 60 \
    --enable_cameras \
    --device cuda
```