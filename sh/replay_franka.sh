# python scripts/eval/replay_robotiq.py \
#   --task Franka-IDM \
#   --json_root /home/feng/lehome_1/droid/droid_output_split/folder_1 \
#   --output_root /home/jiang/robotiq \
#   --enable_cameras \
#   # --device "cpu" \


python scripts/eval/replay_franka.py \
  --task Franka-IDM \
  --json_root /home/feng/lehome_1/folder_0 \
  --output_root /home/feng/lehome_1/outputs \
  --enable_cameras \
  

  
#sudo chmod 666 /dev/ttyACM0

# python scripts/eval/replay_franka.py \
#     --task LeIsaac-SO101-Arrange_Tableware \
#     --dataset_root /home/feng/lehome_1/Datasets/replay/004 \
#     --output_root Datasets/replay \
#     --num_replays 1 \
#     --enable_cameras