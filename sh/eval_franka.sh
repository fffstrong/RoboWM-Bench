python scripts/eval/eval_franka.py \
  --task Franka-hard \
  --json_root /home/feng/lehome_1/replay/hard \
  --enable_cameras \
  --output_root /home/feng/lehome_1/Datasets \
  --device "cpu" \
  --save_dataset  \
  --part_scores \
  # --episode_index 9
  # --device "cpu" \
  # --save_dataset  \


# python scripts/eval/replay_franka.py \
#   --task Franka-IDM \
#   --json_root /home/feng/lehome_1/droid_output_1 \
#   --output_root /home/feng/lehome_1/outputs/droid_replay \
#   --enable_cameras
#   --save_dataset \

  
#sudo chmod 666 /dev/ttyACM0

# python scripts/eval/replay_franka.py \
#     --task LeIsaac-SO101-Arrange_Tableware \
#     --dataset_root /home/feng/lehome_1/Datasets/replay/004 \
#     --output_root Datasets/replay \
#     --num_replays 1 \
#     --enable_cameras