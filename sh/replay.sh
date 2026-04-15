python scripts/eval/replay.py \
  --task LeIsaac-SO101-Arrange_Tableware \
  --dataset_root /home/feng/lehome_1/datasets/idm_last \
  --output_root /home/feng/lehome_1/outputs/replay \
  --num_replays 1 \
  --enable_cameras
  #   --save_successful_only \


  
#sudo chmod 666 /dev/ttyACM0

# python scripts/eval/replay_franka.py \
#     --task LeIsaac-SO101-Arrange_Tableware \
#     --dataset_root /home/feng/lehome_1/Datasets/replay/004 \
#     --output_root Datasets/replay \
#     --num_replays 1 \
#     --enable_cameras