# python /home/feng/lehome_1/tools/convert_pkl_to_lerobot_v30.py \
#   --pkl /home/jiang/nips_real_data/pick/folding_10_demos_2026-04-08_16-04-39.pkl \
#   --out /home/jiang/nips_real_data/lerobot/005 \
#   --fps 30 \
#   --task-description "pick the object on the table" \
#   --video \
#   --video-key top_rgb

python /home/feng/lehome_1/tools/pkl_to_lerobot.py \
  --pkl /home/jiang/nips_real_data/pick/folding_10_demos_2026-04-08_15-35-48.pkl \
  --out /home/jiang/nips_real_data/lerobot/001 \
  --fps 30 \
  --task-description "pick up the object on the table" \
  --video \
  --video-key top_rgb

# 多任务时改用 JSON，并加上:
#   --task-spec /path/to/tasks.json
#
# tasks.json 示例:
# {"tasks": ["task a", "task b"], "episode_task_indices": [0, 0, 1, ...]}
