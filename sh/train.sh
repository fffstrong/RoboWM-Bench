# lerobot-train \
#   --dataset.repo_id=/home/feng/lehome_1/Datasets/replay/004 \
#   --policy.type=diffusion \
#   --output_dir=model/dp_n \
#   --job_name=diffusion \
#   --policy.device=cuda \
#   --wandb.enable=false \
#   --policy.repo_id=004

lerobot-train \
  --dataset.repo_id=/home/feng/lehome_1/Datasets/replay/004 \
  --policy.type=pi05 \
  --output_dir=./outputs/pi05_training \
  --job_name=pi05_training \
  --policy.repo_id=004 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=false \
  --policy.dtype=bfloat16 \
  --steps=3000 \
  --policy.scheduler_decay_steps=3000 \
  --policy.device=cuda \
  --policy.train_expert_only "true"\
  --batch_size=4