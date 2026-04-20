python utils/auto_prompt.py \
    data/raw/hand_dataset/box_bi/2/ \
    "Pick up the big box by grasping both the left and the right edge of it using to hands." \
    --model "qwen3-vl-235b-a22b-thinking" \
    --num_output_frames 93 \
    --chunk_size 93 \
    --chunk_overlap 1