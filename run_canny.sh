#!/bin/bash
# Inference script for Canny control condition

echo "================================"
echo "      Canny Inference"
echo "================================"
echo "Dataset: /root/autodl-tmp/HumanFlow/datasets/MiCoGen_test"
echo "Output directory: /root/autodl-tmp/HumanFlow/generation_demo/canny"
echo "Checkpoint: checkpoint-15000"
echo "Image size: 1024x1024"
echo ""

python inference_micogen.py \
    --control_type canny \
    --checkpoint_path /root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_canny/checkpoint-15000 \
    --test_data_root /root/autodl-tmp/HumanFlow/datasets/MiCoGen_rbust \
    --output_dir /root/autodl-tmp/HumanFlow/generation_robust \
    --img_size 1024 \
    --num_steps 28 \
    --guidance 3.5 \
    --control_weight 0.9 \
    --device cuda \
    --model_type flux-dev

echo ""
echo "Canny inference complete!"

