#!/bin/bash
# Inference script for Densepose control condition

echo "================================"
echo "      Densepose Inference"
echo "================================"
echo "Dataset: /root/autodl-tmp/HumanFlow/datasets/MiCoGen_test"
echo "Output directory: /root/autodl-tmp/HumanFlow/generation_demo/densepose"
echo "Checkpoint: checkpoint-35000"
echo "Image size: 1024x1024"
echo ""

python inference_micogen.py \
    --control_type densepose \
    --checkpoint_path /root/autodl-tmp/HumanFlow/ckpts-train/saves_micogen_densepose/checkpoint-35000 \
    --test_data_root /root/autodl-tmp/HumanFlow/datasets/MiCoGen_test\
    --output_dir /root/autodl-tmp/HumanFlow/generation_test \
    --img_size 1024 \
    --num_steps 28 \
    --guidance 3.5 \
    --control_weight 0.9 \
    --device cuda \
    --model_type flux-dev

echo ""
echo "Densepose inference complete!"

