#!/bin/bash

# Training script for MiCoGen ControlNet with multiple control types
# Usage: bash train_micogen.sh [control_type]
# Example: bash train_micogen.sh densepose

CONTROL_TYPE=${1:-"keypoint"}

echo "=========================================="
echo "Training MiCoGen ControlNet"
echo "Control Type: $CONTROL_TYPE"
echo "=========================================="

# Validate control type
valid_types=("canny" "densepose" "depth" "keypoint" "normal" "seg")
if [[ ! " ${valid_types[@]} " =~ " ${CONTROL_TYPE} " ]]; then
    echo "Error: Invalid control type '$CONTROL_TYPE'"
    echo "Valid types: ${valid_types[@]}"
    exit 1
fi

# Set config file
CONFIG_FILE="train_configs/micogen_${CONTROL_TYPE}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo ""

# Run training
python train_micogen.py --config $CONFIG_FILE

echo ""
echo "=========================================="
echo "Training completed for control type: $CONTROL_TYPE"
echo "=========================================="

