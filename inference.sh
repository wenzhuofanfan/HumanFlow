#!/bin/bash
# =============================================================================
# MiCoGen one-click inference script with optional SDXL Refiner post-processing
# Edit the configuration below, then run: bash batch_inference_micogen.sh
# =============================================================================

# ==================== Core configuration ====================
# Control type: canny, densepose, depth, keypoint, normal, seg, or all
CONTROL_TYPE="densepose"

# Checkpoint configuration
CHECKPOINT_BASE_DIR="/root/autodl-tmp/HumanFlow"
# Uncomment the next line to use a custom checkpoint path
# CHECKPOINT_PATH="/root/autodl-tmp/HumanFlow/saves_micogen_canny/checkpoint-50000"
CHECKPOINT_PATH=""

# Data configuration
TEST_DATA_ROOT="/root/autodl-tmp/HumanFlow/datasets/MiCoGen_tset"
OUTPUT_DIR="inference_results"

# Inference range
NUM_SAMPLES=""      # number of samples to run; leave empty for all
START_IDX=0         # starting sample index

# ==================== Inference parameters ====================
NUM_STEPS=28        # number of diffusion steps
GUIDANCE=3.5        # guidance scale
CONTROL_WEIGHT=0.9  # ControlNet strength
IMG_SIZE=1024       # image size
SEED=42             # random seed

# ==================== Refiner configuration ====================
USE_REFINER=true    # whether to use SDXL Refiner (true/false)
REFINER_PATH="/root/autodl-tmp/HumanFlow/ckpts/stable-diffusion-xl-refiner-1.0"
REFINER_STRENGTH=0.3    # refiner strength (0-1, higher means stronger edits)
REFINER_STEPS=50        # refiner inference steps

# ==================== Other options ====================
SAVE_CONTROL=false  # whether to save control images (true/false)
USE_OFFLOAD=false   # whether to use CPU offload (true/false)

# =============================================================================
# Execution logic below; usually no changes are needed
# =============================================================================

echo "========================================"
echo "MiCoGen One-Click Inference"
echo "========================================"
echo "Control type: $CONTROL_TYPE"
echo "Data directory: $TEST_DATA_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "Inference steps: $NUM_STEPS | Guidance: $GUIDANCE | Control strength: $CONTROL_WEIGHT"
[ "$USE_REFINER" = true ] && echo "Refiner: enabled (strength: $REFINER_STRENGTH, steps: $REFINER_STEPS)" || echo "Refiner: disabled"
[ -n "$NUM_SAMPLES" ] && echo "Sample count: $NUM_SAMPLES (start: $START_IDX)" || echo "Sample count: all"
echo "========================================"
echo ""

# Resolve the list of control types to run
ALL_TYPES=("canny" "densepose" "depth" "keypoint" "normal" "seg")
if [ "$CONTROL_TYPE" == "all" ]; then
    TYPES_TO_RUN=("${ALL_TYPES[@]}")
    echo "Will infer all control types sequentially: ${TYPES_TO_RUN[*]}"
else
    if [[ ! " ${ALL_TYPES[@]} " =~ " ${CONTROL_TYPE} " ]]; then
        echo "Error: invalid control type '$CONTROL_TYPE'"
        echo "Valid types: ${ALL_TYPES[*]} or all"
        exit 1
    fi
    TYPES_TO_RUN=("$CONTROL_TYPE")
fi

# Inference helper function
run_inference() {
    local ctrl_type=$1
    
    # Resolve checkpoint path
    if [ -n "$CHECKPOINT_PATH" ]; then
        local checkpoint="$CHECKPOINT_PATH"
    else
        local checkpoint="${CHECKPOINT_BASE_DIR}/saves_micogen_${ctrl_type}/checkpoint-50000"
    fi
    
    # Check whether the checkpoint exists
    if [ ! -e "$checkpoint" ]; then
        echo "Warning: checkpoint does not exist: $checkpoint"
        echo "    Skipping $ctrl_type"
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Inferring control type: $ctrl_type"
    echo "Checkpoint path: $checkpoint"
    echo "=========================================="
    
    # Build the base command
    local cmd="python inference_micogen.py \
        --checkpoint_path $checkpoint \
        --control_type $ctrl_type \
        --test_data_root $TEST_DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        --num_steps $NUM_STEPS \
        --guidance $GUIDANCE \
        --control_weight $CONTROL_WEIGHT \
        --img_size $IMG_SIZE \
        --seed $SEED \
        --start_idx $START_IDX"
    
    # Add sample count when specified
    [ -n "$NUM_SAMPLES" ] && cmd="$cmd --num_samples $NUM_SAMPLES"
    
    # Add optional flags
    [ "$SAVE_CONTROL" = true ] && cmd="$cmd --save_control_image"
    [ "$USE_OFFLOAD" = true ] && cmd="$cmd --offload"
    
    # Add Refiner parameters
    if [ "$USE_REFINER" = true ]; then
        cmd="$cmd --use_refiner"
        cmd="$cmd --refiner_path $REFINER_PATH"
        cmd="$cmd --refiner_strength $REFINER_STRENGTH"
        cmd="$cmd --refiner_steps $REFINER_STEPS"
    fi
    
    echo "Running command:"
    echo "$cmd"
    echo ""
    
    # Run inference
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "Control type $ctrl_type inference complete"
        return 0
    else
        echo ""
        echo "Control type $ctrl_type inference failed (exit code: $exit_code)"
        return 1
    fi
}

# Main loop
SUCCESS=0
FAIL=0

for ctrl_type in "${TYPES_TO_RUN[@]}"; do
    run_inference "$ctrl_type"
    if [ $? -eq 0 ]; then
        ((SUCCESS++))
    else
        ((FAIL++))
    fi
done

# Print summary
echo ""
echo "========================================"
echo "Batch inference complete"
echo "========================================"
echo "Success: $SUCCESS"
echo "Failed: $FAIL"
echo "Total: ${#TYPES_TO_RUN[@]}"
echo "========================================"

exit 0
