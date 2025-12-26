#!/bin/bash
# Script to run FastVGGT on a custom dataset

if [ -z "$1" ]; then
    echo "Usage: ./run_custom.sh <path_to_dataset_folder> [output_name]"
    echo "Dataset folder should contain an 'images' subfolder or images directly."
    exit 1
fi

DATASET_PATH="$1"
OUTPUT_NAME="${2:-custom_output}"
OUTPUT_DIR="/workspace/output/fastvggt/$OUTPUT_NAME"

# Default checkpoint path in the container (mounted from host)
CKPT_PATH="/workspace/FastVGGT/model_tracker_fixed_e20.pt"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    echo "Please ensure 'model_tracker_fixed_e20.pt' is in the FastVGGT root."
    exit 1
fi

echo "Running FastVGGT on $DATASET_PATH..."
echo "Output will be saved to $OUTPUT_DIR"

# Ensure headless mode for matplotlib/evo
export MPLBACKEND=Agg
# Try to configure evo if installed
if command -v evo_config &> /dev/null; then
    evo_config set plot_backend Agg > /dev/null 2>&1
fi

# Run the evaluation script
# Reduced input_frame to 300 and disabled vis_attn_map to prevent OOM
python3 /workspace/FastVGGT/eval/eval_custom.py \
    --data_path "$DATASET_PATH" \
    --output_path "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --input_frame 300 \
    --plot
    # --vis_attn_map

# Note: Add --enable_evaluation if you have poses and gt_ply
