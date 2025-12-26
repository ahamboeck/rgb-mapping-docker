#!/bin/bash
# Script to run MASt3R Sliding Window on a custom dataset

if [ -z "$1" ]; then
    echo "Usage: ./run_custom.sh <path_to_dataset_folder> [output_name] [extra_args...]"
    echo "Dataset folder should contain images."
    exit 1
fi

DATASET_PATH="$1"
OUTPUT_NAME="${2:-mast3r_output}"
OUTPUT_FILE="/workspace/output/mast3r/${OUTPUT_NAME}.ply"

# Shift arguments to pass extra args to python script
shift 2

echo "Running MASt3R on $DATASET_PATH..."
echo "Output will be saved to $OUTPUT_FILE"

# Run the script
python3 /workspace/run_mast3r_sliding.py \
    --image_path "$DATASET_PATH" \
    --output_file "$OUTPUT_FILE" \
    "$@"
