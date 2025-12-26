#!/bin/bash
# Run NKSR Reconstruction
# Usage: ./run_reconstruction.sh <input_ply> <output_mesh> [detail_level]

INPUT_FILE="$1"
OUTPUT_FILE="$2"
DETAIL_LEVEL="${3:-1.0}"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: ./run_reconstruction.sh <input_ply> <output_mesh> [detail_level]"
    echo "Example: ./run_reconstruction.sh /workspace/datasets/my_scan.ply /workspace/output/my_mesh.ply"
    exit 1
fi

python3 /workspace/reconstruct.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --detail_level "$DETAIL_LEVEL"
