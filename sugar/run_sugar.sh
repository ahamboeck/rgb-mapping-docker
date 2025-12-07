#!/bin/bash
set -e

# Usage: run_sugar <PATH_TO_COLMAP_DATASET> <REGULARIZATION_TYPE>
# Example: run_sugar /workspace/datasets/Doggo dn_consistency

SCENE_PATH="$1"
REGULARIZATION="${2:-dn_consistency}" # Default to dn_consistency (best quality)

if [ -z "$SCENE_PATH" ]; then
    echo "Error: Please provide the path to your COLMAP dataset."
    echo "Usage: run_sugar /workspace/datasets/Doggo [dn_consistency|density|sdf]"
    exit 1
fi

echo "============================================================"
echo " SuGaR Pipeline (RTX 5090 Optimized)"
echo " Scene: ${SCENE_PATH}"
echo " Regularization: ${REGULARIZATION}"
echo "============================================================"

# Navigate to repo root
cd /workspace/SuGaR

# Run the full pipeline
# - high_poly: 1M vertices (set to False for 200k)
# - export_obj: Exports the textured mesh for Blender
python3 train_full_pipeline.py \
    -s "${SCENE_PATH}" \
    -r "${REGULARIZATION}" \
    --high_poly True \
    --export_obj True \
    --refinement_time "short"

# Note: "short" refinement is usually sufficient and takes a few minutes.
# Use "long" for maximum quality (takes significantly longer).

echo "============================================================"
echo " JOB FINISHED"
echo " Check /workspace/SuGaR/output/ for .obj and .ply files"
echo "============================================================"