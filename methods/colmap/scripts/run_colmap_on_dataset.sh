#!/bin/bash

# Check if both path and camera model are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./run_colmap_on_dataset.sh /path/to/dataset CAMERA_MODEL"
    echo "Common models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV"
    exit 1
fi

DATASET_PATH=$1
# Convert camera model to uppercase to match COLMAP requirements
CAMERA_MODEL=$(echo "$2" | tr '[:lower:]' '[:upper:]')

IMAGES_PATH="$DATASET_PATH/images"
DATABASE_PATH="$DATASET_PATH/database.db"
SPARSE_PATH="$DATASET_PATH/sparse"
UNDISTORTED_PATH="$DATASET_PATH/undistorted"

# Ensure images folder exists
if [ ! -d "$IMAGES_PATH" ]; then
    echo "Error: $IMAGES_PATH does not exist."
    exit 1
fi

# Create output directories
mkdir -p "$SPARSE_PATH"
mkdir -p "$UNDISTORTED_PATH"

echo "--- [1/4] Feature Extraction (Full Res) ---"
# Setting max_image_size to 16000 or removing it ensures full resolution processing
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_PATH" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA_MODEL" \
    --SiftExtraction.max_image_size 16000 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

echo "--- [2/4] Exhaustive Matching ---"
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

echo "--- [3/4] Mapping ---"
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGES_PATH" \
    --output_path "$SPARSE_PATH"

echo "--- [4/4] Selecting Largest Model & Undistorting ---"

BEST_MODEL=""
MAX_IMAGES=-1

for d in "$SPARSE_PATH"/* ; do
    if [ -d "$d" ] && [ -f "$d/cameras.bin" ] || [ -f "$d/cameras.txt" ]; then
        # Improved extraction: look for the line, remove non-digits, and ensure it's a number
        IMG_COUNT=$(colmap model_analyzer --path "$d" 2>&1 | grep "Registered images" | sed 's/[^0-9]//g')
        
        # If IMG_COUNT is empty, set it to 0
        IMG_COUNT=${IMG_COUNT:-0}
        
        echo "Found model in $d with $IMG_COUNT registered images."
        
        if [ "$IMG_COUNT" -gt "$MAX_IMAGES" ]; then
            MAX_IMAGES=$IMG_COUNT
            BEST_MODEL=$d
        fi
    fi
done

# Final check and fallback
if [ -z "$BEST_MODEL" ] && [ -d "$SPARSE_PATH/0" ]; then
    echo "Warning: Could not parse image counts, falling back to $SPARSE_PATH/0"
    BEST_MODEL="$SPARSE_PATH/0"
fi

if [ -n "$BEST_MODEL" ]; then
    echo "Using model: $BEST_MODEL"
    colmap image_undistorter \
        --image_path "$IMAGES_PATH" \
        --input_path "$BEST_MODEL" \
        --output_path "$UNDISTORTED_PATH" \
        --output_type COLMAP \
        --max_image_size 16000
else
    echo "Error: No valid sparse model subdirectories found in $SPARSE_PATH"
    exit 1
fi