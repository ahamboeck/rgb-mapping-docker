#!/bin/bash

# Configuration
DATASET_ROOT="/workspace/datasets"
OUTPUT_ROOT="/workspace/output/hloc"
HLOC_SCRIPT="/workspace/hloc_context/run_hloc.py"

# Create output root
mkdir -p "$OUTPUT_ROOT"

echo "Starting HLOC Batch Processing..."
echo "Feature Model: superpoint_inloc"
echo "Matcher Model: superpoint+lightglue"
echo "---------------------------------------------------------"

# Find all 'images' folders to identify datasets
find "$DATASET_ROOT" -type d -name "images" | while read -r img_dir; do
    
    # Get the parent directory (the actual dataset folder)
    parent_dir=$(dirname "$img_dir")
    
    # Calculate the relative path from the dataset root
    rel_path=${parent_dir#$DATASET_ROOT/}
    
    # --- FILTERS ---
    # 1. Skip kw_sim_biomasse_set_3 (Requested)
    if [[ "$rel_path" == *"kw_sim_biomasse_set_3"* ]]; then
        echo ">> [SKIP] $rel_path: Specifically excluded."
        continue
    fi

    # 2. Skip Video folders (Requested)
    if [[ "$rel_path" == *"video"* ]]; then
        echo ">> [SKIP] $rel_path: Video extraction folder."
        continue
    fi

    # Define unique output folder
    current_output="$OUTPUT_ROOT/$rel_path"
    mkdir -p "$current_output"

    echo ">>> PROCESSING: $rel_path"
    
    # Execute the HLOC script
    # Note: We use the specific models you requested here
    python3 "$HLOC_SCRIPT" \
        --data "$parent_dir" \
        --output "$current_output" \
        --feature "superpoint_inloc" \
        --matcher "superpoint+lightglue" \
        --camera_mode "SINGLE"

    # Capture exit status to see if it crashed
    if [ $? -eq 0 ]; then
        echo ">>> SUCCESS: $rel_path"
    else
        echo ">>> ERROR: $rel_path failed. Moving to next dataset."
    fi
    echo "---------------------------------------------------------"

done

echo "All tasks complete."