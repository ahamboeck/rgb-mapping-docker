#!/bin/bash
set -e

# Usage: run_colmap <PATH_TO_IMAGES_FOLDER>
INPUT_IMAGE_PATH="$1"

# 1. Check Input
if [ -z "$INPUT_IMAGE_PATH" ]; then
    echo "Error: Please provide the path to your images folder."
    echo "Usage: run_colmap /workspace/datasets/Doggo/images"
    exit 1
fi

# 2. Derive Parent Directory (Where outputs will go)
# Removes trailing slash if present, then gets directory name
CLEAN_PATH=${INPUT_IMAGE_PATH%/}
PROJECT_PATH=$(dirname "$CLEAN_PATH")

echo "============================================================"
echo " Project Path:   ${PROJECT_PATH}"
echo " Image Input:    ${INPUT_IMAGE_PATH}"
echo "============================================================"

# Define Output Paths
DB_PATH="${PROJECT_PATH}/database.db"
SPARSE_PATH="${PROJECT_PATH}/sparse"
DENSE_PATH="${PROJECT_PATH}/dense"

# RTX 5090 TDR Safety Cap
MAX_IMAGE_SIZE=3200

# 3. CLEANUP (Fresh Start)
echo "[1/6] Cleaning previous run data..."
rm -rf "${DB_PATH}" "${SPARSE_PATH}" "${DENSE_PATH}"
mkdir -p "${SPARSE_PATH}" "${DENSE_PATH}"

# 4. FEATURE EXTRACTION & MATCHING
echo "[2/6] Extracting & Matching Features..."
colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${INPUT_IMAGE_PATH}" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_image_size ${MAX_IMAGE_SIZE}

colmap exhaustive_matcher \
    --database_path "${DB_PATH}" \
    --SiftMatching.use_gpu 1

# 5. SPARSE RECONSTRUCTION
echo "[3/6] Running Sparse Reconstruction..."
colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${INPUT_IMAGE_PATH}" \
    --output_path "${SPARSE_PATH}"

# 6. UNDISTORTION
echo "[4/6] Undistorting Images..."
colmap image_undistorter \
    --image_path "${INPUT_IMAGE_PATH}" \
    --input_path "${SPARSE_PATH}/0" \
    --output_path "${DENSE_PATH}" \
    --output_type COLMAP \
    --max_image_size ${MAX_IMAGE_SIZE}

# 7. DENSE RECONSTRUCTION
echo "[5/6] Patch Match Stereo..."
colmap patch_match_stereo \
    --workspace_path "${DENSE_PATH}" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# 8. FUSION & MESHING
echo "[6/6] Fusing & Meshing..."
colmap stereo_fusion \
    --workspace_path "${DENSE_PATH}" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "${DENSE_PATH}/fused.ply"

colmap poisson_mesher \
    --input_path "${DENSE_PATH}/fused.ply" \
    --output_path "${DENSE_PATH}/meshed.ply" \
    --PoissonMeshing.trim 2

echo "============================================================"
echo " SUCCESS!"
echo " Mesh saved to: ${DENSE_PATH}/meshed.ply"
echo "============================================================"