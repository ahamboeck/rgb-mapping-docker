# HLOC - Hierarchical Localization

This container provides a setup for running **hloc** (Hierarchical Localization) with CUDA 12.8 support on an RTX 5090.

## Quick Start

1.  **Build and Run the Container:**
    ```bash
    ./run.sh
    ```
    This will build the image `vision_lab/hloc:cuda12.8-5090` and drop you into a zsh shell inside the container.

2.  **Run the Python Script:**
    Inside the container, you can use the `run_hloc.py` script to process your datasets.
    ```bash
    python3 run_hloc.py --data /workspace/datasets/MyDataset --feature superpoint_max --matcher superpoint+lightglue
    ```

## Usage of `run_hloc.py`

The script automates the pipeline: Feature Extraction -> Pair Generation -> Feature Matching -> SfM Reconstruction.

### Arguments
*   `--data`: **(Required)** Path to the dataset root. Must contain an `images/` subdirectory.
*   `--output`: Path to save outputs. Defaults to `data/hloc_out`.
*   `--feature`: Feature extraction configuration (see below). Default: `superpoint_max`.
*   `--matcher`: Matcher configuration (see below). Default: `superglue`.
*   `--camera_mode`: COLMAP camera mode (`SINGLE`, `PER_IMAGE`, `PER_FOLDER`). Default: `SINGLE`.

## Available Models

### 1. Feature Extractors (`--feature`)

| Key | Model | Description |
| :--- | :--- | :--- |
| **`superpoint_max`** | SuperPoint | **Default.** Max 4096 keypoints, resized to 1600px. Great for high-quality images. |
| **`superpoint_aachen`** | SuperPoint | Max 4096 keypoints, resized to 1024px. Good for standard outdoor datasets. |
| **`superpoint_inloc`** | SuperPoint | Max 4096 keypoints, resized to 1600px. Optimized for indoor scenes. |
| **`disk`** | DISK | Learned features using reinforcement learning. Max 5000 keypoints. |
| **`aliked-n16`** | ALIKED | Multi-scale, sparse keypoints. |
| **`sift`** | SIFT | Classical SIFT. High precision, but less robust to day/night changes. |
| **`r2d2`** | R2D2 | Reliable and Repeatable Detector and Descriptor. |

### 2. Matchers (`--matcher`)

| Key | Compatible Features | Description |
| :--- | :--- | :--- |
| **`superpoint+lightglue`** | SuperPoint | **Recommended.** Fast, memory-efficient, SOTA accuracy. |
| **`superglue`** | SuperPoint | **Default.** High accuracy, robust graph neural network matcher. |
| **`superglue-fast`** | SuperPoint | Faster version of SuperGlue (fewer iterations). |
| **`disk+lightglue`** | DISK | LightGlue weights trained for DISK features. |
| **`aliked+lightglue`** | ALIKED | LightGlue weights trained for ALIKED features. |
| **`NN-superpoint`** | SuperPoint | Nearest Neighbor matching (faster, less accurate). |
| **`NN-ratio`** | Any | Standard Nearest Neighbor with Lowe's ratio test. |
| **`adalam`** | Any | Hand-crafted outlier rejection. Good for SIFT/non-learned features. |

## Recommended Combinations

1.  **Best Overall (Speed & Accuracy):**
    ```bash
    python3 run_hloc.py --data ... --feature superpoint_max --matcher superpoint+lightglue
    ```

2.  **Maximum Robustness (Legacy Gold Standard):**
    ```bash
    python3 run_hloc.py --data ... --feature superpoint_max --matcher superglue
    ```

3.  **Alternative Learned Features:**
    ```bash
    python3 run_hloc.py --data ... --feature disk --matcher disk+lightglue
    ```

4.  **Classical (High Precision, Good Lighting):**
    ```bash
    python3 run_hloc.py --data ... --feature sift --matcher adalam
    ```
