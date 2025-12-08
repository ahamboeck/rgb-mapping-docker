# RGB Mapping Docker Collection

This repository contains Docker setups for various state-of-the-art 3D reconstruction, mapping, and SLAM repositories. All containers are configured for **NVIDIA RTX 5090 (CUDA 12.8)** and use **Ubuntu 22.04/24.04**.

## 📂 Structure
- `datasets/`: Mount your datasets here. Mapped to `/workspace/datasets` in containers.
- `output/`: Output folder. Mapped to `/workspace/output` in containers.
- `[repo_name]/`: Contains `Dockerfile`, `docker-compose.yml`, and `run.sh` for each project.

## 🚀 General Usage

For any container, the workflow is similar:

1.  **Build & Start**:
    ```bash
    cd [repo_name]
    ./run.sh build
    ```
2.  **Enter Shell**:
    ```bash
    ./run.sh shell
    ```
3.  **Run Commands**:
    ```bash
    ./run.sh [command]
    ```

---

## 📦 Containers

### 1. MASt3R-SLAM
Real-Time Dense SLAM with 3D Reconstruction Priors.
- **Location**: `mast3r-slam/`
- **Setup**:
    ```bash
    cd mast3r-slam
    ./run.sh build
    # First time only: Build CUDA extensions
    ./run.sh install
    ```
- **Usage**:
    ```bash
    ./run.sh python main.py --config config/euroc.yaml ...
    ```

### 2. FastVGGT
Training-Free Acceleration of Visual Geometry Transformer.
- **Location**: `fastvggt/`
- **Setup**:
    ```bash
    cd fastvggt
    ./run.sh build
    ```
- **Custom Dataset Script**:
    ```bash
    # Run on a custom dataset folder
    ./run.sh /workspace/run_custom.sh /workspace/datasets/my_dataset my_output_name
    ```

### 3. VGGT
Visual Geometry Grounded Transformer.
- **Location**: `vggt/`
- **Setup**:
    ```bash
    cd vggt
    ./run.sh build
    ```
- **Chunk Processing Script**:
    ```bash
    # Process large datasets in chunks
    ./run.sh /workspace/datasets/my_dataset my_output_name --overlap 20
    ```

### 4. MASt3R
MASt3R: Multi-view Attention for 3D Reconstruction.
- **Location**: `mast3r/`
- **Setup**:
    ```bash
    cd mast3r
    ./run.sh build
    ```
- **Batch Processing Script**:
    ```bash
    # Run sliding window inference
    ./run_batch.sh /workspace/datasets/my_dataset my_output_name --window_size 10
    ```

### 5. SuGaR
Surface-Aligned Gaussian Splatting.
- **Location**: `sugar/`
- **Setup**:
    ```bash
    cd sugar
    ./run.sh build
    ```
- **Usage**:
    ```bash
    ./run.sh python train.py -s /workspace/datasets/my_scene ...
    ```

### 6. MapAnything
- **Location**: `map_anything/`
- **Setup**:
    ```bash
    cd map_anything
    ./run.sh build
    ```

### 7. HunyuanWorld-Mirror
- **Location**: `worldmirror/`
- **Setup**:
    ```bash
    cd worldmirror
    ./run.sh build
    ```

### 8. Colmap
Standard Colmap with CUDA support.
- **Location**: `colmap/`
- **Setup**:
    ```bash
    cd colmap
    ./run.sh build
    ```

## 🔧 Troubleshooting
- **Permissions**: If scripts are not executable, run `chmod +x run.sh`.
- **OOM Errors**: Reduce batch sizes or window sizes in the provided scripts.
- **Display**: X11 forwarding is configured. Ensure you have an X server running on your host if you want to see GUIs.
