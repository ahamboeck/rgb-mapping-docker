# RGB Mapping Docker Collection

This repository contains Docker setups for various state-of-the-art 3D reconstruction, mapping, and SLAM repositories. All containers are configured for **NVIDIA RTX 5090 (CUDA 12.8)** and use **Ubuntu 22.04/24.04**.

## 📂 Structure
- `datasets/`: Mount your datasets here. Mapped to `/workspace/datasets` in containers.
- `output/`: Output folder. Mapped to `/workspace/output` in containers.
- `methods/`: Contains subfolders for each project.
  - `[repo_name]/`: Contains `Dockerfile`, `docker-compose.yml`, and `README.md` for each project.

## 🚀 General Usage

For any container, the workflow is similar using Docker Compose:

1.  **Navigate to Method**:
    ```bash
    cd methods/[repo_name]
    ```
2.  **Build**:
    ```bash
    docker compose build
    ```
3.  **Start Container**:
    ```bash
    docker compose up -d
    ```
4.  **Enter Shell**:
    ```bash
    # Check the service name in docker-compose.yml, usually matches the folder name or similar
    docker compose exec [service_name] bash
    ```

---

## 📦 Containers

### 1. 3DGUT
3D Gaussian Ray Tracing.
- **Location**: `methods/3dgut/`

### 2. COLMAP
Standard COLMAP installation with CUDA support.
- **Location**: `methods/colmap/`

### 3. FastVGGT
Training-Free Acceleration of Visual Geometry Transformer.
- **Location**: `methods/fastvggt/`

### 4. HLOC
Hierarchical Localization.
- **Location**: `methods/hloc/`

### 5. Azure Kinect Recorder & Converter
Tools for recording MKV datasets with Azure Kinect and converting them for SLAM/Mapping.
- **Location**: `methods/kinect/`

### 6. Map Anything
RTAB-Map Integration with MapAnything for 3D reconstruction.
- **Location**: `methods/map_anything/`

### 7. MASt3R
MASt3R: Multi-view Attention for 3D Reconstruction.
- **Location**: `methods/mast3r/`

### 8. MASt3R-SLAM
Real-Time Dense SLAM with 3D Reconstruction Priors.
- **Location**: `methods/mast3r-slam/`

### 9. Nerfstudio
Modular framework for NeRF development.
- **Location**: `methods/nerfstudio/`

### 10. NKSR
Neural Kernel Surface Reconstruction.
- **Location**: `methods/nksr/`

### 11. RTAB-Map
Real-Time Appearance-Based Mapping with CUDA support.
- **Location**: `methods/rtabmap/`

### 12. SuGaR
Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction.
- **Location**: `methods/sugar/`

### 13. VGGT
Visual Geometry Grounded Transformer.
- **Location**: `methods/vggt/`

### 14. VGGT-Long
Extension of VGGT for long-term visual navigation and mapping.
- **Location**: `methods/vggt-long/`

### 15. VGGT-SLAM
SLAM system leveraging VGGT features.
- **Location**: `methods/vggt-slam/`

### 16. HunyuanWorld-Mirror
World model for video generation and understanding.
- **Location**: `methods/worldmirror/`
