# NKSR Docker Container

This container provides NKSR (Neural Kernel Surface Reconstruction).

## 🔗 Official Repository
- **URL**: [https://github.com/nv-tlabs/nksr](https://github.com/nv-tlabs/nksr)

## 🚀 Usage

### 1. Build & Start
```bash
./run.sh build
./run.sh shell
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.
Input is typically a point cloud (PLY/XYZ) with normals.

### 3. Example Commands
Inside the container:

**Reconstruct from Point Cloud:**
```bash
python reconstruct.py --input /workspace/datasets/my_pointcloud.ply --output /workspace/output/mesh.ply
```

**Run Reconstruction Script:**
```bash
./run_reconstruction.sh /workspace/datasets/my_pointcloud.ply
```
