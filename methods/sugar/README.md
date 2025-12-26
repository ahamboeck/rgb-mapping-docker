# SuGaR Docker Container

This container provides SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering.

## 🔗 Official Repository
- **URL**: [https://github.com/Anttwo/SuGaR](https://github.com/Anttwo/SuGaR)

## 🚀 Usage

### 1. Build & Start
```bash
./run_sugar.sh build
./run_sugar.sh shell
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.
SuGaR typically expects standard Gaussian Splatting / Colmap datasets.
```
/workspace/datasets/my_scene/
    ├── images/
    ├── sparse/
    │   └── 0/
    └── ...
```

### 3. Example Commands
Inside the container:

**Train SuGaR:**
```bash
python train.py -s /workspace/datasets/my_scene -m /workspace/output/my_scene_sugar
```

**Extract Mesh:**
```bash
python extract_mesh.py -m /workspace/output/my_scene_sugar
```
