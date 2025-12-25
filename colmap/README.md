# COLMAP Docker Container

This container provides a standard COLMAP installation with CUDA support, optimized for NVIDIA RTX 5090.

## 🔗 Official Repository
- **URL**: [https://github.com/colmap/colmap](https://github.com/colmap/colmap)
- **Documentation**: [https://colmap.github.io/](https://colmap.github.io/)

## 🚀 Usage

### 1. Build & Start
```bash
./run.sh build
./run.sh shell
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.
```
/workspace/datasets/
    ├── my_dataset/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
```

### 3. Example Commands
Inside the container:

**Automatic Reconstruction:**
```bash
colmap automatic_reconstructor \
    --workspace_path /workspace/output/my_project \
    --image_path /workspace/datasets/my_dataset/images
```

**Feature Extraction:**
```bash
colmap feature_extractor \
    --database_path /workspace/output/database.db \
    --image_path /workspace/datasets/my_dataset/images
```

**Feature Matching:**
```bash
colmap exhaustive_matcher \
    --database_path /workspace/output/database.db
```

**Mapper:**
```bash
mkdir -p /workspace/output/sparse
colmap mapper \
    --database_path /workspace/output/database.db \
    --image_path /workspace/datasets/my_dataset/images \
    --output_path /workspace/output/sparse
```
