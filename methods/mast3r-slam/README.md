# MASt3R-SLAM Docker Container

This container provides MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors.

## 🔗 Official Repository
- **URL**: [https://github.com/BenUCL/MASt3R-SLAM](https://github.com/BenUCL/MASt3R-SLAM) (Fork used for Blackwell support)
- **Original**: [https://github.com/naver/mast3r](https://github.com/naver/mast3r)

## 🚀 Usage

### 1. Build & Start
```bash
./run.sh build
./run.sh install  # First time only, to build CUDA extensions
./run.sh shell
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.
EuRoC format is commonly used.
```
/workspace/datasets/EuRoC/
    ├── MH_01_easy/
    │   ├── mav0/
    │   │   ├── cam0/
    │   │   ├── cam1/
    │   │   └── ...
```

### 3. Example Commands
Inside the container:

**Run SLAM on EuRoC Dataset:**
```bash
python main.py --config config/euroc.yaml --dataset_path /workspace/datasets/EuRoC/MH_01_easy
```

**Run with Realsense (Live):**
```bash
python main.py --config config/realsense.yaml
```
