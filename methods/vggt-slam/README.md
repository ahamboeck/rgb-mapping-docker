# VGGT-SLAM Docker Container

This container provides VGGT-SLAM, a SLAM system leveraging VGGT features.

## 🔗 Official Repository
- **URL**: [https://github.com/MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM)

## 🚀 Usage

### 1. Build & Start
```bash
./run.sh build
./run.sh shell
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.

### 3. Example Commands
Inside the container:

**Run SLAM:**
```bash
python run_slam.py --config configs/my_config.yaml --dataset /workspace/datasets/my_sequence
```
