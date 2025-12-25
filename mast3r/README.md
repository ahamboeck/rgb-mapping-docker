# MASt3R Docker Container

This container provides MASt3R: Multi-view Attention for 3D Reconstruction.

## 🔗 Official Repository
- **URL**: [https://github.com/naver/mast3r](https://github.com/naver/mast3r)

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
    ├── my_scene/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
```

### 3. Example Commands
Inside the container:

**Run Sliding Window Reconstruction:**
```bash
python run_mast3r_sliding.py --input_dir /workspace/datasets/my_scene/images --output_dir /workspace/output/my_scene_mast3r
```

**Run Demo:**
```bash
python demo.py
```
