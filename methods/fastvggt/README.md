# FastVGGT Docker Container

This container provides FastVGGT: Training-Free Acceleration of Visual Geometry Transformer.

## 🔗 Official Repository
- **URL**: [https://github.com/mystorm16/FastVGGT](https://github.com/mystorm16/FastVGGT)

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
    │   │   ├── 00000.jpg
    │   │   ├── 00001.jpg
    │   │   └── ...
```

### 3. Example Commands
Inside the container:

**Run Custom Dataset:**
```bash
# Using the provided helper script
./run_custom.sh /workspace/datasets/my_dataset
```

**Manual Execution:**
```bash
python demo.py --image_path /workspace/datasets/my_dataset/images
```
