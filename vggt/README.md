# VGGT Docker Container

This container provides VGGT: Visual Geometry Grounded Transformer.

## 🔗 Official Repository
- **URL**: [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)

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

**Run Demo:**
```bash
python demo.py --input /workspace/datasets/my_video.mp4
```

**Run Chunk Processing:**
```bash
python run_chunks.py --input_dir /workspace/datasets/my_sequence
```
