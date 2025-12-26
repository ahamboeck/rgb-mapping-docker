# HunyuanWorld-Mirror Docker Container

This container provides HunyuanWorld-Mirror, a world model for video generation and understanding.

## 🔗 Official Repository
- **URL**: [https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror)

## 🚀 Usage

### 1. Build & Start
```bash
docker compose build
docker compose up -d
docker compose exec worldmirror bash
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.

### 3. Example Commands
Inside the container:

**Run Gradio App:**
```bash
python app.py
```
Access the app at `http://localhost:7860` (if port mapped).

**Inference:**
```bash
python infer.py --input /workspace/datasets/my_image.jpg
```
