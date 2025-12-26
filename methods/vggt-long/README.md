# VGGT-Long Docker Container

This container provides VGGT-Long, an extension of VGGT for long-term visual navigation and mapping.

## 🔗 Official Repository
- **URL**: [https://github.com/DengKaiCQ/VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

## 🚀 Usage

### 1. Build & Start
```bash
docker compose build
docker compose up -d
docker compose exec vggt-long bash
```

### 2. Download Weights
Inside the container, run the download script to fetch necessary models (saved to host cache):
```bash
./scripts/download_weights.sh
```

### 3. Data Structure
Mount your datasets to `/workspace/datasets`.

### 4. Example Commands
Inside the container:

**Run VGGT-Long:**
```bash
python vggt_long.py --image_dir /workspace/datasets/my_sequence
```
