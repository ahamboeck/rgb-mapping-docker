# Nerfstudio Docker Container

This container provides Nerfstudio, a modular framework for NeRF development.

## 🔗 Official Repository
- **URL**: [https://github.com/nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
- **Documentation**: [https://docs.nerf.studio/](https://docs.nerf.studio/)

## 🚀 Usage

### 1. Build & Start
```bash
docker compose build
docker compose up -d
docker compose exec nerfstudio bash
```

### 2. Data Structure
Mount your datasets to `/workspace/datasets`.
Nerfstudio supports many formats (Colmap, Polycam, etc.).

### 3. Example Commands
Inside the container:

**Process Data (e.g., from video):**
```bash
ns-process-data video --data /workspace/datasets/my_video.mp4 --output-dir /workspace/datasets/my_processed_data
```

**Train Model (Splatfacto - Gaussian Splatting):**
```bash
ns-train splatfacto --data /workspace/datasets/my_processed_data
```

**Train Model (Nerfacto - NeRF):**
```bash
ns-train nerfacto --data /workspace/datasets/my_processed_data
```

**Viewer:**
The viewer is exposed on port 7007. Access it at `http://localhost:7007` on your host.
