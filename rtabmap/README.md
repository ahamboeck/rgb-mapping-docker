# RTAB-Map Docker Container

This container provides RTAB-Map (Real-Time Appearance-Based Mapping) with CUDA support.

## 🔗 Official Repository
- **URL**: [https://github.com/introlab/rtabmap](https://github.com/introlab/rtabmap)
- **Docker Image**: [introlab3it/rtabmap](https://hub.docker.com/r/introlab3it/rtabmap)

## 🚀 Usage

### 1. Start
```bash
docker compose up -d
```

### 2. GUI Access
This container is configured for X11 forwarding. Ensure you have an X server running on your host (e.g., VcXsrv on Windows, or native X11 on Linux) and `xhost +` allowed if necessary.

### 3. Data Structure
- Datasets are mounted to `/workspace/datasets`.
- Output is mounted to `/workspace/output`.
- RTAB-Map database is persisted in `~/Documents/RTAB-Map`.

### 4. Example Commands
Inside the container:

**Start RTAB-Map GUI:**
```bash
rtabmap
```

**Process Dataset (CLI):**
```bash
rtabmap-console --params /workspace/datasets/my_params.ini /workspace/datasets/my_sequence
```
