# 3DGRUT (3D Gaussian Ray Tracing & Unscented Transform)

This Docker container provides a high-performance environment for training NVIDIA's **3DGRUT** framework. It is specifically optimized for the **Blackwell (RTX 50-series)** architecture, supporting physically accurate reflections, secondary rays, and distorted camera models for robotics simulation.

## 🚀 Usage

### 1. Build and Start the Container
Ensure you have the NVIDIA Container Toolkit installed and your host is running driver **580.xx** or newer for Blackwell support.

```bash
docker compose up -d --build
```

### 2. Standard Training Commands
The `run.sh` script automatically activates the `3dgrut` conda environment inside the container.

### 3. Live Training GUI (`with_gui=True`)

3DGRUT can open an interactive Polyscope window while training. Enable it by adding `with_gui=True` to your training command (requires X11, same as the Playground section below).

```bash
docker exec -it 3dgut_5090 /workspace/run.sh \
    --config-name apps/colmap_3dgut.yaml \
    path=/workspace/datasets/YourDataset \
    out_dir=/workspace/output/3dgut \
    experiment_name=your_experiment \
    with_gui=True
```

### 4. Changing Training Parameters (Hydra overrides)

3DGRUT uses Hydra configs. You can override any config value by appending `key=value` arguments to the command.

For example, to run only 15k iterations:

```bash
docker exec -it 3dgut_5090 /workspace/run.sh \
    --config-name apps/colmap_3dgut.yaml \
    path=/workspace/datasets/YourDataset \
    out_dir=/workspace/output/3dgut \
    experiment_name=your_experiment \
    n_iterations=15000
```

Optional: align checkpoint saving with shorter runs:

```bash
docker exec -it 3dgut_5090 /workspace/run.sh \
    --config-name apps/colmap_3dgut.yaml \
    path=/workspace/datasets/YourDataset \
    out_dir=/workspace/output/3dgut \
    experiment_name=your_experiment \
    n_iterations=15000 \
    checkpoint.iterations='[15000]'
```

Notes:

- Use `+some.new_key=value` to add a new config key.
- Use `++some.existing_key=value` to force-override an existing key (rarely needed).

#### Config Mode

| App Config File | Use Case |
| :--- | :--- |
| `apps/colmap_3dgut.yaml` | **Rasterization**. High-speed training for general navigation. |
| `apps/colmap_3dgut_mcmc.yaml` | **MCMC (Clean)**. Cleaner geometry with fewer "floaters" in background. |
| `apps/colmap_3dgrt.yaml` | **Ray Tracing**. Physically accurate reflections, shadows, and secondary rays. |

#### 🛠️ Advanced Parameter Overrides (Hydra)
This framework uses Hydra for configuration management. Use `+` to add new keys and `++` to force-override existing ones.

#### 🧬 The "Hybrid Mix" (Recommended for Isaac Sim)
This configuration uses fast 3DGUT rasterization for the primary view and 3DGRT ray tracing for secondary reflections.

```bash
docker exec -it 3dgut_5090 /workspace/run.sh \
    --config-name apps/colmap_3dgrt.yaml \
    path=/workspace/datasets/YourDataset \
    out_dir=/workspace/output/3dgut \
    experiment_name=lab_digital_twin \
    +model.use_reflections=true \
    +model.max_reflection_depth=1 \
    optimizer.type=selective_adam \
    export_usdz.enabled=true \
    export_usdz.apply_normalizing_transform=true
```

#### ⚡ Blackwell (RTX 5090) Optimizations
*   `optimizer.type=selective_adam`: **Highly Recommended**. Reduces VRAM traffic and increases throughput on SM 10.0/12.0.
*   `dataset.downsample_factor=2`: Reduces resolution (e.g., 4k to 2k) to save VRAM during secondary ray tracing.
*   `training.max_steps=30000`: Standard for research quality; use 7000 for rapid prototyping.

### 🤖 Isaac Sim & Omniverse Export
To generate a `.usdz` file compatible with the NuRec extension:

```bash
    export_usdz.enabled=true \
    export_usdz.apply_normalizing_transform=true
```

> **Note:** The normalizing transform is vital to center the scene near origin (0,0,0) and scale it to meters.

### 📂 Parameter Matrix

| Category | Parameter | Default / Options |
| :--- | :--- | :--- |
| Reflections | `+model.use_reflections` | `true`, `false` |
| Shadows | `+model.ray_trace_shadows` | `true`, `false` |
| Depth | `+model.max_reflection_depth` | `1` (Mirrors), `2` (Glass) |
| Optimization | `optimizer.type` | `adam`, `selective_adam` |
| Data | `dataset.downsample_factor` | `1` (Native), `2` (Half), `4` (Quarter) |

## 🖥️ Launch the GUI (Playground)

This repo includes the upstream 3DGRUT interactive Playground (`playground.py`). It opens an on-screen OpenGL window, so you need X11 forwarding from the container.

### 1. Allow the container to use your X server (Linux)

On the host:

```bash
xhost +si:localuser:root
```

Your `docker-compose.yml` already passes `DISPLAY` and mounts `/tmp/.X11-unix`, and enables `NVIDIA_DRIVER_CAPABILITIES=...graphics,display`.

### 2. Run the Playground against a trained checkpoint

After you have a checkpoint (for example `/workspace/output/3dgut/<experiment>/ckpt_last.pt`), launch the GUI like this:

```bash
docker exec -it 3dgut_5090 bash -lc \
    'source /root/miniconda3/etc/profile.d/conda.sh && \
     conda activate 3dgrut && \
     cd /workspace/3dgrut && \
     python playground.py --gs_object /workspace/output/3dgut/<experiment>/ckpt_last.pt'
```

### Optional: Viser-based GUI

There is also a simpler browser-based GUI:

```bash
docker exec -it 3dgut_5090 bash -lc \
    'source /root/miniconda3/etc/profile.d/conda.sh && \
     conda activate 3dgrut && \
     cd /workspace/3dgrut && \
     python threedgrut_playground/viser_gui.py --gs_object /workspace/output/3dgut/<experiment>/ckpt_last.pt'
```

## 🔧 Utilities

### Manual PLY to USDZ Conversion
If you have a `.ply` file from Nerfstudio (Splatfacto) or standard 3DGS, convert it manually:

```bash
python -m threedgrut.export.scripts.ply_to_usd \
    /path/to/input.ply \
    --output_file /path/to/output.usdz
```

### COLMAP Image Rectification
Prepare raw images for training by converting them to the PINHOLE model:

```bash
colmap image_undistorter \
    --image_path /data/images \
    --input_path /data/sparse/0 \
    --output_path /data/undistorted \
    --output_type COLMAP
```

## 📊 Performance on RTX 5090
*   **Primary Iterations:** ~50-60 it/s.
*   **VRAM Usage (Standard):** ~12GB.
*   **VRAM Usage (Ray Traced @ 2k):** ~22GB.
