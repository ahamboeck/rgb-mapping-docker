# 3DGUT (3D Gaussian Ray Tracing)

Docker container for 3DGUT.

## Usage

1. Build and start the container:
   ```bash
   docker compose up -d --build
   ```

2. Run training:
   ```bash
   docker exec -it 3dgut_5090 /workspace/run.sh \
       --config-name apps/colmap_3dgut_mcmc.yaml \
       path=/workspace/datasets/YourDataset \
       out_dir=/workspace/output/3dgut \
       experiment_name=your_experiment \
       export_usdz.enabled=true \
       export_usdz.apply_normalizing_transform=true
   ```

## Notes
- The container uses Miniconda to manage the environment.
- The `3dgrut` environment is activated automatically in `run.sh`.
