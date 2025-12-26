# Map Anything - RTAB-Map Integration

This folder contains tools to use RTAB-Map output with MapAnything for 3D reconstruction.

## Overview

[MapAnything](https://github.com/facebookresearch/map-anything) is a universal feed-forward metric 3D reconstruction model that can take various inputs (images, calibration, poses, depth) and produce dense 3D reconstructions.

This integration allows you to:
1. Take RTAB-Map SLAM output (RGB images, depth images, camera poses, calibration)
2. Convert it to MapAnything format
3. Run MapAnything inference for enhanced 3D reconstruction
4. Export results as GLB files

## Prerequisites

- Docker with NVIDIA GPU support
- RTAB-Map output with the following structure:
  ```
  rtabmap_output/
  ├── rgb/
  │   ├── 1.jpg
  │   ├── 4.jpg    # Non-sequential numbering is OK
  │   ├── 5.jpg
  │   └── ...
  ├── depth/
  │   ├── 1.png    # Must match RGB numbers
  │   ├── 4.png
  │   ├── 5.png
  │   └── ...
  ├── poses.txt    # Format: timestamp x y z qx qy qz qw
  └── calibration.yaml
  ```

## Quick Start

### 1. Start the Docker container

```bash
cd rgb-mapping-docker
docker compose -f map_anything/docker-compose.yml up -d
docker exec -it map_anything_5090 bash
```

### 2. Run the conversion and inference

Inside the container:

```bash
# Process all frames
python /workspace/map_anything/rtabmap_to_mapanything.py \
    --rtabmap_path /workspace/datasets/your_rtabmap_data

# Use every 5th frame (recommended for large datasets)
python /workspace/map_anything/rtabmap_to_mapanything.py \
    --rtabmap_path /workspace/datasets/your_rtabmap_data \
    --stride 5

# Limit to first 100 frames
python /workspace/map_anything/rtabmap_to_mapanything.py \
    --rtabmap_path /workspace/datasets/your_rtabmap_data \
    --stride 5 \
    --max_frames 100
```

Or use the convenience script:

```bash
/workspace/map_anything/run_rtabmap.sh /workspace/datasets/your_rtabmap_data 5 100
```

## Script Options

```
--rtabmap_path PATH    Path to RTAB-Map output folder (required)
--output_dir PATH      Output directory (default: rtabmap_path/mapanything_output)
--stride N             Use every N-th frame (default: 1)
--max_frames N         Maximum frames to process after stride (-1 for all)
--no_depth             Don't use depth data (let MapAnything predict depth)
--memory_efficient     Use memory-efficient inference (slower but handles more views)
--export_only          Only load data without running inference
--no_glb               Don't save output as GLB file
--verbose              Enable verbose output
```

## Data Format Details

### poses.txt
```
#timestamp x y z qx qy qz qw
1765816878.700786 0.000001 0.000005 -0.000003 0.569283 -0.554854 0.436877 -0.420942
...
```
- Poses are sequential (1st line = 1st image when sorted numerically)
- Poses are in camera frame (cam2world transformation)
- Quaternion format: qx, qy, qz, qw

### calibration.yaml
```yaml
image_width: 1280
image_height: 720
projection_matrix:
   rows: 3
   cols: 4
   data: [ fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0 ]
local_transform:
   rows: 3
   cols: 4
   data: [ ... ]  # Camera to body frame transformation (not used if poses are in camera frame)
```

### Depth Images
- Format: 16-bit PNG
- Units: Millimeters
- Converted to meters during processing

## Output

The script produces:
- `mapanything_output/rtabmap_mapanything_output.glb` - 3D mesh/pointcloud

## Notes

1. **Image Numbering**: RTAB-Map may export images with non-sequential numbers (1, 4, 5, 7...). The script handles this by sorting images numerically and matching them to sequential poses.

2. **Stride Selection**: For large datasets (1000+ frames), using a stride of 5-10 is recommended to balance reconstruction quality with processing time/memory.

3. **Memory Usage**: MapAnything can use significant GPU memory. For datasets with many frames, use `--memory_efficient` flag or increase stride.

4. **Depth Integration**: The script uses RTAB-Map's depth data as input to MapAnything, which can improve reconstruction accuracy compared to depth-from-scratch estimation.

## Troubleshooting

### Out of Memory
- Increase stride value
- Use `--memory_efficient` flag
- Limit frames with `--max_frames`

### Misaligned Reconstruction
- Verify poses are in camera frame (not body frame)
- Check calibration intrinsics match image resolution
- Ensure depth scale is correct (mm for uint16)

### Missing Frames
- Check that RGB and depth file numbers match
- Verify poses.txt has enough entries for all images
