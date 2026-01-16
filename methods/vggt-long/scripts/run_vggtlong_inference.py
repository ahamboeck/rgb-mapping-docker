#!/usr/bin/env python3
"""
Example usage:

python run_vggt_long_inference.py \
  --input_dir  /workspace/datasets/sequence_01/images \
  --output_dir /workspace/output/vggt_long_results \
  --model_name VGGT \
  --chunk_size 16 \
  --max_points 2000000
"""

import os
import re
import sys
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------
# PATH SETUP (Crucial for relative imports and model loading)
# ------------------------------------------------------------
# Add current directory and parent to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))

# Change working directory to repo root to fix internal relative path issues 
# (e.g. looking for ./LoopModels/dinov2/hubconf.py)
os.chdir(repo_root)

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import the actual repository modules
try:
    from vggt_long import VGGT_Long
    from loop_utils.config_utils import load_config
except ImportError:
    print("!! Could not import VGGT_Long or load_config.")
    print(f"!! Current Working Directory: {os.getcwd()}")
    print(f"!! Current sys.path: {sys.path[:3]}...")
    sys.exit(1)

# ------------------------------------------------------------
# 1) TIMESTAMP + IO UTILS
# ------------------------------------------------------------
def extract_timestamp_from_filename(path: str) -> float:
    matches = re.findall(r"\d+", Path(path).name)
    if not matches:
        return 0.0
    return float(max(matches, key=len))


def list_images_sorted(image_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return [str(p) for p in sorted(files, key=lambda p: p.name)]


def write_ply_xyzrgb_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Minimal ASCII PLY writer for XYZ + RGB (uchar)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])

    with open(path, "w") as f:
        f.write(header + "\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def merge_pcd_directory(pcd_dir: str, output_path: str, max_points: int = None):
    """
    Reads all *_pcd.ply files in the directory and merges them.
    """
    pcd_files = sorted(list(Path(pcd_dir).glob("*_pcd.ply")), key=lambda x: int(x.name.split('_')[0]))
    if not pcd_files:
        print(f"!! No PLY files found in {pcd_dir}")
        return

    all_xyz = []
    all_rgb = []

    print(f"--> [Merging] Reading {len(pcd_files)} chunk PLY files...")
    for f in pcd_files:
        with open(f, 'r') as ply:
            lines = ply.readlines()
            # Find end_header
            header_idx = 0
            for i, line in enumerate(lines):
                if "end_header" in line:
                    header_idx = i + 1
                    break
            
            # Parse data
            for line in lines[header_idx:]:
                parts = line.split()
                if len(parts) == 6:
                    all_xyz.append([float(x) for x in parts[:3]])
                    all_rgb.append([int(x) for x in parts[3:]])

    xyz = np.array(all_xyz, dtype=np.float32)
    rgb = np.array(all_rgb, dtype=np.uint8)

    if max_points and xyz.shape[0] > max_points:
        print(f"--> [Subsampling] Reducing from {xyz.shape[0]} to {max_points} points...")
        indices = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]

    write_ply_xyzrgb_ascii(output_path, xyz, rgb)
    print(f"--> [Success] Merged PCL saved to: {output_path}")

def export_tum_trajectory(
    tum_path: str,
    camera_poses_c2w: np.ndarray,
    frame_paths: list[str],
) -> None:
    """Exports poses to TUM format (timestamp x y z qx qy qz qw)."""
    lines = []
    for i, c2w in enumerate(camera_poses_c2w):
        if c2w is None: continue
        ts = extract_timestamp_from_filename(frame_paths[i])
        t = c2w[:3, 3]
        qx, qy, qz, qw = R.from_matrix(c2w[:3, :3]).as_quat()
        lines.append(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    lines.sort(key=lambda s: float(s.split()[0]))
    with open(tum_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------
# 3) MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("VGGT-Long -> Large-scale PLY + TUM trajectory")
    parser.add_argument("--input_dir", required=True, help="Path to image sequence")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    parser.add_argument("--config", default=os.path.join(repo_root, "configs/base_config.yaml"), help="Path to base config.yaml")
    parser.add_argument("--model_name", default="VGGT", choices=["VGGT", "Pi3", "Mapanything"], help="Base model weights")
    parser.add_argument("--chunk_size", type=int, default=16, help="Frames per chunk")
    parser.add_argument("--max_points", type=int, default=2_000_000, help="Max points in final PLY")
    args = parser.parse_args()

    # Updated allocator config env var to avoid deprecation warning
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load and Patch Configuration
    print(f"--> [Config] Loading base config from: {args.config}")
    config = load_config(args.config)
    
    # Overwrite config with CLI arguments
    config['Weights']['model'] = args.model_name
    config['Model']['chunk_size'] = args.chunk_size
    config['Model']['overlap'] = args.chunk_size // 2
    
    # Ensure pointcloud sampling matches user request
    if 'Pointcloud_Save' not in config['Model']:
        config['Model']['Pointcloud_Save'] = {}
    config['Model']['Pointcloud_Save']['sample_ratio'] = 1.0 # We handle subsampling in the script

    # 2. Initialize VGGT_Long Pipeline
    print(f"--> [System] Initializing VGGT-Long (Implementation: VGGT_Long)")
    # VGGT_Long takes (image_dir, save_dir, config_dict)
    vggt_pipeline = VGGT_Long(args.input_dir, args.output_dir, config)

    # 3. Run Inference
    print(f"--> [Inference] Processing sequence...")
    vggt_pipeline.run()

    # 4. Handle Outputs & Merging
    pcd_dir = os.path.join(args.output_dir, 'pcd')
    final_ply = os.path.join(pcd_dir, 'combined_pcd.ply')
    
    if not os.path.exists(final_ply):
        print("--> [Merging] Global PLY not found. Manually merging chunk files...")
        merge_pcd_directory(pcd_dir, final_ply, max_points=args.max_points)
    else:
        print(f"--> [Output] Found global PLY: {final_ply}")
    
    # 5. Export Trajectory to TUM
    # The pipeline saves camera_poses.txt (4x4 flattened)
    poses_path = os.path.join(args.output_dir, 'camera_poses.txt')
    if os.path.exists(poses_path):
        print("--> [Trajectory] Converting camera_poses.txt to TUM format...")
        raw_poses = np.loadtxt(poses_path)
        frame_paths = list_images_sorted(Path(args.input_dir))
        
        # Reshape to (N, 4, 4)
        num_frames = raw_poses.shape[0]
        camera_poses_c2w = raw_poses.reshape(num_frames, 4, 4)
        
        tum_path = os.path.join(args.output_dir, "trajectory_tum.txt")
        export_tum_trajectory(tum_path, camera_poses_c2w, frame_paths)
        print(f"--> [Output] Saved TUM trajectory: {tum_path}")

    # 6. Cleanup
    vggt_pipeline.close()
    print(f"\n--> [Success] Processing complete. Results in: {args.output_dir}")

if __name__ == "__main__":
    main()