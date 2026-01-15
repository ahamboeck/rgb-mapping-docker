#!/usr/bin/env python3
"""
Example usage:

python run_mapanything_inference.py \
  --input_dir  /path/to/images/ \
  --output_dir /path/to/output \
  --model_name facebook/map-anything \
  --conf_percentile 10 \
  --max_points 1000000
"""

import os
import re
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# ------------------------------------------------------------
# 1) TIMESTAMP + IO UTILS
# ------------------------------------------------------------
def extract_timestamp_from_filename(path: str) -> float:
    matches = re.findall(r"\d+", Path(path).name)
    if not matches:
        # Fallback to index if no digits found
        return 0.0
    return float(max(matches, key=len))


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


def export_tum_trajectory(
    tum_path: str,
    camera_poses_c2w: np.ndarray,
    frame_paths: list[str],
) -> None:
    """Exports poses to TUM format (timestamp x y z qx qy qz qw)."""
    lines = []
    for i, c2w in enumerate(camera_poses_c2w):
        ts = extract_timestamp_from_filename(frame_paths[i])
        t = c2w[:3, 3]
        qx, qy, qz, qw = R.from_matrix(c2w[:3, :3]).as_quat()
        lines.append(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    lines.sort(key=lambda s: float(s.split()[0]))
    with open(tum_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------
# 2) MERGE UTILS
# ------------------------------------------------------------
def merge_points_from_preds(
    predictions,
    conf_thresh: float | None,
    conf_percentile: float | None,
    max_points: int | None,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Merges per-view pts3d and img_no_norm into a single colored cloud."""
    all_pts = []
    all_rgb = []
    rng = np.random.default_rng(seed)

    for pred in predictions:
        # pts3d: (B, H, W, 3) -> (H*W, 3)
        pts = pred["pts3d"].detach().float().cpu().numpy().reshape(-1, 3)
        # img_no_norm: (B, H, W, 3) -> (H*W, 3)
        rgb = pred["img_no_norm"].detach().float().cpu().numpy().reshape(-1, 3)
        # conf: (B, H, W) -> (H*W,)
        conf = pred["conf"].detach().float().cpu().numpy().reshape(-1)
        
        # Apply Confidence Filtering
        if conf_percentile is not None:
            thr = np.percentile(conf, conf_percentile)
            mask = conf >= thr
        elif conf_thresh is not None:
            mask = conf >= conf_thresh
        else:
            mask = np.ones_like(conf, dtype=bool)

        # Apply Validity Mask (from model)
        if "mask" in pred:
            v_mask = pred["mask"].detach().float().cpu().numpy().reshape(-1) > 0.5
            mask = mask & v_mask

        pts = pts[mask]
        rgb = rgb[mask]

        # Clean NaNs
        good = np.isfinite(pts).all(axis=1)
        all_pts.append(pts[good])
        all_rgb.append(rgb[good])

    pts_final = np.concatenate(all_pts, axis=0)
    rgb_final = np.concatenate(all_rgb, axis=0)

    # Convert RGB [0, 1] or [0, 255] to uint8 safely
    if rgb_final.max() <= 1.01:
        rgb_final = (np.clip(rgb_final, 0, 1) * 255).astype(np.uint8)
    else:
        rgb_final = np.clip(rgb_final, 0, 255).astype(np.uint8)

    # Subsample if needed
    if max_points and pts_final.shape[0] > max_points:
        idx = rng.choice(pts_final.shape[0], size=max_points, replace=False)
        pts_final = pts_final[idx]
        rgb_final = rgb_final[idx]

    return pts_final, rgb_final


# ------------------------------------------------------------
# 3) MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("MapAnything -> colored PLY + TUM trajectory")
    parser.add_argument("--input_dir", required=True, help="Folder of images")
    parser.add_argument("--output_dir", required=True, help="Where to write outputs")
    parser.add_argument("--model_name", default="facebook/map-anything", help="Model ID")
    parser.add_argument("--conf_thresh", type=float, default=None, help="Absolute confidence threshold")
    parser.add_argument("--conf_percentile", type=float, default=10, help="Percentile to drop (default 10)")
    parser.add_argument("--max_points", type=int, default=2_000_000, help="Max points in final PLY")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Environment & Device
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    print(f"--> [Model] Loading {args.model_name} on {device}...")
    model = MapAnything.from_pretrained(args.model_name).to(device)

    # 2. Load Images
    print(f"--> [Input] Loading images from {args.input_dir}...")
    views = load_images(args.input_dir)
    # We need the paths for timestamping later
    frame_paths = [str(p) for p in sorted(Path(args.input_dir).iterdir()) if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]

    # 3. Inference
    print("--> [Inference] Running MapAnything...")
    with torch.no_grad():
        predictions = model.infer(
            views,
            memory_efficient_inference=False,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True
        )

    # 4. Export Point Cloud
    print("--> [PointCloud] Merging and filtering points...")
    pts, rgb = merge_points_from_preds(
        predictions,
        conf_thresh=args.conf_thresh,
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
        seed=args.seed
    )

    ply_path = os.path.join(args.output_dir, "reconstruction.ply")
    write_ply_xyzrgb_ascii(ply_path, pts, rgb)
    print(f"--> [Output] Saved PLY: {ply_path} (N={len(pts)})")

    # 5. Export Camera Data
    print("--> [Cameras] Exporting poses and intrinsics...")
    all_c2w = []
    all_intrinsics = []
    
    for pred in predictions:
        # MapAnything returns (B, 4, 4) or (B, 3, 3)
        all_c2w.append(pred["camera_poses"].detach().float().cpu().numpy()[0])
        all_intrinsics.append(pred["intrinsics"].detach().float().cpu().numpy()[0])

    c2w_np = np.stack(all_c2w)
    int_np = np.stack(all_intrinsics)

    # TUM Trajectory
    tum_path = os.path.join(args.output_dir, "trajectory_tum.txt")
    export_tum_trajectory(tum_path, c2w_np, frame_paths)
    
    # Raw Numpy Exports
    np.save(os.path.join(args.output_dir, "camera_poses_c2w.npy"), c2w_np)
    np.save(os.path.join(args.output_dir, "intrinsics.npy"), int_np)

    # Config Dump
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    print(f"--> [Success] Done. Outputs in {args.output_dir}")

if __name__ == "__main__":
    main()