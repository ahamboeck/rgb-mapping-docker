#!/usr/bin/env python3
"""
Example usage (inside your container):

1) Clone + install:
   cd /tmp_build
   git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git
   cd HunyuanWorld-Mirror
   python3 -m pip install -r requirements.txt

2) Run on a folder of timestamped images (filenames contain digits like 20260113163102):
   export HYW_ROOT=/tmp_build/HunyuanWorld-Mirror

   python3 /workspace/scripts/run_hyw_worldmirror.py \
     --input_path /workspace/datasets/my_seq/images \
     --output_dir /workspace/output/hyw_my_seq \
     --image_size 518 \
     --conf_thresh 0.2 \
     --max_points 2000000

3) (Optional) Run with priors (you provide .npy arrays):
   # prior_camera_poses.npy: (S,4,4) camera-to-world
   # prior_depth.npy:        (S,H,W)
   # prior_intrinsics.npy:   (S,3,3)

   python3 /workspace/scripts/run_hyw_worldmirror.py \
     --input_path /workspace/datasets/my_seq/images \
     --output_dir /workspace/output/hyw_my_seq_priors \
     --prior_camera_poses /workspace/datasets/my_seq/prior_camera_poses.npy \
     --prior_depth        /workspace/datasets/my_seq/prior_depth.npy \
     --prior_intrinsics   /workspace/datasets/my_seq/prior_intrinsics.npy

Outputs (in --output_dir):
  - reconstruction.ply                (XYZ only)
  - reconstruction_colored.ply        (XYZRGB, colored from input images at inference resolution)
  - trajectory_tum.txt
  - camera_poses_c2w.npy
  - camera_intrs.npy (if available)
  - config.yaml

Notes:
- Colors come from the preprocessed tensor (target_size = --image_size), so they match the pointmap resolution.
- Timestamps come from digits in filenames (required).
"""

import os
import sys
import re
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------
# 1) DYNAMIC PATH RESOLUTION
# ------------------------------------------------------------
print("--> [System] Initializing HunyuanWorld-Mirror environment...")

HYW_ROOT = os.environ.get("HYW_ROOT", "/tmp_build/HunyuanWorld-Mirror")
HYW_ROOT = os.path.abspath(HYW_ROOT)

if os.path.isdir(HYW_ROOT) and HYW_ROOT not in sys.path:
    sys.path.insert(0, HYW_ROOT)

try:
    # Repo-style imports
    from src.models.models.worldmirror import WorldMirror
    from src.utils.inference_utils import extract_load_and_preprocess_images
except Exception as e:
    print(f"!! [Fatal Error] Failed to import HunyuanWorld-Mirror modules: {e}")
    print("   HYW_ROOT =", HYW_ROOT)
    print("   sys.path head:\n   " + "\n   ".join(sys.path[:8]))
    sys.exit(1)

print("--> [System] Core modules loaded successfully.")

# ------------------------------------------------------------
# 2) UTILS
# ------------------------------------------------------------
def extract_timestamp(filename: str) -> float:
    """
    Parses timestamps from filename (digits).
    e.g. extracts 20260113163102 from .../img_20260113163102.png
    Returns float(timestamp_digits).
    Raises if no digits found (since timestamps are required).
    """
    matches = re.findall(r"\d+", Path(filename).name)
    if not matches:
        raise ValueError(f"No digits found in filename for timestamp extraction: {filename}")
    ts_str = max(matches, key=len)
    try:
        return float(ts_str)
    except Exception as e:
        raise ValueError(f"Failed to parse timestamp '{ts_str}' from {filename}: {e}")


def write_ply_xyz(path: str, xyz: np.ndarray) -> None:
    """
    Minimal ASCII PLY writer for XYZ points (no colors).
    xyz: (N,3) float
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {xyz.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    )

    with open(path, "w") as f:
        f.write(header + "\n")
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """
    ASCII PLY writer for XYZRGB points.
    xyz: (N,3) float
    rgb: (N,3) uint8
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (N,3), got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz and rgb must have same N, got {xyz.shape[0]} vs {rgb.shape[0]}")

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    header = "\n".join(
        [
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
        ]
    )

    with open(path, "w") as f:
        f.write(header + "\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def export_tum_trajectory(
    tum_path: str,
    camera_poses_c2w: np.ndarray,  # (S,4,4)
    frame_paths: list[str],
) -> None:
    """
    TUM format:
      timestamp tx ty tz qx qy qz qw

    Uses camera-to-world (c2w).
    Quaternion from rotation matrix (x, y, z, w).
    Timestamp from filename digits (required).
    """
    if camera_poses_c2w.ndim != 3 or camera_poses_c2w.shape[1:] != (4, 4):
        raise ValueError(f"camera_poses_c2w must be (S,4,4), got {camera_poses_c2w.shape}")

    S = camera_poses_c2w.shape[0]
    if len(frame_paths) != S:
        raise ValueError(
            f"Frame path count ({len(frame_paths)}) != pose count ({S}). "
            "Because timestamps come from filenames, these must match."
        )

    lines = []
    for i in range(S):
        c2w = camera_poses_c2w[i]
        t = c2w[:3, 3]
        qx, qy, qz, qw = R.from_matrix(c2w[:3, :3]).as_quat()
        ts = extract_timestamp(frame_paths[i])

        lines.append(
            f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )

    lines.sort(key=lambda s: float(s.split()[0]))

    with open(tum_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def flatten_pointmaps_to_cloud(
    pts3d: np.ndarray,        # (S,H,W,3)
    conf: np.ndarray | None,  # (S,H,W) or (S,H,W,1)
    conf_thresh: float,
    max_points: int | None,
    seed: int = 0,
) -> np.ndarray:
    """
    Merge per-view pointmaps into one point cloud (XYZ only).
    Optionally filter by confidence and cap total points by random sampling.
    """
    if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
        raise ValueError(f"pts3d must be (S,H,W,3), got {pts3d.shape}")

    pts = pts3d.reshape(-1, 3)

    if conf is not None:
        c = conf
        if c.ndim == 4 and c.shape[-1] == 1:
            c = c[..., 0]
        if c.ndim != 3:
            raise ValueError(f"conf must be (S,H,W) or (S,H,W,1), got {conf.shape}")
        c = c.reshape(-1)
        pts = pts[c >= conf_thresh]

    pts = pts[np.isfinite(pts).all(axis=1)]

    if max_points is not None and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    return pts


def flatten_pointmaps_to_colored_cloud(
    pts3d: np.ndarray,              # (S,H,W,3)
    imgs_01: np.ndarray,            # (S,3,H,W) float in [0,1]
    conf: np.ndarray | None,        # (S,H,W) or (S,H,W,1)
    conf_thresh: float,
    max_points: int | None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (xyz, rgb_uint8) both (N,3).
    RGB is taken from the preprocessed image tensor, aligned pixel-wise with pts3d.
    """
    if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
        raise ValueError(f"pts3d must be (S,H,W,3), got {pts3d.shape}")
    if imgs_01.ndim != 4 or imgs_01.shape[1] != 3:
        raise ValueError(f"imgs_01 must be (S,3,H,W), got {imgs_01.shape}")

    S, H, W, _ = pts3d.shape
    if imgs_01.shape[0] != S or imgs_01.shape[2] != H or imgs_01.shape[3] != W:
        raise ValueError(
            f"Shape mismatch: pts3d is (S,H,W,3)=({S},{H},{W},3) "
            f"but imgs_01 is (S,3,H,W)={imgs_01.shape}"
        )

    xyz = pts3d.reshape(-1, 3)

    # RGB: (S,3,H,W) -> (S,H,W,3) -> flatten to (N,3)
    rgb = np.transpose(imgs_01, (0, 2, 3, 1)).reshape(-1, 3)
    rgb = (rgb * 255.0 + 0.5).astype(np.uint8)

    if conf is not None:
        c = conf
        if c.ndim == 4 and c.shape[-1] == 1:
            c = c[..., 0]
        if c.ndim != 3:
            raise ValueError(f"conf must be (S,H,W) or (S,H,W,1), got {conf.shape}")
        c = c.reshape(-1)
        keep = c >= conf_thresh
        xyz = xyz[keep]
        rgb = rgb[keep]

    good = np.isfinite(xyz).all(axis=1)
    xyz = xyz[good]
    rgb = rgb[good]

    if max_points is not None and xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def list_images_in_dir(image_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files_sorted = sorted(files, key=lambda p: p.name)
    return [str(p) for p in files_sorted]


# ------------------------------------------------------------
# 3) MAIN
# ------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Model
    print(f"--> [Model] Loading: {args.model_name}")
    model = WorldMirror.from_pretrained(args.model_name).to(device)
    model.eval()

    # 2) Images / frames
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"--> [Input] Loading & preprocessing: {input_path}")
    imgs = extract_load_and_preprocess_images(
        input_path,
        fps=args.fps,
        target_size=args.image_size,
    ).to(device)

    if imgs.ndim != 5 or imgs.shape[0] != 1:
        raise RuntimeError(f"Unexpected image tensor shape: {tuple(imgs.shape)} (expected [1,S,3,H,W])")

    S = int(imgs.shape[1])
    if S < 2:
        raise RuntimeError("Need at least 2 images/frames.")

    # 2b) Frame paths for timestamp extraction
    if input_path.is_dir():
        frame_paths = list_images_in_dir(input_path)
        if len(frame_paths) != S:
            raise RuntimeError(
                f"Preprocess returned S={S} frames, but folder contains {len(frame_paths)} images.\n"
                "Because timestamps come from filenames, these must match.\n"
                "Fix by ensuring preprocessing does not subsample (fps irrelevant for folders), "
                "or provide exactly the images you want in the folder."
            )
    else:
        raise RuntimeError(
            "You provided a video input, but this script requires real timestamps from filenames.\n"
            "Use a folder of timestamped images, or implement a separate timestamp source for video."
        )

    # 3) Optional priors
    cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]
    inputs = {"img": imgs}

    if args.prior_camera_poses is not None:
        cam = np.load(args.prior_camera_poses)
        cam_t = torch.from_numpy(cam).float()
        if cam_t.ndim == 3:  # (S,4,4) -> (1,S,4,4)
            cam_t = cam_t.unsqueeze(0)
        inputs["camera_poses"] = cam_t.to(device)
        cond_flags[0] = 1

    if args.prior_depth is not None:
        dep = np.load(args.prior_depth)
        dep_t = torch.from_numpy(dep).float()
        if dep_t.ndim == 3:  # (S,H,W) -> (1,S,H,W)
            dep_t = dep_t.unsqueeze(0)
        inputs["depthmap"] = dep_t.to(device)
        cond_flags[1] = 1

    if args.prior_intrinsics is not None:
        intr = np.load(args.prior_intrinsics)
        intr_t = torch.from_numpy(intr).float()
        if intr_t.ndim == 3:  # (S,3,3) -> (1,S,3,3)
            intr_t = intr_t.unsqueeze(0)
        inputs["camera_intrs"] = intr_t.to(device)
        cond_flags[2] = 1

    print(f"--> [Inference] Frames: {S}, cond_flags={cond_flags}")
    with torch.no_grad():
        predictions = model(views=inputs, cond_flags=cond_flags)

    # 4) Outputs
    pts3d = predictions["pts3d"][0].detach().cpu().numpy()  # (S,H,W,3)

    pts3d_conf = None
    if "pts3d_conf" in predictions and predictions["pts3d_conf"] is not None:
        pts3d_conf = predictions["pts3d_conf"][0].detach().cpu().numpy()  # (S,H,W) likely

    camera_poses = predictions["camera_poses"][0].detach().cpu().numpy()  # (S,4,4) c2w

    camera_intrs = None
    if "camera_intrs" in predictions and predictions["camera_intrs"] is not None:
        camera_intrs = predictions["camera_intrs"][0].detach().cpu().numpy()  # (S,3,3)

    # 5) Save
    out_dir = args.output_dir

    print("--> [Output] Exporting TUM trajectory...")
    export_tum_trajectory(
        tum_path=os.path.join(out_dir, "trajectory_tum.txt"),
        camera_poses_c2w=camera_poses,
        frame_paths=frame_paths,
    )

    print("--> [Output] Exporting merged PLY point cloud (XYZ only)...")
    cloud_xyz = flatten_pointmaps_to_cloud(
        pts3d=pts3d,
        conf=pts3d_conf,
        conf_thresh=args.conf_thresh,
        max_points=args.max_points,
        seed=args.seed,
    )
    write_ply_xyz(os.path.join(out_dir, "reconstruction.ply"), cloud_xyz)

    print("--> [Output] Exporting colored merged PLY point cloud (XYZRGB)...")
    imgs_np = imgs[0].detach().cpu().numpy()  # (S,3,H,W) in [0,1]
    cloud_xyz_c, cloud_rgb = flatten_pointmaps_to_colored_cloud(
        pts3d=pts3d,
        imgs_01=imgs_np,
        conf=pts3d_conf,
        conf_thresh=args.conf_thresh,
        max_points=args.max_points,
        seed=args.seed,
    )
    write_ply_xyzrgb(os.path.join(out_dir, "reconstruction_colored.ply"), cloud_xyz_c, cloud_rgb)

    print("--> [Output] Saving numpy dumps...")
    np.save(os.path.join(out_dir, "camera_poses_c2w.npy"), camera_poses)
    if camera_intrs is not None:
        np.save(os.path.join(out_dir, "camera_intrs.npy"), camera_intrs)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    print(f"\n--> [Success] Done. Output saved to: {out_dir}")


# ------------------------------------------------------------
# 4) CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("HunyuanWorld-Mirror -> PLY + Colored PLY + TUM trajectory")
    ap.add_argument("--input_path", required=True, help="Folder of timestamped images (required for TUM timestamps)")
    ap.add_argument("--output_dir", required=True, help="Where to write outputs")
    ap.add_argument("--model_name", default="tencent/HunyuanWorld-Mirror", help="HF model id")
    ap.add_argument("--image_size", type=int, default=518, help="target_size passed to preprocessing")
    ap.add_argument("--fps", type=float, default=1.0, help="Only used for video; folder inputs ignore it")
    ap.add_argument("--conf_thresh", type=float, default=0.2, help="Confidence filter for points (if pts3d_conf exists)")
    ap.add_argument("--max_points", type=int, default=2_000_000, help="Cap merged cloud size via random sampling (0 disables)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for point sampling")

    # Optional priors (simple .npy arrays you control)
    ap.add_argument("--prior_camera_poses", default=None, help="npy/npz: (S,4,4) or (1,S,4,4)")
    ap.add_argument("--prior_depth", default=None, help="npy/npz: (S,H,W) or (1,S,H,W)")
    ap.add_argument("--prior_intrinsics", default=None, help="npy/npz: (S,3,3) or (1,S,3,3)")

    args = ap.parse_args()

    # normalize max_points: allow 0 to mean "no cap"
    if args.max_points == 0:
        args.max_points = None

    main(args)
