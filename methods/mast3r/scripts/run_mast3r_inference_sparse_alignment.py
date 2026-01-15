#!/usr/bin/env python3
# =============================================================================
# MASt3R Sparse Global Alignment CLI (demo-style, scalable for 100s of images)
# =============================================================================
#
# USE TEMPLATES
# -------------
# 1) Good default for long RGB sequences (recommended starting point):
#
# python3 /workspace/scripts/run_mast3r_inference_sparse.py \
#   --input_dir  /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10 \
#   --output_dir /workspace/output/mast3r/biomasse_sga_swin10_384 \
#   --scene_graph swin-10 \
#   --image_size 384 \
#   --opt_level refine+depth \
#   --lr1 0.07 --niter1 300 \
#   --lr2 0.01 --niter2 300
#
# 2) Faster / lighter smoke test:
#
# python3 /workspace/scripts/run_mast3r_inference_sparse.py \
#   --input_dir  /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10 \
#   --output_dir /workspace/output/mast3r/biomasse_sga_swin5_320_coarse \
#   --scene_graph swin-5 \
#   --image_size 320 \
#   --opt_level coarse \
#   --niter1 200 --niter2 0
#
# 3) If drift / breaks (more temporal links):
#
#   --scene_graph swin-20
#
# OUTPUTS
# -------
#   <output_dir>/trajectory_tum.txt     (timestamp tx ty tz qx qy qz qw)
#   <output_dir>/reconstruction.ply     (dense point cloud)
#   <output_dir>/config.yaml            (run settings)
#   <output_dir>/cache/                 (intermediate cache for scalability)
#
# NOTES
# -----
# - This avoids the RAM blow-up you saw with `preds = inference(...)` by using
#   MASt3R's demo pipeline: `sparse_global_alignment(...)` + a cache directory.
# - `scene_graph` accepts: complete, swin-<K>, logwin-<K>, oneref-<ID>, etc.
# =============================================================================

import os
import sys
import re
import yaml
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

print("--> [System] Initializing MASt3R/DUSt3R environment...")

MAST3R_ROOT = os.environ.get("MAST3R_ROOT", "/tmp_build/mast3r")
DUST3R_PARENT = os.path.join(MAST3R_ROOT, "dust3r")

# Ensure imports work from anywhere (e.g., /workspace)
for p in (DUST3R_PARENT, MAST3R_ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

    # This matches MASt3R demo behavior for dust3r path wiring
    import mast3r.utils.path_to_dust3r  # noqa: F401

    from dust3r.utils.image import load_images
except Exception as e:
    print(f"!! [Fatal Error] Imports failed: {e}")
    print("   sys.path head:\n   " + "\n   ".join(sys.path[:10]))
    sys.exit(1)


def extract_timestamp(path: str) -> float:
    """
    Extract float timestamp from filename like '1768318763.511240.jpg'.
    Falls back to longest integer substring if no float is present.
    """
    base = os.path.basename(path)

    m = re.search(r"(\d+\.\d+)", base)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    nums = re.findall(r"\d+", base)
    if nums:
        try:
            return float(max(nums, key=len))
        except Exception:
            return 0.0

    return 0.0


def list_images_in_dir(input_dir: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    files.sort(key=extract_timestamp)
    return files


def export_tum_poses(scene, filelist, output_dir: str) -> None:
    """TUM: timestamp tx ty tz qx qy qz qw (camera-to-world)."""
    print("--> [Cameras] Exporting trajectory to TUM format...")

    poses = scene.get_im_poses().detach().cpu().numpy()  # (N,4,4) c2w

    tum_lines = []
    for i, fpath in enumerate(filelist):
        c2w = poses[i]
        ts = extract_timestamp(fpath)

        t = c2w[:3, 3]
        q = R.from_matrix(c2w[:3, :3]).as_quat()  # x y z w

        tum_lines.append(
            f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
        )

    tum_lines.sort(key=lambda s: float(s.split()[0]))

    out = os.path.join(output_dir, "trajectory_tum.txt")
    with open(out, "w") as f:
        f.write("\n".join(tum_lines))
    print(f"--> [Cameras] Saved: {out}")

def _to_numpy(x):
    """Convert torch tensors / lists-of-tensors to numpy without assuming stacking."""
    if isinstance(x, list):
        return [_to_numpy(xx) for xx in x]
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x  # already numpy

def save_dense_pointcloud_ply(scene, out_path: str, min_conf_thr: float = 2.0, clean_depth: bool = True):
    """
    Build one point cloud from dense points after optimization and save ASCII PLY.
    Handles both tensor and list-of-tensors outputs across repo versions.
    """
    print(f"--> [Output] Building dense point cloud (min_conf_thr={min_conf_thr}, clean_depth={clean_depth})...")

    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=clean_depth)

    pts3d = _to_numpy(pts3d)   # list of (H,W,3) or (H*W,3)
    confs = _to_numpy(confs)   # list of (H,W) or (H*W,)
    imgs  = _to_numpy(scene.imgs)  # list or tensor, RGB in [0..1]

    # Normalize to lists
    if not isinstance(pts3d, list):
        pts3d = list(pts3d)
    if not isinstance(confs, list):
        confs = list(confs)
    if not isinstance(imgs, list):
        imgs = list(imgs)

    all_pts, all_col = [], []

    for i in range(len(imgs)):
        img = imgs[i]
        conf = confs[i]
        p = pts3d[i]

        # Some versions return (H*W, 3); reshape to (H, W, 3)
        if p.ndim == 2:
            H, W = img.shape[:2]
            p = p.reshape(H, W, 3)

        # Some versions return conf as (H*W,) too
        if conf.ndim == 1:
            conf = conf.reshape(img.shape[:2])

        valid = (conf > min_conf_thr)
        finite = np.isfinite(p).all(axis=-1)
        m = valid & finite

        pts = p[m]
        col = img[m]

        if pts.size == 0:
            continue

        all_pts.append(pts.reshape(-1, 3))
        all_col.append((col.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8))

    if not all_pts:
        print("!! [Warn] No valid points to save (try lowering --min_conf_thr).")
        return

    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_col, axis=0)

    print(f"--> [Output] Saving PLY with {len(P)} points: {out_path}")
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(P)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(P, C):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Build filelist for sparse_global_alignment
    filelist = list_images_in_dir(args.input_dir)
    if args.max_images > 0:
        filelist = filelist[: args.max_images]

    if len(filelist) < 2:
        print("!! [Error] Need at least 2 images.")
        return

    # Load images (as demo does)
    print(f"--> [Images] Found {len(filelist)} files. Loading (size={args.image_size}) ...")
    imgs = load_images(filelist, size=args.image_size, verbose=True)

    # Load model
    print(f"--> [Model] Loading weights: {args.model_name} ...")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(device)

    # Build pairs
    print(f"--> [Pairs] Building pairs using scene_graph='{args.scene_graph}' (symmetrize={args.symmetrize})...")
    pairs = make_pairs(
        imgs,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=args.symmetrize,
        sim_mat=None,
    )

    # Cache dir (critical for scaling)
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Demo-like behavior: opt_depth controlled by opt_level string
    opt_depth = ("depth" in args.opt_level)

    # If coarse: skip refinement stage
    niter2 = args.niter2
    lr2 = args.lr2
    if args.opt_level == "coarse":
        niter2 = 0

    print("--> [SparseGA] Running sparse_global_alignment...")
    scene = sparse_global_alignment(
        filelist,
        pairs,
        cache_dir,
        model,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=lr2,
        niter2=niter2,
        device=device,
        opt_depth=opt_depth,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
    )

    # Export outputs
    export_tum_poses(scene, filelist, args.output_dir)

    ply_path = os.path.join(args.output_dir, "reconstruction.ply")
    save_dense_pointcloud_ply(
        scene,
        ply_path,
        min_conf_thr=args.min_conf_thr,
        clean_depth=args.clean_depth,
    )

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    print(f"\n--> [Success] Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MASt3R Sparse Global Alignment CLI (exports TUM + PLY)")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument(
        "--model_name",
        type=str,
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    )

    parser.add_argument("--image_size", type=int, default=384, help="Input image size for load_images()")
    parser.add_argument(
        "--scene_graph",
        type=str,
        default="swin-10",
        help="e.g. swin-10, swin-20, logwin-4, complete (not recommended for 700+ imgs)",
    )
    parser.add_argument("--symmetrize", action="store_true", default=True, help="Symmetrize pairs (demo default)")

    parser.add_argument("--opt_level", type=str, default="refine+depth", choices=["coarse", "refine", "refine+depth"])
    parser.add_argument("--lr1", type=float, default=0.07, help="Coarse LR")
    parser.add_argument("--niter1", type=int, default=300, help="Coarse iterations")
    parser.add_argument("--lr2", type=float, default=0.01, help="Fine LR")
    parser.add_argument("--niter2", type=int, default=300, help="Fine iterations")

    parser.add_argument("--matching_conf_thr", type=float, default=0.0, help="Matching confidence threshold")
    parser.add_argument("--shared_intrinsics", action="store_true", default=False, help="Optimize shared intrinsics")

    parser.add_argument("--min_conf_thr", type=float, default=2.0, help="Min confidence threshold for point cloud export")
    parser.add_argument("--clean_depth", action="store_true", default=True, help="Clean depthmaps for export")

    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="If >0, only use first N images (sorted by timestamp). Useful for quick tests.",
    )

    args = parser.parse_args()
    main(args)
