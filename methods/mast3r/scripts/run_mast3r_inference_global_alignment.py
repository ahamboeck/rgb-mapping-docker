#!/usr/bin/env python3
# Example usage:
#
#   # (Recommended for long sequences) Keep pairing sparse to avoid RAM blowups:
#   python3 /workspace/scripts/run_mast3r_inference_global_alignment.py \
#     --input_dir /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10 \
#     --output_dir /workspace/output/mast3r/biomasse_every_10_swin5_384 \
#     --scene_graph swin-5 \
#     --image_size 384 \
#     --batch_size 32 \
#     --niter 300
#
#   # Even lighter (if you previously hit OOM / host became unresponsive):
#   python3 /workspace/scripts/run_mast3r_inference_global_alignment.py \
#     --input_dir /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10 \
#     --output_dir /workspace/output/mast3r/biomasse_every_10_swin3_320 \
#     --scene_graph swin-3 \
#     --image_size 320 \
#     --batch_size 16 \
#     --niter 150
#
# Notes:
# - batch_size is "pairs per forward pass", NOT "images per forward pass".
# - Avoid '--scene_graph complete' for 700+ images (pair count explodes).
# - Outputs: reconstruction.ply + trajectory_tum.txt + config.yaml

import os
import sys
import re
import time
import yaml
import torch
import argparse
import importlib.util
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------
# 1. DYNAMIC PATH & MODULE RESOLUTION
# ------------------------------------------------------------
print("--> [System] Initializing MASt3R/DUSt3R environment...")

MAST3R_ROOT = os.environ.get("MAST3R_ROOT", "/tmp_build/mast3r")
DUST3R_PARENT = os.path.join(MAST3R_ROOT, "dust3r")

for p in (DUST3R_PARENT, MAST3R_ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    import mast3r.utils.path_to_dust3r  # noqa: F401

    from dust3r.inference import inference
    from dust3r.utils.image import load_images
except Exception as e:
    print(f"!! [Fatal Error] Core imports failed: {e}")
    print("   sys.path head:\n   " + "\n   ".join(sys.path[:8]))
    sys.exit(1)

# ------------------------------------------------------------
# 1b. ALIGNER API RESOLUTION
# ------------------------------------------------------------
def resolve_aligner_api():
    # Preferred: function-based API
    try:
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        return {"kind": "function", "global_aligner": global_aligner, "mode_enum": GlobalAlignerMode}
    except Exception:
        pass

    # Fallback: class-based API if present
    candidates = [
        "dust3r.cloud_opt.global_aligner",
        "dust3r.cloud_opt.global_alignment",
        "dust3r.cloud_opt",
    ]
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=["GlobalAligner"])
            if hasattr(mod, "GlobalAligner"):
                return {"kind": "class", "GlobalAligner": getattr(mod, "GlobalAligner")}
        except Exception:
            continue

    # Last resort: search filesystem for "GlobalAligner"
    print("!! [Debug] Standard aligner imports failed. Searching filesystem for GlobalAligner...")
    import glob

    search_pattern = os.path.join(MAST3R_ROOT, "**/global_align*.py")
    for module_path in glob.glob(search_pattern, recursive=True):
        try:
            spec = importlib.util.spec_from_file_location("dynamic_aligner", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            if hasattr(module, "GlobalAligner"):
                print(f"--> [Debug] Found GlobalAligner in: {module_path}")
                return {"kind": "class", "GlobalAligner": getattr(module, "GlobalAligner")}
        except Exception:
            continue

    raise ImportError(
        "Could not resolve any aligner API.\n"
        "Tried: dust3r.cloud_opt global_aligner (function) and GlobalAligner (class).\n"
        "Make sure dust3r is importable and contains cloud_opt."
    )


try:
    ALIGNER = resolve_aligner_api()
    print("--> [System] All core modules loaded successfully.")
except Exception as e:
    print(f"!! [Fatal Error] {e}")
    sys.exit(1)

# ------------------------------------------------------------
# 2. UTILITY FUNCTIONS
# ------------------------------------------------------------
def extract_timestamp(filename: str) -> float:
    match = re.findall(r"\d+", os.path.basename(filename))
    if match:
        ts_str = max(match, key=len)
        try:
            return float(ts_str)
        except Exception:
            return 0.0
    return 0.0


def export_tum_poses(scene, images, output_dir: str) -> None:
    """Exports trajectory in TUM format: [timestamp] [tx] [ty] [tz] [qx] [qy] [qz] [qw]"""
    print("--> [Cameras] Exporting trajectory to TUM format...")
    poses = scene.get_im_poses().detach().cpu().numpy()  # c2w

    tum_lines = []
    for i in range(len(images)):
        c2w = poses[i]
        timestamp = extract_timestamp(images[i]["instance"])
        t = c2w[:3, 3]
        quat = R.from_matrix(c2w[:3, :3]).as_quat()  # (x, y, z, w)
        tum_lines.append(
            f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
        )

    tum_lines.sort(key=lambda x: float(x.split()[0]))
    tum_path = os.path.join(output_dir, "trajectory_tum.txt")
    with open(tum_path, "w") as f:
        f.write("\n".join(tum_lines))
    print(f"--> [Cameras] Saved TUM file to: {tum_path}")


def _export_pointcloud_ply_from_scene(scene, ply_path: str, min_conf_thr: float = 2.0, clean_depth: bool = True) -> dict:
    """
    Manual PLY export fallback for scenes that do not implement save_ply().
    Uses scene.get_dense_pts3d() if available.
    Exports a colored pointcloud if scene.imgs exists.
    """
    import numpy as np
    import trimesh

    if not hasattr(scene, "get_dense_pts3d"):
        raise AttributeError("Scene has no save_ply() and no get_dense_pts3d(); cannot export PLY.")

    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=clean_depth)
    # pts3d: list/tuple of (H,W,3) tensors
    # confs: list/tuple of (H,W) tensors

    pts_list = []
    col_list = []

    has_imgs = hasattr(scene, "imgs") and scene.imgs is not None
    imgs = scene.imgs if has_imgs else None

    for i in range(len(pts3d)):
        p = pts3d[i].detach().cpu().numpy().reshape(-1, 3)
        c = confs[i].detach().cpu().numpy().reshape(-1)

        valid = np.isfinite(p).all(axis=1) & (c > float(min_conf_thr))
        if not np.any(valid):
            continue

        pts_list.append(p[valid])

        if has_imgs:
            im = imgs[i]
            if torch.is_tensor(im):
                im = im.detach().cpu().numpy()
            # im is typically (H,W,3) float in [0..1]
            im = im.reshape(-1, 3)
            col = im[valid]
            # convert to uint8
            col = (np.clip(col, 0.0, 1.0) * 255.0).astype(np.uint8)
            col_list.append(col)

    if not pts_list:
        raise RuntimeError("No valid points to export (all filtered out). Try lowering --min_conf_thr.")

    pts = np.concatenate(pts_list, axis=0)
    if col_list:
        col = np.concatenate(col_list, axis=0)
        pc = trimesh.PointCloud(vertices=pts, colors=col)
        color_mode = "rgb"
    else:
        pc = trimesh.PointCloud(vertices=pts)
        color_mode = "none"

    pc.export(ply_path)

    return {
        "export_method": "fallback_trimesh_pointcloud",
        "min_conf_thr": float(min_conf_thr),
        "clean_depth": bool(clean_depth),
        "points_exported": int(pts.shape[0]),
        "color_mode": color_mode,
    }


def save_reconstruction(scene, out_dir: str, notes: list, min_conf_thr: float = 2.0, clean_depth: bool = True) -> dict:
    """
    Save reconstruction as PLY. If scene.save_ply does not exist, export manually.
    Returns dict with export metadata to include in config.yaml.
    """
    ply_path = os.path.join(out_dir, "reconstruction.ply")
    print(f"--> [Output] Saving PLY: {ply_path}")

    if hasattr(scene, "save_ply") and callable(getattr(scene, "save_ply")):
        scene.save_ply(ply_path)
        return {"export_method": "scene.save_ply", "ply_path": ply_path}

    # Bug note requested by user
    msg = "Scene object has no save_ply(); exported pointcloud manually from get_dense_pts3d()."
    notes.append(msg)
    print(f"!! [Warn] {msg}")

    meta = _export_pointcloud_ply_from_scene(scene, ply_path, min_conf_thr=min_conf_thr, clean_depth=clean_depth)
    meta["ply_path"] = ply_path
    return meta

# ------------------------------------------------------------
# 3. ALIGNMENT WRAPPER
# ------------------------------------------------------------
def _pick_mode(mode_enum):
    for cand in ("PointCloudOptimizer", "PCD", "PnP", "Pairwise"):
        if hasattr(mode_enum, cand):
            return getattr(mode_enum, cand)
    return None


def _call_global_aligner_dust3r_output(global_aligner, dust3r_output, device, mode):
    """
    In your repo, dust3r.cloud_opt.global_aligner expects a single dust3r_output object
    (the return of inference), not (pairs, preds).

    Avoid keyword args for device/mode to prevent "multiple values" errors.
    Try common signatures:
      1) global_aligner(out, device, mode)
      2) global_aligner(out, device)
      3) global_aligner(out)
      4) global_aligner(out, mode, device)
    """
    if mode is not None:
        try:
            return global_aligner(dust3r_output, device, mode)
        except TypeError:
            pass
        try:
            return global_aligner(dust3r_output, mode, device)
        except TypeError:
            pass

    try:
        return global_aligner(dust3r_output, device)
    except TypeError:
        pass

    return global_aligner(dust3r_output)


def run_alignment(dust3r_output, device: str, niter: int):
    print(f"--> [Alignment] Optimizing for {niter} iterations...")

    if ALIGNER["kind"] == "function":
        global_aligner = ALIGNER["global_aligner"]
        ModeEnum = ALIGNER["mode_enum"]
        mode = _pick_mode(ModeEnum)

        scene = _call_global_aligner_dust3r_output(global_aligner, dust3r_output, device, mode)
        scene.compute_global_alignment(init="mst", niter=niter)
        return scene

    # class-based fallback
    GlobalAligner = ALIGNER["GlobalAligner"]

    mode = None
    if hasattr(GlobalAligner, "Mode"):
        for cand in ("PCD", "PointCloudOptimizer"):
            if hasattr(GlobalAligner.Mode, cand):
                mode = getattr(GlobalAligner.Mode, cand)
                break

    if mode is None:
        scene = GlobalAligner(dust3r_output, device)
    else:
        scene = GlobalAligner(dust3r_output, device, mode=mode)

    scene.compute_global_alignment(init="mst", niter=niter)
    return scene

# ------------------------------------------------------------
# 4. EXECUTION PIPELINE
# ------------------------------------------------------------
def main(args):
    t0 = time.time()
    notes = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--> [Model] Loading weights: {args.model_name}...")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(device)

    print(f"--> [Images] Loading from: {args.input_dir}...")
    images = load_images(args.input_dir, size=args.image_size, verbose=True)
    if len(images) < 2:
        print("!! [Error] Scene requires at least 2 images.")
        return

    print(f"--> [Pairs] Building pairs using scene_graph='{args.scene_graph}' (symmetrize={args.symmetrize})...")
    pairs = make_pairs(
        images,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=args.symmetrize,
        sim_mat=None,
    )

    # Inference timing
    t_infer0 = time.time()
    print(f"--> [Inference] Processing {len(pairs)} pairs (Batch Size: {args.batch_size})...")
    dust3r_output = inference(pairs, model, device, batch_size=args.batch_size)
    t_infer1 = time.time()

    # Alignment timing
    t_align0 = time.time()
    scene = run_alignment(dust3r_output, device=device, niter=args.niter)
    t_align1 = time.time()

    print("--> [Output] Saving files...")
    export_meta = save_reconstruction(
        scene,
        args.output_dir,
        notes=notes,
        min_conf_thr=args.min_conf_thr,
        clean_depth=args.clean_depth,
    )
    export_tum_poses(scene, images, args.output_dir)

    t1 = time.time()

    config = vars(args).copy()
    config["timing_sec"] = {
        "total": float(t1 - t0),
        "inference": float(t_infer1 - t_infer0),
        "alignment": float(t_align1 - t_align0),
        "io_export": float(t1 - t_align1),
    }
    config["export"] = export_meta
    if notes:
        config["notes"] = notes

    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n--> [Success] Processing complete. Data saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MASt3R to TUM Trajectory Script (Global Aligner)")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument(
        "--model_name",
        type=str,
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (pairs per forward pass)")
    parser.add_argument("--niter", type=int, default=300, help="Alignment iterations")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size for load_images()")
    parser.add_argument(
        "--scene_graph",
        type=str,
        default="complete",
        help="Pair graph (e.g., complete, swin-10, logwin-4, oneref-0, etc.)",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        default=True,
        help="Symmetrize pairs (matches demo default).",
    )
    parser.add_argument(
        "--min_conf_thr",
        type=float,
        default=2.0,
        help="Confidence threshold when exporting fallback pointcloud PLY.",
    )
    parser.add_argument(
        "--clean_depth",
        action="store_true",
        default=True,
        help="Use clean_depth=True for get_dense_pts3d() during fallback export.",
    )

    args = parser.parse_args()
    main(args)
