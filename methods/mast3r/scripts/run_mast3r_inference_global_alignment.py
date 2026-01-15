#!/usr/bin/env python3
# Example usage:
#
#   python3 /workspace/scripts/run_mast3r_inference_global_alignment.py \
#     --input_dir /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10 \
#     --output_dir /workspace/output/mast3r/biomasse_every_10_swin3_320 \
#     --scene_graph swin-3 \
#     --image_size 320 \
#     --batch_size 64 \
#     --niter 300 \
#     --ply_stride 4 \
#     --max_points 5000000
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


def resolve_aligner_api():
    try:
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        return {"kind": "function", "global_aligner": global_aligner, "mode_enum": GlobalAlignerMode}
    except Exception:
        pass

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

    raise ImportError("Could not resolve any aligner API (global_aligner or GlobalAligner).")


try:
    ALIGNER = resolve_aligner_api()
    print("--> [System] All core modules loaded successfully.")
except Exception as e:
    print(f"!! [Fatal Error] {e}")
    sys.exit(1)


# -------------------------
# Utilities
# -------------------------
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
    print("--> [Cameras] Exporting trajectory to TUM format...")
    poses = scene.get_im_poses().detach().cpu().numpy()  # c2w
    tum_lines = []
    for i in range(len(images)):
        c2w = poses[i]
        timestamp = extract_timestamp(images[i]["instance"])
        t = c2w[:3, 3]
        quat = R.from_matrix(c2w[:3, :3]).as_quat()  # (x,y,z,w)
        tum_lines.append(
            f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
        )
    tum_lines.sort(key=lambda x: float(x.split()[0]))
    tum_path = os.path.join(output_dir, "trajectory_tum.txt")
    with open(tum_path, "w") as f:
        f.write("\n".join(tum_lines))
    print(f"--> [Cameras] Saved TUM file to: {tum_path}")


def _pick_mode(mode_enum):
    for cand in ("PointCloudOptimizer", "PCD", "PnP", "Pairwise"):
        if hasattr(mode_enum, cand):
            return getattr(mode_enum, cand)
    return None


def _call_global_aligner_dust3r_output(global_aligner, dust3r_output, device, mode):
    # Avoid keyword args for device/mode to dodge "multiple values" issues.
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

    GlobalAligner = ALIGNER["GlobalAligner"]
    scene = GlobalAligner(dust3r_output, device)
    scene.compute_global_alignment(init="mst", niter=niter)
    return scene


# -------------------------
# Robust getters for depth
# -------------------------
def _stack_any_depth_list(depth_list):
    """
    depth_list: list/tuple of tensors/ndarrays possibly nested.
    Return torch.Tensor [N,H,W] on CPU float32.
    """
    # unwrap nested tuples like [(depth, conf), ...] -> [depth, ...]
    if len(depth_list) > 0 and isinstance(depth_list[0], (tuple, list)) and len(depth_list[0]) > 0:
        depth_list = [d[0] for d in depth_list]

    out = []
    for d in depth_list:
        if torch.is_tensor(d):
            out.append(d.detach().cpu())
        else:
            # numpy or list
            out.append(torch.as_tensor(d).cpu())
    # make sure shapes match
    return torch.stack(out, dim=0).to(torch.float32)


def _get_depthmaps_from_scene(scene):
    """
    Try multiple DUSt3R variants and normalize to torch.Tensor [N,H,W] on CPU.
    Handles:
      - Tensor [N,H,W]
      - list[Tensor[H,W]]
      - tuple/list nested: [(Tensor[H,W], ...), ...]
      - Parameter
    """
    candidates = []
    if hasattr(scene, "get_depthmaps") and callable(scene.get_depthmaps):
        candidates.append(scene.get_depthmaps)
    if hasattr(scene, "get_im_depthmaps") and callable(scene.get_im_depthmaps):
        candidates.append(scene.get_im_depthmaps)

    for fn in candidates:
        d = fn()
        # Case 1: tensor already
        if torch.is_tensor(d):
            d = d.detach().cpu()
            if d.ndim == 3:
                return d.to(torch.float32)
            if d.ndim == 2:
                return d.unsqueeze(0).to(torch.float32)
            # Sometimes [N,1,H,W]
            if d.ndim == 4 and d.shape[1] == 1:
                return d[:, 0].to(torch.float32)
        # Case 2: list/tuple
        if isinstance(d, (list, tuple)):
            return _stack_any_depth_list(d)

    # Attribute variant
    if hasattr(scene, "im_depthmaps"):
        d = scene.im_depthmaps
        if torch.is_tensor(d):
            d = d.detach().cpu()
            if d.ndim == 3:
                return d.to(torch.float32)
            if d.ndim == 2:
                return d.unsqueeze(0).to(torch.float32)
            if d.ndim == 4 and d.shape[1] == 1:
                return d[:, 0].to(torch.float32)
        if isinstance(d, (list, tuple)):
            return _stack_any_depth_list(d)

    raise AttributeError("Could not obtain depthmaps: no get_depthmaps/get_im_depthmaps/im_depthmaps found.")


# -------------------------
# PLY export (from depthmaps)
# -------------------------
def _export_ply_from_depthmaps(scene, ply_path: str, ply_stride: int, max_points: int, notes: list) -> dict:
    import numpy as np
    import trimesh

    depth = _get_depthmaps_from_scene(scene)  # [N,H,W]
    poses = scene.get_im_poses().detach().cpu()  # [N,4,4]
    focals = scene.get_focals().detach().cpu()   # [N] or [N,2]

    # focals normalization
    if focals.ndim == 2:
        f = focals[:, 0]
    else:
        f = focals.reshape(-1)

    N, H, W = depth.shape
    stride = max(1, int(ply_stride))
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.reshape(-1)
    vv = vv.reshape(-1)

    pts_all = []
    col_all = []

    has_imgs = hasattr(scene, "imgs") and scene.imgs is not None
    imgs = scene.imgs if has_imgs else None

    total_pts = 0
    for i in range(min(N, poses.shape[0])):
        z = depth[i][::stride, ::stride].reshape(-1).numpy().astype(np.float32)

        valid = np.isfinite(z) & (z > 0)
        if not np.any(valid):
            continue

        fi = float(f[i].item()) if i < f.numel() else float(f[0].item())
        x = (uu - cx) / fi * z
        y = (vv - cy) / fi * z

        pts_cam = np.stack([x, y, z], axis=1)[valid]

        T = poses[i].numpy().astype(np.float32)  # c2w
        Rm = T[:3, :3]
        t = T[:3, 3]
        pts_w = (pts_cam @ Rm.T) + t

        pts_all.append(pts_w)
        total_pts += pts_w.shape[0]

        if has_imgs:
            im = imgs[i]
            if torch.is_tensor(im):
                im = im.detach().cpu().numpy()
            im_s = im[::stride, ::stride, :].reshape(-1, 3)
            col = (np.clip(im_s[valid], 0.0, 1.0) * 255.0).astype(np.uint8)
            col_all.append(col)

        if max_points > 0 and total_pts >= max_points:
            break

    if not pts_all:
        raise RuntimeError("No valid points exported from depthmaps (all invalid/zero).")

    pts = np.concatenate(pts_all, axis=0)

    # cap points
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        if col_all:
            col = np.concatenate(col_all, axis=0)[idx]
        else:
            col = None
        notes.append(f"PLY subsampled to max_points={max_points}.")
    else:
        col = np.concatenate(col_all, axis=0) if col_all else None

    if col is not None:
        pc = trimesh.PointCloud(vertices=pts, colors=col)
        color_mode = "rgb"
    else:
        pc = trimesh.PointCloud(vertices=pts)
        color_mode = "none"

    pc.export(ply_path)

    return {
        "export_method": "fallback_depthmaps_backprojection",
        "ply_path": ply_path,
        "ply_stride": int(ply_stride),
        "max_points": int(max_points),
        "points_exported": int(pts.shape[0]),
        "color_mode": color_mode,
        "depth_shape": [int(N), int(H), int(W)],
    }


def save_reconstruction(scene, out_dir: str, notes: list, ply_stride: int, max_points: int) -> dict:
    ply_path = os.path.join(out_dir, "reconstruction.ply")
    print(f"--> [Output] Saving PLY: {ply_path}")

    if hasattr(scene, "save_ply") and callable(getattr(scene, "save_ply")):
        scene.save_ply(ply_path)
        return {"export_method": "scene.save_ply", "ply_path": ply_path}

    msg = "Scene object has no save_ply(); exporting pointcloud from depthmaps + poses + focals."
    print(f"!! [Warn] {msg}")
    notes.append(msg)

    return _export_ply_from_depthmaps(scene, ply_path, ply_stride=ply_stride, max_points=max_points, notes=notes)


# -------------------------
# Main
# -------------------------
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

    t_infer0 = time.time()
    print(f"--> [Inference] Processing {len(pairs)} pairs (Batch Size: {args.batch_size})...")
    dust3r_output = inference(pairs, model, device, batch_size=args.batch_size)
    t_infer1 = time.time()

    t_align0 = time.time()
    scene = run_alignment(dust3r_output, device=device, niter=args.niter)
    t_align1 = time.time()

    print("--> [Output] Saving files...")
    export_meta = save_reconstruction(
        scene,
        args.output_dir,
        notes=notes,
        ply_stride=args.ply_stride,
        max_points=args.max_points,
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
    parser.add_argument("--model_name", type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (pairs per forward pass)")
    parser.add_argument("--niter", type=int, default=300, help="Alignment iterations")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size for load_images()")
    parser.add_argument("--scene_graph", type=str, default="complete", help="complete, swin-3, swin-5, logwin-4, ...")
    parser.add_argument("--symmetrize", action="store_true", default=True, help="Symmetrize pairs (demo default).")

    parser.add_argument("--ply_stride", type=int, default=4, help="Export every Nth pixel (bigger = smaller PLY).")
    parser.add_argument("--max_points", type=int, default=5_000_000, help="Cap points in PLY (0 = no cap).")

    args = parser.parse_args()
    main(args)
