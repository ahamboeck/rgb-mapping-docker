#!/usr/bin/env python3
"""
Example usage (inside your VGGT container):

python /workspace/scripts/run_vggt_inference.py \
  --input_dir  /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10_subset_first_50/ \
  --output_dir /workspace/output/vggt \
  --model_name facebook/VGGT-1B \
  --conf_thresh 0.2 \
  --max_points 2000000 \
  --extrinsic_is w2c
"""

import os
import re
import sys
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------
# 1) TIMESTAMP + IO UTILS
# ------------------------------------------------------------
def extract_timestamp_from_filename(path: str) -> float:
    matches = re.findall(r"\d+", Path(path).name)
    if not matches:
        raise ValueError(f"No digits found in filename for timestamp extraction: {path}")
    return float(max(matches, key=len))


def list_images_sorted(image_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return [str(p) for p in sorted(files, key=lambda p: p.name)]


def write_ply_xyzrgb_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """
    Minimal ASCII PLY writer for XYZ + RGB (uchar).
    xyz: (N,3) float
    rgb: (N,3) uint8
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] != xyz.shape[0]:
        raise ValueError(f"rgb must be (N,3) and match xyz, got {rgb.shape} vs {xyz.shape}")

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
    camera_poses_c2w: np.ndarray,   # (S,4,4)
    frame_paths: list[str],
) -> None:
    camera_poses_c2w = np.asarray(camera_poses_c2w, dtype=np.float64)
    if camera_poses_c2w.ndim != 3 or camera_poses_c2w.shape[1:] != (4, 4):
        raise ValueError(f"camera_poses_c2w must be (S,4,4), got {camera_poses_c2w.shape}")

    S = camera_poses_c2w.shape[0]
    if len(frame_paths) != S:
        raise ValueError(f"Need same number of frames and poses. frames={len(frame_paths)} poses={S}")

    lines = []
    for i in range(S):
        ts = extract_timestamp_from_filename(frame_paths[i])
        c2w = camera_poses_c2w[i]
        t = c2w[:3, 3]
        qx, qy, qz, qw = R.from_matrix(c2w[:3, :3]).as_quat()
        lines.append(
            f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )

    lines.sort(key=lambda s: float(s.split()[0]))
    with open(tum_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------
# 2) GEOMETRY / MERGE UTILS
# ------------------------------------------------------------
def extrinsic_3x4_to_c2w_4x4(extrinsic_3x4: np.ndarray, extrinsic_is: str) -> np.ndarray:
    E = np.asarray(extrinsic_3x4, dtype=np.float64)
    if E.ndim != 3 or E.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must be (S,3,4), got {E.shape}")

    S = E.shape[0]
    c2w = np.zeros((S, 4, 4), dtype=np.float64)
    c2w[:, 3, 3] = 1.0

    for i in range(S):
        R_ = E[i, :3, :3]
        t_ = E[i, :3, 3]

        if extrinsic_is == "c2w":
            c2w[i, :3, :3] = R_
            c2w[i, :3, 3] = t_
        elif extrinsic_is == "w2c":
            Rt = R_.T
            c2w[i, :3, :3] = Rt
            c2w[i, :3, 3] = -Rt @ t_
        else:
            raise ValueError("--extrinsic_is must be either 'w2c' or 'c2w'")

    return c2w


def merge_world_points_with_color_from_preds(
    world_points: np.ndarray,              # (S,H,W,3) or (1,S,H,W,3)
    preds_images: np.ndarray,              # (S,3,H,W) or (1,S,3,H,W)
    world_points_conf: np.ndarray | None,  # optional
    conf_thresh: float | None,
    conf_percentile: float | None,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    pts = np.asarray(world_points)
    if pts.ndim == 5 and pts.shape[0] == 1:
        pts = pts[0]
    if pts.ndim != 4 or pts.shape[-1] != 3:
        raise ValueError(f"world_points must be (S,H,W,3) (optionally with leading batch), got {pts.shape}")

    S, H, W, _ = pts.shape

    imgs = np.asarray(preds_images)
    if imgs.ndim == 5 and imgs.shape[0] == 1:
        imgs = imgs[0]  # (S,3,H,W)
    if imgs.ndim != 4 or imgs.shape != (S, 3, H, W):
        raise ValueError(f"preds['images'] must be (S,3,H,W) matching points; got {imgs.shape} vs {(S,3,H,W)}")

    # (S,3,H,W) -> (S,H,W,3) -> flatten. Values are usually in [0,1].
    rgb01 = np.transpose(imgs, (0, 2, 3, 1)).reshape(-1, 3).astype(np.float32)
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    pts_flat = pts.reshape(-1, 3).astype(np.float32)

    if world_points_conf is not None:
        conf = np.asarray(world_points_conf)
        # drop batch dim if present
        if conf.ndim == 4 and conf.shape[0] == 1:
            conf = conf[0]
        if conf.ndim == 5 and conf.shape[0] == 1:
            conf = conf[0]
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.shape != (S, H, W):
            raise ValueError(f"world_points_conf must be (S,H,W) (optionally with batch/channel), got {conf.shape}")

        conf_flat = conf.reshape(-1).astype(np.float32)

        if conf_percentile is not None:
            p = float(conf_percentile)
            if not (0.0 <= p < 100.0):
                raise ValueError("--conf_percentile must be in [0,100)")
            finite = np.isfinite(conf_flat)
            thr = np.percentile(conf_flat[finite], p) if finite.any() else -np.inf
            keep = conf_flat >= thr
        elif conf_thresh is not None:
            keep = conf_flat >= float(conf_thresh)
        else:
            keep = np.ones_like(conf_flat, dtype=bool)

        pts_flat = pts_flat[keep]
        rgb_u8 = rgb_u8[keep]

    good = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[good]
    rgb_u8 = rgb_u8[good]

    if max_points is not None and pts_flat.shape[0] > max_points:
        idx = rng.choice(pts_flat.shape[0], size=max_points, replace=False)
        pts_flat = pts_flat[idx]
        rgb_u8 = rgb_u8[idx]

    return pts_flat, rgb_u8


# ------------------------------------------------------------
# 3) MAIN
# ------------------------------------------------------------
def main(args):
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pose_decode_ok = True
    except Exception as e:
        print(f"!! Pose decoder import failed (will skip TUM/poses if needed): {e}")
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        pose_decode_ok = False
        pose_encoding_to_extri_intri = None  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input_dir must be a directory of images: {input_dir}")

    frame_paths = list_images_sorted(input_dir)
    if len(frame_paths) < 2:
        raise RuntimeError("Need at least 2 images.")

    # Match your working chunk script dtype selection
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    print(f"--> [Device] {device} | autocast dtype={dtype}")
    print(f"--> [Input] {len(frame_paths)} images from: {input_dir}")

    # Same approach as working chunk script: preprocess -> to(device)->to(dtype)
    images = load_and_preprocess_images(frame_paths).to(device).to(dtype)

    # Infer S,H,W robustly for pose decoding (if available)
    if images.ndim == 5:
        if images.shape[0] != 1:
            raise RuntimeError(f"Unexpected batch size in images: {images.shape[0]}")
        S = int(images.shape[1])
        H, W = int(images.shape[-2]), int(images.shape[-1])
    elif images.ndim == 4:
        S = int(images.shape[0])
        H, W = int(images.shape[-2]), int(images.shape[-1])
    else:
        raise RuntimeError(f"Unexpected images tensor shape: {tuple(images.shape)}")

    if S != len(frame_paths):
        raise RuntimeError(f"Preprocess produced S={S} but you have {len(frame_paths)} filenames.")

    print(f"--> [Model] Loading: {args.model_name}")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()

    print("--> [Inference] Running VGGT forward pass...")
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                preds = model(images)
        else:
            preds = model(images)

    if "world_points" not in preds:
        raise RuntimeError(f"preds missing 'world_points'. Available keys: {list(preds.keys())}")
    if "images" not in preds:
        raise RuntimeError(f"preds missing 'images' (needed for colors). Available keys: {list(preds.keys())}")

    world_points = preds["world_points"]
    preds_images = preds["images"]
    world_points_conf = preds.get("world_points_conf", None)

    extrinsic_np = None
    intrinsic_np = None
    camera_poses_c2w = None

    if pose_decode_ok and ("pose_enc" in preds):
        pose_enc = preds["pose_enc"]
        print("--> [Cameras] Decoding pose_enc -> extrinsic/intrinsic ...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))  # type: ignore

        def _to_numpy_f32(x: torch.Tensor) -> np.ndarray:
            # important: cast bf16/fp16 to fp32 before numpy()
            x = x.detach()
            if x.ndim >= 1 and x.shape[0] == 1:
                x = x.squeeze(0)
            return x.float().cpu().numpy()

        extrinsic_np = _to_numpy_f32(extrinsic)   # (S,3,4)
        intrinsic_np = _to_numpy_f32(intrinsic)   # (S,3,3)

        if extrinsic_np.shape[0] != S:
            raise RuntimeError(f"extrinsic S mismatch: extrinsic={extrinsic_np.shape[0]} images={S}")

        camera_poses_c2w = extrinsic_3x4_to_c2w_4x4(extrinsic_np, extrinsic_is=args.extrinsic_is)
    else:
        if not pose_decode_ok:
            print("!! [Warn] pose decoding not available; skipping extrinsic/intrinsic and TUM export.")
        elif "pose_enc" not in preds:
            print("!! [Warn] preds has no 'pose_enc'; skipping extrinsic/intrinsic and TUM export.")

    print("--> [PointCloud] Merging colored world_points (colors from preds['images']) ...")

    # IMPORTANT FIX: bf16 cannot be converted to numpy directly -> cast to float32 first
    pts_np = world_points.detach().float().cpu().numpy()
    imgs_np = preds_images.detach().float().cpu().numpy()
    conf_np = None
    if world_points_conf is not None and isinstance(world_points_conf, torch.Tensor):
        conf_np = world_points_conf.detach().float().cpu().numpy()

    pts, rgb_u8 = merge_world_points_with_color_from_preds(
        world_points=pts_np,
        preds_images=imgs_np,
        world_points_conf=conf_np,
        conf_thresh=None if args.conf_percentile is not None else args.conf_thresh,
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    ply_path = out_dir / "reconstruction_colored.ply"
    print(f"--> [Output] Writing PLY: {ply_path} (N={pts.shape[0]})")
    write_ply_xyzrgb_ascii(str(ply_path), pts, rgb_u8)

    if camera_poses_c2w is not None:
        tum_path = out_dir / "trajectory_tum.txt"
        print(f"--> [Output] Writing TUM: {tum_path}")
        export_tum_trajectory(str(tum_path), camera_poses_c2w, frame_paths)

        np.save(out_dir / "extrinsic.npy", extrinsic_np)
        np.save(out_dir / "intrinsic.npy", intrinsic_np)
        np.save(out_dir / "camera_poses_c2w.npy", camera_poses_c2w)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    print(f"\n--> [Success] Done. Output saved to: {out_dir}")


# ------------------------------------------------------------
# 4) CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("VGGT -> colored PLY + (optional) TUM trajectory (timestamped filenames)")

    ap.add_argument("--input_dir", required=True, help="Folder of timestamped images (digits in filenames)")
    ap.add_argument("--output_dir", required=True, help="Where to write outputs")
    ap.add_argument("--model_name", default="facebook/VGGT-1B", help="HuggingFace model id")

    ap.add_argument("--conf_thresh", type=float, default=0.2,
                    help="Keep points with conf >= thresh (ignored if --conf_percentile is set)")
    ap.add_argument("--conf_percentile", type=float, default=None,
                    help="Drop the lowest X%% confidence points (e.g. 25 drops lowest 25%%)")

    ap.add_argument("--max_points", type=int, default=2_000_000,
                    help="Randomly subsample merged cloud to this many points (0 disables)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for point subsampling")

    ap.add_argument(
        "--extrinsic_is",
        choices=["w2c", "c2w"],
        default="w2c",
        help="How to interpret VGGT 'extrinsic' from pose decoding. We convert to c2w for TUM.",
    )

    args = ap.parse_args()
    if args.max_points == 0:
        args.max_points = None

    main(args)
