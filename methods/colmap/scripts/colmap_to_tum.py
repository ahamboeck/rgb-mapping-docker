#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import pycolmap


def extract_timestamp(name: str, regex: str, scale: float) -> float:
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.search(regex, stem)
    if not m:
        raise ValueError(f"Could not extract timestamp from '{stem}' with regex '{regex}'")
    return float(m.group(1)) * scale


def quat_conjugate_wxyz(qwxyz: np.ndarray) -> np.ndarray:
    # input [w, x, y, z]
    q = np.asarray(qwxyz, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_normalize_wxyz(qwxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(qwxyz, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return q / n


def convert_colmap_to_tum(model_path: str,
                          output_name: str = "colmap_camera_poses.tum",
                          timestamp_regex: str = r"(\d+\.?\d*)",
                          timestamp_scale: float = 1.0,
                          export_txt: bool = False):
    model_path = os.path.abspath(model_path)

    # Load the reconstruction
    recon = pycolmap.Reconstruction(model_path)

    if export_txt:
        recon.write_text(model_path)
        print(f"-> Exported .txt model to {model_path}")

    # Sort by timestamp extracted from filename (not lexicographic)
    items = list(recon.images.items())

    def key_fn(kv):
        _, img = kv
        return extract_timestamp(img.name, timestamp_regex, timestamp_scale)

    items.sort(key=key_fn)

    lines = []
    for img_id, img in items:
        ts = extract_timestamp(img.name, timestamp_regex, timestamp_scale)

        # camera center in world coords (consistent with TUM position)
        Cw = np.asarray(img.projection_center(), dtype=float)

        # world->cam rotation quaternion (w,x,y,z)
        # In pycolmap, cam_from_world can be method or property depending on version.
        pose = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        q_w2c = np.asarray(pose.rotation.quat, dtype=float)

        # invert rotation to get cam->world (c2w)
        q_c2w = quat_normalize_wxyz(quat_conjugate_wxyz(q_w2c))

        # TUM wants qx qy qz qw
        qw, qx, qy, qz = q_c2w.tolist()
        lines.append(f"{ts:.9f} {Cw[0]:.6f} {Cw[1]:.6f} {Cw[2]:.6f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}")

    out_path = os.path.join(model_path, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"-> Wrote TUM: {out_path} ({len(lines)} poses)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert COLMAP model to TUM trajectory using pycolmap.")
    ap.add_argument("path", help="Path to COLMAP sparse model folder (contains images.bin)")
    ap.add_argument("--out", default="colmap_camera_poses.tum", help="Output TUM filename (written into model folder)")
    ap.add_argument("--timestamp_regex", default=r"(\d+\.?\d*)",
                    help="Regex with one capture group for timestamp from filename stem")
    ap.add_argument("--timestamp_scale", type=float, default=1.0,
                    help="Scale factor for timestamps (e.g. 1e-9 if filenames are nanoseconds)")
    ap.add_argument("--export_txt", action="store_true", help="Also export model to text files for debugging")
    args = ap.parse_args()

    convert_colmap_to_tum(
        args.path,
        output_name=args.out,
        timestamp_regex=args.timestamp_regex,
        timestamp_scale=args.timestamp_scale,
        export_txt=args.export_txt,
    )
