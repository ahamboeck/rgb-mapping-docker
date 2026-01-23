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


def get_rotation_matrix_from_pycolmap(rotation_obj) -> np.ndarray:
    """
    Robustly obtain a 3x3 rotation matrix from pycolmap rotation object across versions.
    """
    # Common patterns across pycolmap versions / builds:
    # - rotation.matrix() method
    # - rotation.R() method
    # - rotation.matrix property
    for attr in ("matrix", "R"):
        if hasattr(rotation_obj, attr):
            val = getattr(rotation_obj, attr)
            if callable(val):
                M = val()
            else:
                M = val
            M = np.asarray(M, dtype=float)
            if M.shape == (3, 3):
                return M

    raise AttributeError("Could not retrieve 3x3 rotation matrix from pose.rotation (no matrix()/R()).")


def rotmat_to_quat_xyzw(Rm: np.ndarray) -> np.ndarray:
    """
    Convert a proper rotation matrix to quaternion in (x, y, z, w) order.
    Numerically stable enough for typical SfM/SLAM outputs.
    """
    Rm = np.asarray(Rm, dtype=float)
    if Rm.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {Rm.shape}")

    tr = np.trace(Rm)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0  # S = 4*w
        w = 0.25 * S
        x = (Rm[2, 1] - Rm[1, 2]) / S
        y = (Rm[0, 2] - Rm[2, 0]) / S
        z = (Rm[1, 0] - Rm[0, 1]) / S
    else:
        # Find the largest diagonal element and proceed accordingly
        if (Rm[0, 0] > Rm[1, 1]) and (Rm[0, 0] > Rm[2, 2]):
            S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2.0  # S = 4*x
            w = (Rm[2, 1] - Rm[1, 2]) / S
            x = 0.25 * S
            y = (Rm[0, 1] + Rm[1, 0]) / S
            z = (Rm[0, 2] + Rm[2, 0]) / S
        elif Rm[1, 1] > Rm[2, 2]:
            S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2.0  # S = 4*y
            w = (Rm[0, 2] - Rm[2, 0]) / S
            x = (Rm[0, 1] + Rm[1, 0]) / S
            y = 0.25 * S
            z = (Rm[1, 2] + Rm[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2.0  # S = 4*z
            w = (Rm[1, 0] - Rm[0, 1]) / S
            x = (Rm[0, 2] + Rm[2, 0]) / S
            y = (Rm[1, 2] + Rm[2, 1]) / S
            z = 0.25 * S

    q = np.array([x, y, z, w], dtype=float)
    # Normalize
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Quaternion conversion produced zero norm.")
    q /= n
    return q


def convert_colmap_to_tum(
    model_path: str,
    output_name: str = "colmap_camera_poses.tum",
    timestamp_regex: str = r"(\d+\.?\d*)",
    timestamp_scale: float = 1.0,
    export_txt: bool = False,
    debug: bool = False,
):
    model_path = os.path.abspath(model_path)

    recon = pycolmap.Reconstruction(model_path)

    if export_txt:
        recon.write_text(model_path)
        print(f"-> Exported .txt model to {model_path}")

    items = list(recon.images.items())

    def key_fn(kv):
        _, img = kv
        return extract_timestamp(img.name, timestamp_regex, timestamp_scale)

    items.sort(key=key_fn)

    lines = []
    for img_id, img in items:
        ts = extract_timestamp(img.name, timestamp_regex, timestamp_scale)

        # Camera center in world coordinates
        Cw = np.asarray(img.projection_center(), dtype=float)

        # Get pose: cam_from_world (world->cam)
        pose = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world

        # Rotation world->cam as matrix
        R_cw = get_rotation_matrix_from_pycolmap(pose.rotation)

        # Invert rotation to get cam->world
        R_wc = R_cw.T

        # Convert to quaternion (x,y,z,w)
        qx, qy, qz, qw = rotmat_to_quat_xyzw(R_wc).tolist()

        if debug and len(lines) < 5:
            # Print first few quaternions and implied angle for sanity
            angle_deg = 2.0 * np.degrees(np.arccos(min(1.0, abs(qw))))
            print(f"[debug] {img.name} qw={qw:+.3f} angle~{angle_deg:.1f}deg  Cw={Cw}")

        # TUM: timestamp tx ty tz qx qy qz qw
        lines.append(
            f"{ts:.9f} {Cw[0]:.6f} {Cw[1]:.6f} {Cw[2]:.6f} "
            f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}"
        )

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
    ap.add_argument("--debug", action="store_true", help="Print sanity debug for first few poses")
    args = ap.parse_args()

    convert_colmap_to_tum(
        args.path,
        output_name=args.out,
        timestamp_regex=args.timestamp_regex,
        timestamp_scale=args.timestamp_scale,
        export_txt=args.export_txt,
        debug=args.debug,
    )
