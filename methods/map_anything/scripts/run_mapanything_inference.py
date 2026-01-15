#!/usr/bin/env python3
# Example usage (your dataset):
#
#   python3 /workspace/scripts/run_mapanything_inference.py \
#     --input_dir /workspace/datasets/biomasse_2/rosbags/timestamp_filtered_my_run_20260113_163102_extracted/rgb_every_10_subset_first_50/ \
#     --output_dir /workspace/output/mapanything/biomasse_every_10_subset_first_50 \
#     --device cuda \
#     --resize_long 1024 \
#     --amp
#
# Notes:
# - MapAnything’s public repo is primarily Gradio-first; the exact Python/CLI inference API
#   can differ by commit. This script is robust:
#     1) Try in-process Python inference if a predictor API is discoverable
#     2) Otherwise, try to call a repo CLI script (infer.py / inference.py / demo.py) if present
# - Output is always:
#     output_dir/
#       run_info.json
#       manifest.json
#       <image_stem>/input_preview.jpg
#       <image_stem>/result.json (python backend) OR cli_run.json (cli backend)
#
# If you want specific artifacts (masks/depth/etc) written as PNGs, tell me which ones
# MapAnything produces in your checkout and I’ll wire them explicitly.

from __future__ import annotations

import os
import sys
import re
import json
import time
import glob
import shutil
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2

# ------------------------------------------------------------
# 1) ENV / PATH RESOLUTION (matches your container)
# ------------------------------------------------------------
MAPANYTHING_ROOT = os.environ.get("MAPANYTHING_ROOT", "/tmp_build/map-anything")

if os.path.isdir(MAPANYTHING_ROOT) and MAPANYTHING_ROOT not in sys.path:
    sys.path.insert(0, MAPANYTHING_ROOT)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ------------------------------------------------------------
# 2) UTILITIES
# ------------------------------------------------------------
def list_images(input_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def extract_timestamp_from_name(path: str) -> float:
    nums = re.findall(r"\d+", os.path.basename(path))
    if not nums:
        return 0.0
    ts = max(nums, key=len)
    try:
        return float(ts)
    except Exception:
        return 0.0


def imread_rgb(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_side(rgb, long_side: Optional[int]):
    if not long_side:
        return rgb
    h, w = rgb.shape[:2]
    cur = max(h, w)
    if cur <= long_side:
        return rgb
    scale = long_side / float(cur)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def ensure_empty_dir(path: str, clobber: bool):
    if os.path.exists(path):
        if not clobber:
            raise RuntimeError(
                f"Output dir exists: {path}\nUse --clobber to overwrite."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def run(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out_lines: List[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        out_lines.append(line)
    rc = p.wait()
    return rc, "".join(out_lines)


# ------------------------------------------------------------
# 3) BACKEND DISCOVERY / EXECUTION
# ------------------------------------------------------------
@dataclass
class Backend:
    name: str
    kind: str  # "python" or "cli"
    detail: str


def discover_backends() -> List[Backend]:
    backends: List[Backend] = []

    # A) In-process python modules (commit-dependent)
    python_candidates = [
        "map_anything",
        "map_anything.inference",
        "map_anything.predictor",
        "map_anything.api",
        "map_anything.demo",
    ]
    for mod in python_candidates:
        try:
            __import__(mod)
            backends.append(Backend(name=f"py:{mod}", kind="python", detail=f"import {mod}"))
        except Exception:
            pass

    # B) Repo CLI scripts (commit-dependent)
    scripts_dir = os.path.join(MAPANYTHING_ROOT, "scripts")
    for fname in ("infer.py", "inference.py", "demo.py"):
        p = os.path.join(scripts_dir, fname)
        if os.path.isfile(p):
            backends.append(Backend(name=f"cli:{fname}", kind="cli", detail=p))

    # Info-only: gradio exists but we don't use it for batch
    if os.path.isfile(os.path.join(scripts_dir, "gradio_app.py")):
        backends.append(Backend(name="info:gradio_app", kind="cli", detail="scripts/gradio_app.py (interactive)"))

    return backends


def _to_jsonable(x):
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if np is not None and isinstance(x, np.ndarray):
        if x.size <= 2048:
            return x.tolist()
        return {"_type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
    if torch is not None and isinstance(x, torch.Tensor):
        if x.numel() <= 2048:
            return x.detach().cpu().tolist()
        return {"_type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype)}
    return {"_type": "unknown", "repr": repr(x)}


def try_inprocess_infer(rgb, device: str, amp: bool) -> Dict:
    import torch

    probes = [
        ("map_anything", ["Predictor", "MapAnythingPredictor", "predict", "infer", "run_inference"]),
        ("map_anything.inference", ["Predictor", "predict", "infer", "run_inference"]),
        ("map_anything.predictor", ["Predictor", "MapAnythingPredictor", "predict"]),
        ("map_anything.api", ["predict", "infer"]),
        ("map_anything.demo", ["predict", "infer"]),
    ]

    last_err = None
    for modname, attrs in probes:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception as e:
            last_err = e
            continue

        for attr in attrs:
            if not hasattr(mod, attr):
                continue
            obj = getattr(mod, attr)

            try:
                if isinstance(obj, type):
                    init_vars = getattr(obj.__init__, "__code__", None)
                    if init_vars and "device" in init_vars.co_varnames:
                        predictor = obj(device=device)
                    else:
                        predictor = obj()
                    fn = predictor.predict if hasattr(predictor, "predict") else predictor
                else:
                    fn = obj

                with torch.inference_mode():
                    if amp and device.startswith("cuda"):
                        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                    else:
                        class _NullCtx:
                            def __enter__(self): return None
                            def __exit__(self, *a): return False
                        ctx = _NullCtx()

                    with ctx:
                        try:
                            out = fn(rgb, device=device)
                        except TypeError:
                            try:
                                out = fn(rgb)
                            except TypeError:
                                out = fn(image=rgb)

                return {
                    "backend": f"py:{modname}.{attr}",
                    "raw_type": str(type(out)),
                    "result": _to_jsonable(out),
                }
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(
        "No stable in-process inference API found in this MapAnything checkout.\n"
        f"Last error: {last_err}"
    )


def try_cli_infer_single(image_path: str, out_dir: str, device: str, amp: bool) -> Dict:
    scripts_dir = os.path.join(MAPANYTHING_ROOT, "scripts")
    candidates = [os.path.join(scripts_dir, f) for f in ("infer.py", "inference.py", "demo.py")]
    candidates = [c for c in candidates if os.path.isfile(c)]

    if not candidates:
        raise RuntimeError(f"No CLI inference scripts found under: {scripts_dir}")

    attempts: List[Tuple[List[str], str]] = []
    logs: List[str] = []

    for script in candidates:
        # Common-ish argument layouts
        attempts.append((
            ["python3", script, "--input", image_path, "--output", out_dir, "--device", device] + (["--amp"] if amp else []),
            f"{os.path.basename(script)} --input/--output"
        ))
        attempts.append((
            ["python3", script, "--image", image_path, "--outdir", out_dir, "--device", device] + (["--amp"] if amp else []),
            f"{os.path.basename(script)} --image/--outdir"
        ))
        attempts.append((
            ["python3", script, image_path, "--outdir", out_dir, "--device", device] + (["--amp"] if amp else []),
            f"{os.path.basename(script)} positional + --outdir"
        ))

    for cmd, label in attempts:
        rc, out = run(cmd, cwd=MAPANYTHING_ROOT)
        logs.append(f"\n--- Attempt: {label}\nCMD: {' '.join(cmd)}\nRC: {rc}\n{out}")
        if rc == 0:
            return {"backend": f"cli:{label}", "cmd": cmd, "log_tail": out[-2000:]}

    raise RuntimeError("All CLI attempts failed.\n" + "\n".join(logs[-3:]))


# ------------------------------------------------------------
# 4) MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MapAnything batch inference (folder -> outputs + manifest)")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--resize_long", type=int, default=0, help="If >0: downscale so long side <= this")
    parser.add_argument("--amp", action="store_true", help="Use autocast on CUDA to reduce memory")
    parser.add_argument("--clobber", action="store_true", help="Delete output_dir if it exists")
    parser.add_argument("--prefer", type=str, default="python", choices=["python", "cli"], help="Preferred backend")
    args = parser.parse_args()

    ensure_empty_dir(args.output_dir, args.clobber)

    images = list_images(args.input_dir)
    if len(images) < 1:
        raise SystemExit(f"No images found in {args.input_dir}")

    backends = discover_backends()

    run_info = {
        "time_unix": time.time(),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "device": args.device,
        "resize_long": args.resize_long,
        "amp": bool(args.amp),
        "mapanything_root": MAPANYTHING_ROOT,
        "discovered_backends": [b.__dict__ for b in backends],
    }
    with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    manifest: List[Dict] = []
    failures: List[Dict] = []

    use_python_first = (args.prefer == "python")

    for idx, path in enumerate(images):
        stem = os.path.splitext(os.path.basename(path))[0]
        item_out = os.path.join(args.output_dir, stem)
        os.makedirs(item_out, exist_ok=True)

        rec = {
            "index": idx,
            "path": path,
            "timestamp": extract_timestamp_from_name(path),
            "output_dir": item_out,
        }

        try:
            if use_python_first:
                rgb = imread_rgb(path)
                rgb = resize_long_side(rgb, args.resize_long if args.resize_long > 0 else None)

                # save preview always
                preview_path = os.path.join(item_out, "input_preview.jpg")
                cv2.imwrite(preview_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                result = try_inprocess_infer(rgb, device=args.device, amp=args.amp)
                with open(os.path.join(item_out, "result.json"), "w") as f:
                    json.dump(result, f, indent=2)

                rec.update({"ok": True, "backend": result.get("backend", "python"), "result": "result.json"})
            else:
                result = try_cli_infer_single(path, item_out, device=args.device, amp=args.amp)
                with open(os.path.join(item_out, "cli_run.json"), "w") as f:
                    json.dump(result, f, indent=2)

                rec.update({"ok": True, "backend": result.get("backend", "cli"), "cli_run": "cli_run.json"})

        except Exception as e:
            rec.update({"ok": False, "error": str(e)})
            failures.append(rec)

        manifest.append(rec)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(images):
            print(f"[{idx+1}/{len(images)}] processed")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    ok = sum(1 for m in manifest if m.get("ok"))
    bad = len(manifest) - ok
    print(f"\nDone. OK={ok} FAIL={bad}")
    print(f"Manifest: {os.path.join(args.output_dir, 'manifest.json')}")

    if bad:
        with open(os.path.join(args.output_dir, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)
        print(f"Failures: {os.path.join(args.output_dir, 'failures.json')}")


if __name__ == "__main__":
    main()
