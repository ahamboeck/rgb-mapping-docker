import sys
import os
import torch
import glob
import argparse
import numpy as np
import gc

# Ensure we can find mast3r and dust3r
# In Docker, PYTHONPATH should be set, but we add fallback just in case
sys.path.append("/tmp_build/mast3r")

try:
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from mast3r.model import AsymmetricMASt3R
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: {e}")
    print("Ensure PYTHONPATH includes the mast3r repository root.")
    exit(1)

def save_ply(filepath, points, colors=None):
    print(f"Writing {len(points)} points to {filepath}...")
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is not None:
            if colors.max() <= 1.0: colors = (colors * 255).astype(np.uint8)
            else: colors = colors.astype(np.uint8)
            for i in range(len(points)):
                p = points[i]
                c = colors[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        else:
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, torch.Tensor):
            cpu_dict[k] = v.detach().cpu()
        elif isinstance(v, list):
            cpu_dict[k] = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            cpu_dict[k] = v
    return cpu_dict

def main():
    parser = argparse.ArgumentParser(description="Run MASt3R Hybrid (GPU Inference + CPU Alignment)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to images")
    parser.add_argument("--output_file", type=str, default="mast3r_result.ply", help="Output file")
    parser.add_argument("--image_size", type=int, default=512, help="Resolution (Default 512)")
    parser.add_argument("--window_size", type=int, default=5, help="Sliding window size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    print(f"Loading MASt3R Model: {model_name}...")

    # PyTorch 2.6+ Security Patch
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
    
    try:
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. Load Images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(args.image_path, ext)))
    file_list = sorted(file_list)
    
    print(f"Found {len(file_list)} images in {args.image_path}")
    if len(file_list) == 0: return

    # Load images
    images = load_images(file_list, size=args.image_size)

    # 2. Pairs
    pairs = []
    print(f"Creating pairs (Window: {args.window_size})...")
    for i in range(len(images)):
        for j in range(1, args.window_size + 1):
            if i + j < len(images):
                pairs.append((images[i], images[i+j]))
    print(f"Computed {len(pairs)} pairs.")

    if len(pairs) == 0:
        print("No pairs generated. Check window size or image count.")
        return

    # 3. Inference on GPU
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Running Inference on {device} (Batch: {args.batch_size})...")
    output_gpu = inference(pairs, model, device, batch_size=args.batch_size, verbose=True)

    # --- HYBRID STEP: MOVE TO CPU ---
    print("Moving data to CPU to prevent VRAM OOM...")
    output_cpu = dict_to_cpu(output_gpu)
    
    # Clear GPU
    del output_gpu
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Global Alignment on CPU
    print("Running Global Alignment on CPU (Safe Mode)...")
    
    scene = global_aligner(output_cpu, device='cpu', mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=500, schedule='cosine', lr=0.01)

    # 5. Extraction
    print("Extracting data...")
    pts3d_list = scene.get_pts3d()
    imgs_list = scene.imgs 
    
    # Handle confidence
    if hasattr(scene, 'im_conf'):
        conf_list = scene.im_conf
    else:
        conf_list = [torch.ones_like(p[..., 0]) * 2.0 for p in pts3d_list]

    all_pts, all_cols = [], []

    for i in range(len(pts3d_list)):
        pts = pts3d_list[i].detach().numpy().reshape(-1, 3)
        cols = imgs_list[i]
        if isinstance(cols, torch.Tensor): cols = cols.detach().numpy()
        cols = cols.reshape(-1, 3)
        conf = conf_list[i].detach().numpy().reshape(-1)

        # Filtering
        mask = conf > 1.0
        all_pts.append(pts[mask])
        all_cols.append(cols[mask])

    if not all_pts:
        print("No points survived filtering.")
        return

    pts3d = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    save_ply(args.output_file, pts3d, colors)
    print(f"Done! Saved to {args.output_file}")

if __name__ == '__main__':
    main()
