import torch
import argparse
import os
import glob
import numpy as np
import math
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# --- Configuration ---
# Defaults are now handled via argparse, but keeping these as fallbacks or reference
DEFAULT_BATCH_SIZE = 60
DEFAULT_CHUNK_SIZE = 60

def save_colored_ply(points, colors, filename):
    print(f"Saving {len(points)} points to {filename}...")
    
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        # Stack points and colors horizontally
        # Colors are 0-1 float from model, convert to 0-255 uint8
        colors_uint8 = (colors * 255).astype(np.uint8)
        data = np.hstack((points, colors_uint8))
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")
    
    print(f"Success! Saved to {os.path.abspath(filename)}")

def process_chunk(model, image_files, device, dtype, output_filename, batch_size):
    print(f"Processing chunk of {len(image_files)} images -> {output_filename}")
    
    all_points = []
    all_colors = []

    # Process in sub-batches to avoid OOM
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        print(f"  Processing batch {i}-{i+len(batch_files)} / {len(image_files)}...")
        
        images = load_and_preprocess_images(batch_files).to(device).to(dtype)
        
        with torch.no_grad():
            # VGGT inference
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                preds = model(images)
            
            # Extract point clouds and colors
            if 'world_points' not in preds:
                print(f"Warning: 'world_points' not found in predictions. Keys: {preds.keys()}")
                continue

            # Shape: [1, S, H, W, 3] (assuming batch size 1 from model internal unsqueeze)
            points = preds['world_points'] 
            
            # Shape: [1, S, 3, H, W] -> [1, S, H, W, 3]
            colors = preds['images'].permute(0, 1, 3, 4, 2)

            # Flatten to [N_points, 3]
            points = points.reshape(-1, 3).float().cpu().numpy()
            colors = colors.reshape(-1, 3).float().cpu().numpy()

            # Filter out points at origin or invalid if needed (optional)
            # For now, we keep everything as per original demo logic
            
            all_points.append(points)
            all_colors.append(colors)

    if not all_points:
        print("No points generated for this chunk.")
        return

    final_points = np.concatenate(all_points, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    
    save_colored_ply(final_points, final_colors, output_filename)

def main():
    parser = argparse.ArgumentParser(description="Run VGGT Inference in Chunks")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_dir", type=str, default="/workspace/output/vggt", help="Directory to save output PLY files")
    parser.add_argument("--name", type=str, default="pointcloud", help="Base name for the output files (e.g., 'powerplant')")
    parser.add_argument("--chunk_size", type=int, default=60, help="Number of images per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="Number of overlapping images between consecutive chunks")
    parser.add_argument("--batch_size", type=int, default=60, help="Batch size for GPU processing")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Loading VGGT-1B to {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(args.image_path, ext)))
    file_list = sorted(file_list)
    
    total_images = len(file_list)
    print(f"Found {total_images} images in {args.image_path}")

    if total_images == 0:
        print("No images found!")
        return

    # Calculate chunks with overlap
    stride = args.chunk_size - args.overlap
    if stride <= 0:
        raise ValueError(f"Overlap ({args.overlap}) must be smaller than chunk size ({args.chunk_size})")

    chunks = []
    start_idx = 0
    while start_idx < total_images:
        end_idx = min(start_idx + args.chunk_size, total_images)
        chunks.append((start_idx, end_idx))
        
        if end_idx == total_images:
            break
            
        start_idx += stride

    num_chunks = len(chunks)
    print(f"Splitting into {num_chunks} parts (max {args.chunk_size} images, overlap {args.overlap}).")

    for i, (start_idx, end_idx) in enumerate(chunks):
        chunk_files = file_list[start_idx:end_idx]
        part_num = i + 1
        output_filename = os.path.join(args.output_dir, f"{args.name}_part_{part_num}.ply")
        
        print(f"\n--- Starting Part {part_num}/{num_chunks} (Images {start_idx} to {end_idx}) ---")
        process_chunk(model, chunk_files, device, dtype, output_filename, args.batch_size)

    print("\nAll parts processed successfully!")

if __name__ == "__main__":
    main()
