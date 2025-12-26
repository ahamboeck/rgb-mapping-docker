import argparse
import os
import subprocess
import sys
import cv2
import re
from pathlib import Path
import shutil

def natural_sort_key(s):
    """
    Key for natural sorting (e.g., 2.jpg comes before 10.jpg).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def process_images(input_path, output_path, target_size=(3840, 2160)):
    """
    Reads images from input_path, resizes to target_size, 
    and saves as PNG in output_path.
    Does NOT delete originals.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_files = []
    
    # Gather all images
    for ext in extensions:
        image_files.extend(list(input_path.glob(ext)))
    
    if not image_files:
        print(f"No images found in {input_path}")
        return False

    # Sort naturally
    image_files.sort(key=lambda p: natural_sort_key(p.name))

    print(f"Found {len(image_files)} images.")
    print(f"Processing to: {output_path}")
    print(f"Target Resolution: {target_size}")
    
    for img_path in image_files:
        try:
            # Construct output filename (force .png)
            # We preserve the stem (001.jpg -> 001.png)
            output_file = output_path / img_path.with_suffix(".png").name

            # Optimization: Skip if output already exists and is correct size
            if output_file.exists():
                # Optional: Check dimensions to be safe, or trust file existence
                # To be robust, we read it quickly or just skip
                # Let's skip to save time on restarts
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            h, w = img.shape[:2]
            
            # Resize logic
            if w != target_size[0] or h != target_size[1]:
                print(f"Resizing {img_path.name}: {w}x{h} -> {target_size[0]}x{target_size[1]}")
                interp = cv2.INTER_AREA if (w > target_size[0]) else cv2.INTER_LINEAR
                img = cv2.resize(img, target_size, interpolation=interp)
            else:
                # Even if size matches, we must ensure it is saved as PNG in the new folder
                pass

            cv2.imwrite(str(output_file), img)
            # No deletion here! Originals stay in input_path.

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False
            
    return True

def run_mast3r_slam(dataset_path, config_path="config/base.yaml", calib_path=None):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found at {dataset_path}")
        return

    # Determine paths
    original_images_path = dataset_path
    if os.path.exists(os.path.join(dataset_path, "images")):
        print("Found 'images' subdirectory, using that as source.")
        original_images_path = os.path.join(dataset_path, "images")

    # Create a separate folder for the processed images
    # e.g. /workspace/datasets/Kitchen/images -> /workspace/datasets/Kitchen/images_4k_resized
    processed_images_path = str(Path(original_images_path).parent / "images_4k_resized")
    
    print(f"Source: {original_images_path}")
    print(f"Dest  : {processed_images_path}")

    # Process images (Sort + Resize + Save to new folder)
    success = process_images(original_images_path, processed_images_path, target_size=(3840, 2160))
    
    if not success:
        print("Image processing failed or no images found. Aborting.")
        return

    print(f"Running MASt3R-SLAM on processed folder: {processed_images_path}...")

    cmd = [
        sys.executable, "main.py",
        "--dataset", processed_images_path,  # Pointing to the NEW folder
        "--config", config_path,
    ]
    
    if calib_path:
        if not os.path.exists(calib_path):
             print(f"Warning: Calibration file not found at {calib_path}. Running without explicit calibration.")
        else:
            cmd.extend(["--calib", calib_path])
    
    print("Executing:", " ".join(cmd))
    
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    
    # Setup Paths
    mast3r_path = os.path.abspath("thirdparty/mast3r")
    dust3r_path = os.path.abspath("thirdparty/mast3r/dust3r")
    in3d_path = os.path.abspath("thirdparty/in3d")
    
    paths_to_add = []
    if os.path.exists(mast3r_path): paths_to_add.append(mast3r_path)
    if os.path.exists(dust3r_path): paths_to_add.append(dust3r_path)
    if os.path.exists(in3d_path): paths_to_add.append(in3d_path)
        
    if paths_to_add:
        env["PYTHONPATH"] = os.pathsep.join(paths_to_add + [python_path])
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"MASt3R-SLAM failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MASt3R-SLAM safely (resizing to separate folder)")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--config", default="config/base.yaml", help="Path to config file")
    parser.add_argument("--calib", default=None, help="Path to calibration YAML")
    
    args = parser.parse_args()
    
    # Context switching to root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    if not os.path.exists("main.py"):
        if os.path.exists(os.path.join(root_dir, "main.py")):
            os.chdir(root_dir)
        elif os.path.exists(os.path.join(root_dir, "MASt3R-SLAM", "main.py")):
            os.chdir(os.path.join(root_dir, "MASt3R-SLAM"))

    run_mast3r_slam(args.dataset_path, args.config, args.calib)