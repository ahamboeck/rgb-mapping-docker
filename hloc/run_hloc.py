import argparse
from pathlib import Path
import sys
import numpy as np

# Added pairs_from_exhaustive to the imports
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive

try:
    import pycolmap
    print(f"PyCOLMAP found: {pycolmap.__version__}")
except ImportError:
    print("WARNING: PyCOLMAP not found. Geometric verification might fail or fall back to OpenCV.")

def main():
    parser = argparse.ArgumentParser(description="Run HLOC AI Matching for COLMAP")
    
    # Input/Output Arguments
    parser.add_argument("--data", type=Path, required=True, 
                        help="Path to the dataset root (contains 'images/' folder)")
    parser.add_argument("--output", type=Path, default=None, 
                        help="Path to save outputs (defaults to data/hloc_out)")
    
    # Model Arguments
    parser.add_argument("--feature", type=str, default="superpoint_max",
                        help="Feature extraction config (default: superpoint_max)")
    parser.add_argument("--matcher", type=str, default="superglue",
                        help="Matcher config (default: superglue)")
    
    # Camera Argument
    parser.add_argument("--camera_mode", type=str, default="SINGLE",
                        choices=["SINGLE", "PER_IMAGE", "PER_FOLDER"],
                        help="COLMAP camera mode (default: SINGLE for stability)")

    args = parser.parse_args()

    # Setup paths
    images = args.data / "images"
    outputs = args.output if args.output else args.data / "hloc_out"
    outputs.mkdir(parents=True, exist_ok=True)

    sfm_pairs = outputs / "pairs-exhaustive.txt"
    sfm_dir = outputs / "sfm"
    
    feature_conf = extract_features.confs[args.feature]
    matcher_conf = match_features.confs[args.matcher]

    # 1. Feature Extraction
    print(f"--- Extracting {args.feature} features ---")
    feature_path = extract_features.main(feature_conf, images, outputs)

    # 2. Pair Generation
    print("--- Generating Exhaustive Pairs ---")
    image_list = [p.relative_to(images).as_posix() for p in images.iterdir() if p.is_file()]
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)

    # 3. Feature Matching
    print(f"--- Matching with {args.matcher} ---")
    match_path = match_features.main(
        matcher_conf, 
        sfm_pairs, 
        feature_conf['output'], 
        outputs
    )

    # 4. Reconstruction (Database creation + Verification + Triangulation)
    print("--- Running Reconstruction (Database Creation & Verification) ---")
    sfm_dir.mkdir(exist_ok=True, parents=True)
    
    # reconstruction.main handles database creation, import, verification, and triangulation.
    # This ensures the database is correctly populated for COLMAP.
    try:
        model = reconstruction.main(
            sfm_dir, 
            images, 
            sfm_pairs, 
            feature_path, 
            match_path,
            camera_mode=args.camera_mode
        )
        print(f"--- Finished! Database created at: {sfm_dir / 'database.db'} ---")
    except Exception as e:
        print(f"ERROR during reconstruction: {e}")
        print("Ensure pycolmap is installed and the database is writable.")

if __name__ == "__main__":
    main()
