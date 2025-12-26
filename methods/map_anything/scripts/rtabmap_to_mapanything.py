#!/usr/bin/env python3
"""
RTAB-Map to MapAnything Converter

This script converts RTAB-Map output (RGB images, depth images, poses, calibration)
to the format expected by MapAnything for 3D reconstruction.

RTAB-Map exports:
- RGB images in rgb/ folder (numbered with potential gaps: 1.jpg, 4.jpg, 5.jpg, ...)
- Depth images in depth/ folder (matching numbers: 1.png, 4.png, 5.png, ...)
- Poses in poses.txt (timestamp, x, y, z, qx, qy, qz, qw) - in camera frame
- Calibration in calibration.yaml (intrinsics, local_transform)

MapAnything expects:
- img: (H, W, 3) tensor with [0, 255] range
- intrinsics: (3, 3) matrix
- camera_poses: (4, 4) cam2world matrix in OpenCV convention (+X Right, +Y Down, +Z Forward)
- depth_z: (H, W) depth in meters (optional)
- is_metric_scale: boolean flag

Key considerations:
- Images are numbered non-sequentially but poses are sequential (1st pose = 1st image when sorted)
- The local_transform in calibration represents the camera orientation in body frame
- RTAB-Map poses are in camera frame (cam2world)
- Depth scale: RTAB-Map typically uses mm (uint16) or meters (float32)

Usage:
    python rtabmap_to_mapanything.py --rtabmap_path /path/to/rtabmap/output --stride 5

"""

import os
import sys
import argparse
import glob
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import cv2
import yaml
from PIL import Image

# Add map-anything to path for imports
MAP_ANYTHING_PATH = "/tmp_build/map-anything"
if os.path.exists(MAP_ANYTHING_PATH):
    sys.path.insert(0, MAP_ANYTHING_PATH)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix.
    
    Args:
        q: Quaternion as [qx, qy, qz, qw]
    
    Returns:
        3x3 rotation matrix
    """
    qx, qy, qz, qw = q
    
    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    return R


def read_rtabmap_calibration(calib_path: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Read RTAB-Map calibration.yaml file.
    
    Args:
        calib_path: Path to calibration.yaml
    
    Returns:
        Tuple of (width, height, intrinsics_3x3, local_transform_4x4)
    """
    with open(calib_path, 'r') as f:
        content = f.read().replace("%YAML:1.0", "")
        data = yaml.safe_load(content)
    
    # Parse local_transform (camera orientation in body frame)
    T_local = np.eye(4)
    if 'local_transform' in data:
        try:
            lt_data = data['local_transform']['data']
            lt_matrix = np.array(lt_data).reshape(3, 4)
            T_local[:3, :] = lt_matrix
            print(f"Found local_transform:\n{lt_matrix}")
        except Exception as e:
            print(f"Warning: Could not parse local_transform: {e}")
    
    # Extract intrinsics - prefer projection_matrix
    fx, fy, cx, cy = None, None, None, None
    width, height = None, None
    
    if 'projection_matrix' in data:
        pm = data['projection_matrix'].get('data', [])
        if len(pm) >= 12:
            fx = pm[0]
            cx = pm[2]
            fy = pm[5]
            cy = pm[6]
            width = data.get('image_width')
            height = data.get('image_height')
            print(f"Using projection_matrix: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    elif 'camera_matrix' in data:
        cm = data['camera_matrix']
        matrix_data = cm.get('data', [])
        if len(matrix_data) >= 9:
            fx = matrix_data[0]
            cx = matrix_data[2]
            fy = matrix_data[4]
            cy = matrix_data[5]
            # Handle weird format where data is not properly arranged
            if fy == 0 and len(matrix_data) > 6:
                fy = matrix_data[5]
                cy = matrix_data[6]
            width = data.get('image_width')
            height = data.get('image_height')
            print(f"Using camera_matrix: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    if fx is None:
        raise ValueError("Could not parse intrinsics from calibration file")
    
    # Build 3x3 intrinsics matrix
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return width, height, intrinsics, T_local


def read_rtabmap_poses(pose_path: str) -> List[np.ndarray]:
    """
    Read RTAB-Map poses.txt file.
    
    Format: timestamp x y z qx qy qz qw
    Poses are in camera frame (cam2world).
    
    Args:
        pose_path: Path to poses.txt
    
    Returns:
        List of 4x4 cam2world transformation matrices
    """
    poses = []
    
    with open(pose_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
            
            # Parse translation and quaternion
            # Format: timestamp x y z qx qy qz qw
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(q)
            
            # Build 4x4 cam2world transformation matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            
            poses.append(T)
    
    print(f"Loaded {len(poses)} poses from {pose_path}")
    return poses


def get_sorted_image_files(folder: str, extension: str = None) -> List[str]:
    """
    Get image files sorted numerically by their base filename.
    
    Handles non-sequential numbering (1.jpg, 4.jpg, 5.jpg, 7.jpg, ...)
    
    Args:
        folder: Path to image folder
        extension: File extension filter (e.g., '.jpg', '.png'). If None, accepts common formats.
    
    Returns:
        List of sorted file paths
    """
    if extension:
        pattern = os.path.join(folder, f"*{extension}")
        files = glob.glob(pattern)
    else:
        files = []
        for ext in ['.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG']:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    
    # Sort by numeric value of filename (handle non-sequential numbering)
    def get_numeric_key(filepath):
        basename = os.path.splitext(os.path.basename(filepath))[0]
        try:
            return int(basename)
        except ValueError:
            return float('inf')  # Non-numeric files go to the end
    
    sorted_files = sorted(files, key=get_numeric_key)
    return sorted_files


def find_matching_pairs(rgb_folder: str, depth_folder: str) -> List[Tuple[str, str]]:
    """
    Find matching RGB and depth image pairs based on filename numbers.
    
    Args:
        rgb_folder: Path to RGB images folder
        depth_folder: Path to depth images folder
    
    Returns:
        List of (rgb_path, depth_path) tuples, sorted by image number
    """
    # Get all files
    rgb_files = get_sorted_image_files(rgb_folder)
    depth_files = get_sorted_image_files(depth_folder)
    
    # Build lookup by base number
    def get_number(path):
        basename = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(basename)
        except ValueError:
            return None
    
    rgb_by_num = {get_number(f): f for f in rgb_files if get_number(f) is not None}
    depth_by_num = {get_number(f): f for f in depth_files if get_number(f) is not None}
    
    # Find matching numbers
    common_numbers = sorted(set(rgb_by_num.keys()) & set(depth_by_num.keys()))
    
    pairs = [(rgb_by_num[n], depth_by_num[n]) for n in common_numbers]
    print(f"Found {len(pairs)} matching RGB-depth pairs")
    
    return pairs


def load_rtabmap_data(
    rtabmap_path: str,
    stride: int = 1,
    max_frames: int = -1,
    use_depth: bool = True,
    verbose: bool = False
) -> Tuple[List[Dict], np.ndarray, Tuple[int, int]]:
    """
    Load RTAB-Map data and convert to MapAnything format.
    
    CRITICAL: Images are numbered with gaps (1, 4, 5, 7, ...) but poses are sequential.
    The 1st pose corresponds to the 1st image when sorted numerically,
    the 2nd pose to the 2nd image, etc.
    
    Args:
        rtabmap_path: Path to RTAB-Map output folder containing rgb/, depth/, poses.txt, calibration.yaml
        stride: Use every N-th frame (1 = all frames, 5 = every 5th frame)
        max_frames: Maximum number of frames to load (-1 for all)
        use_depth: Whether to include depth data
        verbose: Print detailed progress
    
    Returns:
        Tuple of (views_list, intrinsics_3x3, (width, height))
    """
    # Validate paths
    rgb_folder = os.path.join(rtabmap_path, "rgb")
    depth_folder = os.path.join(rtabmap_path, "depth")
    pose_file = os.path.join(rtabmap_path, "poses.txt")
    calib_file = os.path.join(rtabmap_path, "calibration.yaml")
    
    for path, name in [(rgb_folder, "rgb folder"), (pose_file, "poses.txt"), (calib_file, "calibration.yaml")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required {name} not found at: {path}")
    
    if use_depth and not os.path.exists(depth_folder):
        print(f"Warning: Depth folder not found at {depth_folder}, proceeding without depth")
        use_depth = False
    
    # Load calibration
    width, height, intrinsics, T_local = read_rtabmap_calibration(calib_file)
    print(f"Image resolution: {width}x{height}")
    print(f"Intrinsics:\n{intrinsics}")
    
    # Load poses
    poses = read_rtabmap_poses(pose_file)
    
    # Get sorted image files
    if use_depth:
        # Match RGB and depth by filename number
        pairs = find_matching_pairs(rgb_folder, depth_folder)
        rgb_files = [p[0] for p in pairs]
        depth_files = [p[1] for p in pairs]
    else:
        rgb_files = get_sorted_image_files(rgb_folder)
        depth_files = [None] * len(rgb_files)
    
    # Verify we have matching counts
    num_images = len(rgb_files)
    num_poses = len(poses)
    
    if num_images != num_poses:
        print(f"Warning: Mismatch between images ({num_images}) and poses ({num_poses})")
        # Use minimum to avoid index errors
        num_frames = min(num_images, num_poses)
    else:
        num_frames = num_images
    
    if max_frames > 0:
        num_frames = min(num_frames, max_frames * stride)  # Account for stride
    
    print(f"Processing {num_frames} frames with stride={stride}")
    
    # Build views list
    views = []
    frame_idx = 0
    
    for i in range(0, num_frames, stride):
        rgb_path = rgb_files[i]
        depth_path = depth_files[i] if use_depth else None
        pose = poses[i]
        
        if verbose:
            print(f"Loading frame {frame_idx}: {os.path.basename(rgb_path)}")
        
        # Load RGB image
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_array = np.array(rgb_img, dtype=np.uint8)  # (H, W, 3) [0, 255]
        
        # Verify resolution matches calibration
        actual_h, actual_w = rgb_array.shape[:2]
        if (actual_w, actual_h) != (width, height):
            print(f"Warning: Image {rgb_path} has size {actual_w}x{actual_h}, expected {width}x{height}")
            # Update intrinsics for different resolution
            scale_x = actual_w / width
            scale_y = actual_h / height
            scaled_intrinsics = intrinsics.copy()
            scaled_intrinsics[0, 0] *= scale_x  # fx
            scaled_intrinsics[0, 2] *= scale_x  # cx
            scaled_intrinsics[1, 1] *= scale_y  # fy
            scaled_intrinsics[1, 2] *= scale_y  # cy
            view_intrinsics = scaled_intrinsics
        else:
            view_intrinsics = intrinsics.copy()
        
        # Build view dictionary for MapAnything
        view = {
            "img": torch.from_numpy(rgb_array),  # (H, W, 3) [0, 255]
            "intrinsics": torch.from_numpy(view_intrinsics.astype(np.float32)),  # (3, 3)
            "camera_poses": torch.from_numpy(pose.astype(np.float32)),  # (4, 4) cam2world
            "is_metric_scale": torch.tensor([True]),  # RTAB-Map data is metric
        }
        
        # Load depth if available
        if depth_path is not None:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if depth_img is None:
                print(f"Warning: Could not load depth image: {depth_path}")
            else:
                # Determine depth scale
                if depth_img.dtype == np.float32:
                    # Already in meters
                    depth_meters = depth_img
                elif depth_img.dtype == np.uint16:
                    # Typically millimeters, convert to meters
                    depth_meters = depth_img.astype(np.float32) / 1000.0
                else:
                    # Assume uint8 or similar, scale appropriately
                    depth_meters = depth_img.astype(np.float32) / 1000.0
                
                view["depth_z"] = torch.from_numpy(depth_meters)  # (H, W)
        
        views.append(view)
        frame_idx += 1
        
        if max_frames > 0 and frame_idx >= max_frames:
            break
    
    print(f"Loaded {len(views)} views for MapAnything")
    return views, intrinsics, (width, height)


def run_mapanything_inference(
    views: List[Dict],
    output_dir: str,
    memory_efficient: bool = False,
    save_glb: bool = True,
    verbose: bool = False
) -> None:
    """
    Run MapAnything inference on the loaded views.
    
    Args:
        views: List of view dictionaries
        output_dir: Output directory for results
        memory_efficient: Use memory-efficient inference (slower but handles more views)
        save_glb: Save output as GLB file
        verbose: Print detailed progress
    """
    # Import MapAnything modules
    try:
        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs
        from mapanything.utils.geometry import depthmap_to_world_frame
        from mapanything.utils.viz import predictions_to_glb
    except ImportError as e:
        print(f"Error importing MapAnything modules: {e}")
        print("Make sure MapAnything is installed in the environment.")
        print("Expected path: /tmp_build/map-anything")
        return
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading MapAnything model...")
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    
    # Preprocess inputs
    print("Preprocessing inputs...")
    processed_views = preprocess_inputs(views, verbose=verbose)
    
    # Run inference
    print(f"Running MapAnything inference on {len(processed_views)} views...")
    outputs = model.infer(
        processed_views,
        memory_efficient_inference=memory_efficient,
        # Use provided inputs (depth and poses from RTAB-Map)
        ignore_calibration_inputs=False,  # Use RTAB-Map calibration
        ignore_depth_inputs=False,        # Use RTAB-Map depth
        ignore_pose_inputs=False,         # Use RTAB-Map poses
        ignore_depth_scale_inputs=False,  # Depth is metric
        ignore_pose_scale_inputs=False,   # Poses are metric
        # Inference settings
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("Inference complete!")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save results
    if save_glb:
        print("Preparing GLB export...")
        world_points_list = []
        images_list = []
        masks_list = []
        
        for view_idx, pred in enumerate(outputs):
            # Extract data from predictions
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
            intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
            camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
            
            # Compute pts3d using depth, intrinsics, and camera pose
            pts3d_computed, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsics_torch, camera_pose_torch
            )
            
            # Convert to numpy arrays
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            mask = mask & valid_mask.cpu().numpy()
            pts3d_np = pts3d_computed.cpu().numpy()
            image_np = pred["img_no_norm"][0].cpu().numpy()
            
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)
        
        # Stack all views
        world_points = np.stack(world_points_list, axis=0)
        images = np.stack(images_list, axis=0)
        final_masks = np.stack(masks_list, axis=0)
        
        # Create predictions dict for GLB export
        predictions = {
            "world_points": world_points,
            "images": images,
            "final_masks": final_masks,
        }
        
        # Convert to GLB scene
        scene_3d = predictions_to_glb(predictions, as_mesh=True)
        
        # Save GLB file
        glb_path = os.path.join(output_dir, "rtabmap_mapanything_output.glb")
        scene_3d.export(glb_path)
        print(f"Saved GLB file: {glb_path}")
    
    print(f"All outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RTAB-Map output to MapAnything format and run reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - process all frames
    python rtabmap_to_mapanything.py --rtabmap_path /path/to/rtabmap/output

    # Use every 5th frame
    python rtabmap_to_mapanything.py --rtabmap_path /path/to/rtabmap/output --stride 5

    # Limit to 100 frames
    python rtabmap_to_mapanything.py --rtabmap_path /path/to/rtabmap/output --max_frames 100

    # Just export data without running inference
    python rtabmap_to_mapanything.py --rtabmap_path /path/to/rtabmap/output --export_only

Expected RTAB-Map folder structure:
    rtabmap_output/
    ├── rgb/
    │   ├── 1.jpg
    │   ├── 4.jpg    # Non-sequential numbering is OK
    │   ├── 5.jpg
    │   └── ...
    ├── depth/
    │   ├── 1.png
    │   ├── 4.png    # Must match RGB numbers
    │   ├── 5.png
    │   └── ...
    ├── poses.txt    # Sequential poses (1st line = 1st image when sorted)
    └── calibration.yaml
        """
    )
    
    parser.add_argument(
        "--rtabmap_path",
        type=str,
        required=True,
        help="Path to RTAB-Map output folder containing rgb/, depth/, poses.txt, calibration.yaml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: rtabmap_path/mapanything_output)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every N-th frame (1 = all frames, 5 = every 5th frame). Default: 1"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process after applying stride (-1 for all). Default: -1"
    )
    parser.add_argument(
        "--no_depth",
        action="store_true",
        help="Don't use depth data (let MapAnything predict depth)"
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory-efficient inference (slower but handles more views)"
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Only export data to MapAnything format without running inference"
    )
    parser.add_argument(
        "--no_glb",
        action="store_true",
        help="Don't save output as GLB file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.rtabmap_path, "mapanything_output")
    
    print("=" * 60)
    print("RTAB-Map to MapAnything Converter")
    print("=" * 60)
    print(f"RTAB-Map path: {args.rtabmap_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stride: {args.stride}")
    print(f"Max frames: {args.max_frames if args.max_frames > 0 else 'all'}")
    print(f"Use depth: {not args.no_depth}")
    print("=" * 60)
    
    # Load RTAB-Map data
    views, intrinsics, resolution = load_rtabmap_data(
        rtabmap_path=args.rtabmap_path,
        stride=args.stride,
        max_frames=args.max_frames,
        use_depth=not args.no_depth,
        verbose=args.verbose
    )
    
    if args.export_only:
        print("\n--export_only specified, skipping inference")
        print(f"Loaded {len(views)} views ready for MapAnything")
        return
    
    # Run MapAnything inference
    run_mapanything_inference(
        views=views,
        output_dir=args.output_dir,
        memory_efficient=args.memory_efficient,
        save_glb=not args.no_glb,
        verbose=args.verbose
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
