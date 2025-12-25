import open3d as o3d
import numpy as np
import cv2
import os
import glob
import yaml
import time
import sys

# --- CONFIGURATION ---
DATA_PATH = "./"
RGB_FOLDER = "rgb"
DEPTH_FOLDER = "depth"
POSE_FILE = "poses.txt"
CALIB_FILE = "calibration.yaml"

# TSDF Settings
VOXEL_SIZE = 0.01  # 2cm resolution (Good balance)
TRUNC_MARGIN = 0.03  # 3 times voxel size
MAX_FRAMES = -1 # Set to -1 to process all frames

def read_calibration(filename):
    """ Reads FX, FY, CX, CY and local_transform from RTAB-Map calibration.yaml """
    with open(filename, 'r') as f:
        content = f.read().replace("%YAML:1.0", "")
        data = yaml.safe_load(content)

    # Parse local_transform (usually body -> camera)
    T_local = np.eye(4)
    if 'local_transform' in data:
        try:
            lt_data = data['local_transform']['data']
            lt_matrix = np.array(lt_data).reshape(3, 4)
            T_local[:3, :] = lt_matrix
            print("Found local_transform in calibration.")
        except Exception as e:
            print(f"Warning: Could not parse local_transform: {e}")

    # Support multiple possible YAML formats produced by RTAB-Map or camera_info
    # 1) RTAB-Map style: data['Camera']['k']['data'], width/height at Camera.width
    # 2) camera_info style (this repo): image_width/image_height and camera_matrix.data
    if isinstance(data, dict):
        # PREFER projection_matrix as it seems to be more reliably standard in this dataset
        if 'projection_matrix' in data:
            pm = data['projection_matrix'].get('data', [])
            if len(pm) >= 12:
                fx = pm[0]
                cx = pm[2]
                fy = pm[5]
                cy = pm[6]
                width = data.get('image_width') or data.get('width')
                height = data.get('image_height') or data.get('height')
                return width, height, fx, fy, cx, cy, T_local

        # RTAB-Map style
        if 'Camera' in data:
            cam = data['Camera']
            matrix_data = cam['k']['data']
            fx = matrix_data[0]
            cx = matrix_data[2]
            fy = matrix_data[4]
            cy = matrix_data[5]
            width = cam.get('width')
            height = cam.get('height')
            return width, height, fx, fy, cx, cy, T_local

        # camera_info / ros image style fallback
        if 'camera_matrix' in data:
            cm = data['camera_matrix']
            matrix_data = cm.get('data', [])
            if len(matrix_data) >= 9:
                # Try standard indices first
                fx = matrix_data[0]
                cx = matrix_data[2]
                fy = matrix_data[4]
                cy = matrix_data[5]
                # If fy is 0, try the weird indices seen in this file
                if fy == 0 and len(matrix_data) > 6:
                     fy = matrix_data[5]
                     cy = matrix_data[6]
                
                width = data.get('image_width') or data.get('width')
                height = data.get('image_height') or data.get('height')
                return width, height, fx, fy, cx, cy, T_local

    raise KeyError('Calibration format not recognized')


def read_poses(filename):
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            data = line.split()
            if len(data) < 8:
                continue
            timestamp = data[0]

            t = np.array([float(x) for x in data[1:4]])
            q = np.array([float(x) for x in data[4:8]]) # qx qy qz qw
            R = o3d.geometry.get_rotation_matrix_from_quaternion(np.array([q[3], q[0], q[1], q[2]]))
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)
    return poses

def main():
    # 1. Load Calibration
    try:
        WIDTH, HEIGHT, FX, FY, CX, CY, T_local = read_calibration(os.path.join(DATA_PATH, CALIB_FILE))
        print(f"Loaded Calibration: {WIDTH}x{HEIGHT}, FX={FX}, FY={FY}")
    except Exception as e:
        print(f"Error reading calibration: {e}")
        print("Please check yaml format or hardcode values.")
        return

    # 2. Initialize Volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=TRUNC_MARGIN,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # 3. Load Data
    poses = read_poses(os.path.join(DATA_PATH, POSE_FILE))
    
    # The user confirmed that poses and images are sequential (1st pose = 1st image).
    # The filenames might have gaps (e.g. 1.jpg, 4.jpg, 5.jpg), but they correspond 
    # to the sequence of poses in poses.txt.
    # So we just sort them numerically to get the correct capture order, 
    # and then zip them with the poses list directly.
    rgb_files = sorted(glob.glob(os.path.join(DATA_PATH, RGB_FOLDER, "*")), 
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    depth_files = sorted(glob.glob(os.path.join(DATA_PATH, DEPTH_FOLDER, "*")), 
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    print(f"Found {len(rgb_files)} images and {len(poses)} poses. Starting integration...")

    # Ensure we don't go out of bounds if counts differ slightly
    num_frames = min(len(poses), len(rgb_files), len(depth_files))
    
    if MAX_FRAMES > 0:
        num_frames = min(num_frames, MAX_FRAMES)
        print(f"Limiting to first {num_frames} frames for testing.")
    
    start_time = time.time()

    for i in range(num_frames):
        rgb_path = rgb_files[i]
        depth_path = depth_files[i]
        
        # Apply local transform (Body -> Camera)
        # User confirmed poses are already in Camera Frame.
        # So we do NOT apply T_local.
        pose = poses[i]
        
        if i % 50 == 0: 
            elapsed = time.time() - start_time
            print(f"Integrating frame {i}/{num_frames} (Elapsed: {elapsed:.1f}s)...", flush=True)
        
        # Read Images using OpenCV to handle formats safely
        color = cv2.imread(rgb_path)
        if color is None:
            print(f"Warning: Could not read RGB image {rgb_path}. Skipping frame.")
            continue
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: Could not read Depth image {depth_path}. Skipping frame.")
            continue

        
        # Auto-detect depth scale
        d_scale = 1000.0
        if depth.dtype == np.float32:
            d_scale = 1.0
        
        # Create Open3D Images
        o3d_color = o3d.geometry.Image(color)
        o3d_depth = o3d.geometry.Image(depth)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, 
            depth_scale=d_scale, 
            depth_trunc=4.0, 
            convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    # 4. Extract & Save
    integration_time = time.time() - start_time
    print(f"Integration finished in {integration_time:.1f}s.", flush=True)
    print("Extracting mesh from TSDF volume... (This creates the 3D geometry)", flush=True)
    
    mesh = volume.extract_triangle_mesh()
    print(f"Mesh extracted. Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}", flush=True)
    
    print("Computing normals...", flush=True)
    mesh.compute_vertex_normals()
    
    print("Saving to tsdf_mesh.ply...", flush=True)
    o3d.io.write_triangle_mesh("tsdf_mesh.ply", mesh)
    print("Done! Saved to tsdf_mesh.ply", flush=True)

if __name__ == "__main__":
    main()
