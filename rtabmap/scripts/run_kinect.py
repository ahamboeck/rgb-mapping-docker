import os
import argparse
import numpy as np
import subprocess
import cv2
import shutil

def create_calibration_yaml(intrinsic_path, output_yaml, width, height):
    try:
        K = np.loadtxt(intrinsic_path)
        # Handle case where file might have extra info or different shape
        if K.size >= 9:
             # Reshape if it's a flat array
             K = K.reshape((3,3)) if K.shape != (3,3) else K
             fx, fy = K[0, 0], K[1, 1]
             cx, cy = K[0, 2], K[1, 2]
        else:
             print("Error: Intrinsic file format not recognized.")
             return False

        dist = [0, 0, 0, 0, 0]
        
        content = f"""%YAML:1.0
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ {fx}, 0., {cx}, 0., {fy}, {cy}, 0., 0., 1. ]
dist_coeff: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ {dist[0]}, {dist[1]}, {dist[2]}, {dist[3]}, {dist[4]} ]
image_width: {width}
image_height: {height}
"""
        with open(output_yaml, "w") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error creating calibration file: {e}")
        return False

def get_rtabmap_args():
    return [
        "--Mem/BinDataKept", "true",
        "--Mem/IntermediateNodeDataKept", "true",
        "--RGBD/CreateOccupancyGrid", "false",
        "--Rtabmap/ImagesAlreadyRectified", "true",
        "--Rtabmap/PublishStats", "true",
        "--Mem/SaveDepth16Format", "true"
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to extracted folder")
    parser.add_argument("--output", required=True, help="Path to output database")
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_db = os.path.abspath(args.output)
    rgb_dir = os.path.join(input_dir, "images")
    depth_dir = os.path.join(input_dir, "depth")
    intrinsic_file = os.path.join(input_dir, "intrinsic.txt")
    
    # 1. Detect Resolution
    img_files = sorted(os.listdir(rgb_dir))
    if not img_files:
        print("Error: No images found.")
        return
    img = cv2.imread(os.path.join(rgb_dir, img_files[0]))
    height, width = img.shape[:2]
    print(f"Resolution: {width}x{height}")

    # 2. Calibration
    calib_file = os.path.join(input_dir, "calibration.yaml")
    create_calibration_yaml(intrinsic_file, calib_file, width, height)
    
    # 3. Clean previous DBs
    output_dir_path = os.path.dirname(output_db)
    os.makedirs(output_dir_path, exist_ok=True)
    for name in ["rtabmap.db", "rtabmapconsole.db"]:
        p = os.path.join(output_dir_path, name)
        if os.path.exists(p): os.remove(p)
        p = os.path.join(os.getcwd(), name)
        if os.path.exists(p): os.remove(p)

    # 4. Run RTAB-Map
    cmd = [
        "rtabmap-console",
        "-rgbd",
        "-rgb", rgb_dir,
        "-depth", depth_dir,
        "-camera_config", calib_file
    ] + get_rtabmap_args() + [
        "--Rtabmap/WorkingDirectory", output_dir_path,
        rgb_dir
    ]
    
    print("Running RTAB-Map...")
    subprocess.run(cmd, check=False)

    # 5. Move DB
    found_db = None
    for name in ["rtabmap.db", "rtabmapconsole.db"]:
        p = os.path.join(output_dir_path, name)
        if os.path.exists(p):
            found_db = p
            break
        p = os.path.join(os.getcwd(), name)
        if os.path.exists(p):
            found_db = p
            break
            
    if found_db:
        if os.path.abspath(found_db) != output_db:
            if os.path.exists(output_db): os.remove(output_db)
            shutil.move(found_db, output_db)
        print(f"Database saved to {output_db}")
        
        # 6. Export Colored Point Cloud
        ply_output = output_db.replace(".db", ".ply")
        print(f"Exporting colored cloud to {ply_output}...")
        
        # CORRECTION: 
        # - Removed --texture and --mesh (causes crash/confusion)
        # - Using strictly positional arguments: [db] [output]
        export_cmd = [
            "rtabmap-export",
            "--cloud",             
            output_db,
            ply_output
        ]
        
        try:
            print(f"Exec: {' '.join(export_cmd)}")
            subprocess.run(export_cmd, check=True)
            print("Export finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Export failed with code {e.returncode}")
    else:
        print("Error: Database creation failed.")

if __name__ == "__main__":
    main()