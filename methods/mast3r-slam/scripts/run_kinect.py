import argparse
import os
import numpy as np
import yaml
import subprocess
import sys

def run_mast3r_slam(kinect_output_dir, config_path="config/base.yaml"):
    # Paths
    images_dir = os.path.join(kinect_output_dir, "images")
    intrinsic_file = os.path.join(kinect_output_dir, "intrinsic.txt")
    info_file = os.path.join(kinect_output_dir, "info.txt")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return

    # Read Info (Width/Height)
    width = 0
    height = 0
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            for line in f:
                if "width:" in line:
                    width = int(line.split(":")[1].strip())
                elif "height:" in line:
                    height = int(line.split(":")[1].strip())
    else:
        print(f"Error: info.txt not found at {info_file}")
        return

    # Read Intrinsics
    fx, fy, cx, cy = 0, 0, 0, 0
    if os.path.exists(intrinsic_file):
        try:
            K = np.loadtxt(intrinsic_file)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        except Exception as e:
            print(f"Error reading intrinsic.txt: {e}")
            return
    else:
        print(f"Error: intrinsic.txt not found at {intrinsic_file}")
        return

    # Create temporary calibration YAML
    calib_data = {
        "width": width,
        "height": height,
        "calibration": [float(fx), float(fy), float(cx), float(cy)]
    }
    
    temp_calib_file = os.path.join(kinect_output_dir, "mast3r_calib.yaml")
    with open(temp_calib_file, "w") as f:
        yaml.dump(calib_data, f)
    
    print(f"Generated calibration file: {temp_calib_file}")
    print(f"Running MASt3R-SLAM on {images_dir}...")

    # Construct command
    # Assuming we are running from the MASt3R-SLAM root directory
    cmd = [
        sys.executable, "main.py",
        "--dataset", images_dir,
        "--config", config_path,
        "--calib", temp_calib_file
    ]
    
    print("Executing:", " ".join(cmd))
    
    # Add thirdparty modules to PYTHONPATH
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    
    # We are in MASt3R-SLAM root
    mast3r_path = os.path.abspath("thirdparty/mast3r")
    dust3r_path = os.path.abspath("thirdparty/mast3r/dust3r")
    in3d_path = os.path.abspath("thirdparty/in3d")
    
    paths_to_add = []
    if os.path.exists(mast3r_path):
        paths_to_add.append(mast3r_path)
    if os.path.exists(dust3r_path):
        paths_to_add.append(dust3r_path)
    if os.path.exists(in3d_path):
        paths_to_add.append(in3d_path)
        
    if paths_to_add:
        env["PYTHONPATH"] = os.pathsep.join(paths_to_add + [python_path])
        print(f"Added to PYTHONPATH: {paths_to_add}")
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"MASt3R-SLAM failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MASt3R-SLAM on Kinect data")
    parser.add_argument("kinect_output", help="Path to the kinect output directory (containing images/, intrinsic.txt, etc.)")
    parser.add_argument("--config", default="config/base.yaml", help="Path to MASt3R-SLAM config file")
    
    args = parser.parse_args()
    
    # Ensure we are in the MASt3R-SLAM root directory if running from scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Check if main.py exists in current dir, if not try to switch to root
    if not os.path.exists("main.py"):
        if os.path.exists(os.path.join(root_dir, "main.py")):
            os.chdir(root_dir)
            print(f"Changed working directory to {root_dir}")
        elif os.path.exists(os.path.join(root_dir, "MASt3R-SLAM", "main.py")):
            os.chdir(os.path.join(root_dir, "MASt3R-SLAM"))
            print(f"Changed working directory to {os.path.join(root_dir, 'MASt3R-SLAM')}")
        else:
            print("Warning: main.py not found. Make sure you run this script from the MASt3R-SLAM root or scripts directory.")

    # Ensure checkpoints exist (link from system if available)
    if not os.path.exists("checkpoints") and os.path.exists("/opt/mast3r_checkpoints"):
        print("Linking pre-downloaded checkpoints from /opt/mast3r_checkpoints...")
        try:
            os.symlink("/opt/mast3r_checkpoints", "checkpoints")
        except Exception as e:
            print(f"Warning: Failed to link checkpoints: {e}")

    run_mast3r_slam(args.kinect_output, args.config)
