import argparse
import os
import pycolmap
import numpy as np

def convert_colmap_to_tum(model_path, output_name="camera_frame_poses.txt"):
    # 1. Check for COLMAP model files
    bin_path = os.path.join(model_path, "images.bin")
    txt_path = os.path.join(model_path, "images.txt")
    
    if not os.path.exists(bin_path) and not os.path.exists(txt_path):
        print(f"Error: No images.bin or images.txt found in {model_path}")
        return

    # 2. Load the reconstruction
    reconstruction = pycolmap.Reconstruction(model_path)
    
    # 3. Export to TXT (useful for debugging)
    reconstruction.write_text(model_path)
    print(f"-> Exported .txt files to {model_path}")

    # 4. Process Poses for TUM Format
    poses = []
    
    sorted_image_ids = sorted(reconstruction.images.keys(), 
                             key=lambda x: reconstruction.images[x].name)

    for img_id in sorted_image_ids:
        image = reconstruction.images[img_id]
        
        # FIX: Call cam_from_world() as a method to get the pose object
        pose_data = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        
        # Get C2W translation (projection center)
        t_c2w = image.projection_center()
        
        # Get W2C rotation and conjugate it to get C2W
        # pycolmap rotation quat is usually [w, x, y, z]
        q_w2c = pose_data.rotation.quat
        q_c2w = np.array([q_w2c[0], -q_w2c[1], -q_w2c[2], -q_w2c[3]])
        
        timestamp = os.path.splitext(image.name)[0]
        # TUM Format: name/timestamp x y z qx qy qz qw
        poses.append(f"{timestamp} {t_c2w[0]} {t_c2w[1]} {t_c2w[2]} "
                     f"{q_c2w[1]} {q_c2w[2]} {q_c2w[3]} {q_c2w[0]}")

    output_path = os.path.join(model_path, output_name)
    with open(output_path, "w") as f:
        f.write("\n".join(poses))
    
    print(f"-> Created TUM format: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the COLMAP sparse folder")
    args = parser.parse_args()
    convert_colmap_to_tum(args.path)