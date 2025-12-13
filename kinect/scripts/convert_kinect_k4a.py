#!/usr/bin/env python3
"""
Extract RGB, Aligned Depth, IMU data, and timestamps from Azure Kinect MKV files
using the native Azure Kinect SDK (pyk4a).

Updates:
- Added depth-to-color alignment so Depth and RGB resolutions match (Critical for RTAB-Map).
- Retained all original IMU and timestamp extraction logic.
"""

import argparse
import os
import numpy as np
import cv2
from pyk4a import PyK4APlayback

def process_mkv(input_mkv, output_dir):
    if not os.path.exists(input_mkv):
        print(f"Error: Input file {input_mkv} not found.")
        return

    print(f"Opening {input_mkv}...")
    
    # Open the playback file
    playback = PyK4APlayback(input_mkv)
    playback.open()
    
    # Get calibration data (Required for transformation)
    calibration = playback.calibration
    
    # Create output directories
    rgb_path = os.path.join(output_dir, "images")
    depth_path = os.path.join(output_dir, "depth")
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    
    # Extract and save intrinsics
    # Note: Because we align depth to color, we use the Color Camera intrinsics for both.
    color_intrinsics = calibration.get_camera_matrix(0)  # 0 = color camera
    intrinsics_file = os.path.join(output_dir, "intrinsic.txt")
    np.savetxt(intrinsics_file, color_intrinsics)
    print(f"Saved camera intrinsics to {intrinsics_file}")
    
    # Prepare data collection
    timestamps_list = []
    imu_data_list = []
    
    idx = 0
    print(f"Extracting frames to {output_dir}...")
    
    info_saved = False

    try:
        while True:
            try:
                capture = playback.get_next_capture()
                
                if capture.color is not None and capture.depth is not None:
                    # --- 1. Process Color ---
                    color = capture.color
                    
                    # Handle MJPEG (1D) vs BGRA (3D)
                    if color.ndim == 1:
                        # MJPEG compressed data -> Decode to BGR
                        color_bgr = cv2.imdecode(color, cv2.IMREAD_COLOR)
                    else:
                        # BGRA raw data -> Convert to BGR
                        color_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

                    # --- 2. Process Depth (THE FIX) ---
                    # We must transform the raw depth map to match the color camera's
                    # resolution and perspective.
                    # If we don't do this, RTAB-Map drops the data because dimensions mismatch.
                    try:
                        aligned_depth = calibration.transform_to_color_camera(capture.depth)
                    except AttributeError:
                        print("Error: PyK4A missing 'transform_to_color_camera'. Update your library.")
                        break

                    # --- 3. Save Info (once) ---
                    if not info_saved:
                        h, w, _ = color_bgr.shape
                        dh, dw = aligned_depth.shape
                        
                        # Sanity check to ensure alignment worked
                        if h != dh or w != dw:
                            print(f"Warning: Alignment failed? Color: {w}x{h}, Depth: {dw}x{dh}")

                        with open(os.path.join(output_dir, "info.txt"), "w") as f:
                            f.write(f"width: {w}\n")
                            f.write(f"height: {h}\n")
                            f.write("depth_scale: 1000.0\n")  # Kinect depth is in mm
                        info_saved = True

                    # --- 4. Save to Disk ---
                    cv2.imwrite(f"{rgb_path}/{idx:05d}.png", color_bgr)
                    
                    # Save Aligned Depth image (16-bit PNG)
                    cv2.imwrite(f"{depth_path}/{idx:05d}.png", aligned_depth)
                    
                    # Save timestamp (in microseconds)
                    timestamp_usec = capture.color_timestamp_usec
                    timestamps_list.append(timestamp_usec)
                    
                    # --- 5. IMU Extraction ---
                    try:
                        # Some versions might have it on capture
                        imu_sample = getattr(capture, 'imu_sample', None)
                        
                        if imu_sample is not None:
                             # IMU data: timestamp, accel (x,y,z), gyro (x,y,z)
                            imu_data_list.append({
                                'timestamp': imu_sample.acc_timestamp_usec,
                                'accel': imu_sample.acc_sample,
                                'gyro': imu_sample.gyro_sample
                            })
                    except Exception:
                        pass # IMU extraction failed for this frame
                    
                    idx += 1
                    if idx % 100 == 0:
                        print(f"Processed {idx} frames")
                        
            except EOFError:
                break
                
    finally:
        playback.close()
    
    # --- 6. Fallback IMU Extraction (C++ Helper) ---
    if not imu_data_list:
        # Only run this if Python extraction returned nothing
        
        imu_file = os.path.join(output_dir, "imu.csv")
        cpp_binary = os.path.join(os.path.dirname(__file__), "extract_imu")
        
        # Check if binary exists before trying to run it
        if os.path.exists(cpp_binary):
            print("Attempting separate IMU extraction pass using C++ helper...")
            cmd = f"{cpp_binary} {input_mkv} {imu_file}"
            ret = os.system(cmd)
            
            if ret == 0:
                print(f"IMU extraction successful. Data saved to {imu_file}")
            else:
                print("Error: C++ IMU extractor failed (non-zero exit code).")
        else:
             print("Notice: No Python IMU data found and 'extract_imu' binary missing. Skipping IMU.")

    # --- 7. Save Timestamps ---
    timestamps_file = os.path.join(output_dir, "timestamps.txt")
    with open(timestamps_file, "w") as f:
        for ts in timestamps_list:
            f.write(f"{ts}\n")
    print(f"Saved {len(timestamps_list)} timestamps to {timestamps_file}")
    
    # --- 8. Save Python-extracted IMU data ---
    if imu_data_list:
        imu_file = os.path.join(output_dir, "imu.csv")
        with open(imu_file, "w") as f:
            f.write("timestamp_usec,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
            for imu in imu_data_list:
                acc = imu['accel']
                gyro = imu['gyro']
                f.write(f"{imu['timestamp']},{acc[0]},{acc[1]},{acc[2]},{gyro[0]},{gyro[1]},{gyro[2]}\n")
        print(f"Saved {len(imu_data_list)} IMU samples to {imu_file}")
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from Azure Kinect MKV file")
    parser.add_argument("input", help="path to .mkv file")
    parser.add_argument("output", help="folder to save extracted data")
    args = parser.parse_args()
    
    process_mkv(args.input, args.output)