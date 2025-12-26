#!/bin/bash

# Default output filename with timestamp
OUTPUT_FILE="output_narrow_$(date +%Y%m%d_%H%M%S).mkv"

if [ ! -z "$1" ]; then
    OUTPUT_FILE="$1"
fi

echo "Recording Narrow FOV to $OUTPUT_FILE..."
echo "Press Ctrl+C to stop recording."

# NFOV_UNBINNED is a depth mode.
# 1080p is a color mode.
# IMU is ON by default.

k4arecorder \
    --device 0 \
    --imu ON \
    --exposure-control auto \
    --color-mode 1080p \
    --depth-mode NFOV_UNBINNED \
    --rate 30 \
    "$OUTPUT_FILE"
