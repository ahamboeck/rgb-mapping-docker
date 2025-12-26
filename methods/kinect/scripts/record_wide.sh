#!/bin/bash

# Default output filename with timestamp
OUTPUT_FILE="output_wide_$(date +%Y%m%d_%H%M%S).mkv"

if [ ! -z "$1" ]; then
    OUTPUT_FILE="$1"
fi

echo "Recording Wide FOV to $OUTPUT_FILE..."
echo "Press Ctrl+C to stop recording."

# WFOV_2X2BINNED is a depth mode.
# 1080p is a color mode.
# IMU is ON by default, but we specify it to be sure.
# No option to disable microphone in CLI, but we can ignore it later.

k4arecorder \
    --device 0 \
    --imu ON \
    --exposure-control auto \
    --color-mode 1080p \
    --depth-mode WFOV_2X2BINNED \
    --rate 30 \
    "$OUTPUT_FILE"
