#!/bin/bash
# Run VGGT demo in chunks
# Usage: ./run.sh <path_to_images> [output_name]

# Pass help flag directly to python script
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    python3 /workspace/run_chunks.py --help
    exit 0
fi

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <path_to_images> [output_name] [extra_args...]"
    exit 1
fi

IMAGE_PATH="$1"
shift

# Check if the next argument is a name (not starting with -)
if [ -n "$1" ] && [[ "$1" != -* ]]; then
    NAME="$1"
    shift
else
    NAME="pointcloud"
fi

python3 /workspace/run_chunks.py --image_path "$IMAGE_PATH" --name "$NAME" "$@"
