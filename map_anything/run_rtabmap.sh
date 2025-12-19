#!/bin/bash
# Run RTAB-Map to MapAnything conversion inside the Docker container
#
# Usage:
#   ./run_rtabmap.sh /path/to/rtabmap/output [stride] [max_frames]
#
# Examples:
#   ./run_rtabmap.sh /workspace/datasets/my_rtabmap_data
#   ./run_rtabmap.sh /workspace/datasets/my_rtabmap_data 5        # Every 5th frame
#   ./run_rtabmap.sh /workspace/datasets/my_rtabmap_data 10 100   # Every 10th frame, max 100 frames

set -e

# Default values
RTABMAP_PATH="${1:-}"
STRIDE="${2:-1}"
MAX_FRAMES="${3:--1}"

if [ -z "$RTABMAP_PATH" ]; then
    echo "Usage: ./run_rtabmap.sh <rtabmap_path> [stride] [max_frames]"
    echo ""
    echo "Arguments:"
    echo "  rtabmap_path  Path to RTAB-Map output folder (required)"
    echo "  stride        Use every N-th frame (default: 1)"
    echo "  max_frames    Maximum frames to process, -1 for all (default: -1)"
    echo ""
    echo "Examples:"
    echo "  ./run_rtabmap.sh /workspace/datasets/Kitchen_1"
    echo "  ./run_rtabmap.sh /workspace/datasets/Kitchen_1 5"
    echo "  ./run_rtabmap.sh /workspace/datasets/Kitchen_1 10 100"
    exit 1
fi

# Check if we're inside the container or running from host
if [ -d "/tmp_build/map-anything" ]; then
    # Inside Docker container
    cd /tmp_build/map-anything
    SCRIPT_PATH="/workspace/map_anything/rtabmap_to_mapanything.py"
else
    # On host - script should be run inside container
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "This script should be run inside the map_anything Docker container."
    echo ""
    echo "To start the container:"
    echo "  cd $(dirname "$SCRIPT_DIR")"
    echo "  docker compose -f map_anything/docker-compose.yml up -d"
    echo "  docker exec -it map_anything_5090 bash"
    echo ""
    echo "Then run this script from inside the container."
    exit 1
fi

# Run the conversion script
python "$SCRIPT_PATH" \
    --rtabmap_path "$RTABMAP_PATH" \
    --stride "$STRIDE" \
    --max_frames "$MAX_FRAMES" \
    --verbose

echo ""
echo "Done! Output saved to: ${RTABMAP_PATH}/mapanything_output/"
