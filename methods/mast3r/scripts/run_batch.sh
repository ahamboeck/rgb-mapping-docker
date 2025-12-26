#!/bin/bash
# Helper script to run MASt3R batch processing from host

if [ -z "$1" ]; then
    echo "Usage: ./run_batch.sh <path_to_dataset_folder> [output_name] [extra_args...]"
    echo "Example: ./run_batch.sh /workspace/datasets/my_room my_room_output --window_size 10"
    exit 1
fi

# Ensure container is running
if [ ! "$(docker ps -q -f name=mast3r_5090)" ]; then
    echo "Starting mast3r container..."
    docker compose up -d
fi

# Run the script inside the container
docker exec -it mast3r_5090 /workspace/run_custom.sh "$@"
