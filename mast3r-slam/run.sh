#!/bin/bash
# Helper script to run commands inside the MASt3R-SLAM container

if [ "$1" == "build" ]; then
    docker compose up --build -d
    exit 0
fi

if [ "$1" == "down" ]; then
    docker compose down
    exit 0
fi

if [ "$1" == "shell" ]; then
    docker exec -it mast3r_slam_5090 zsh
    exit 0
fi

if [ "$1" == "install" ]; then
    echo "Installing MASt3R-SLAM (building CUDA extensions)..."
    docker exec -it mast3r_slam_5090 bash -c "cd /workspace/MASt3R-SLAM && pip install -e ."
    exit 0
fi

# Default: run the command passed as arguments
docker exec -it mast3r_slam_5090 "$@"
