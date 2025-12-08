#!/bin/bash
# Helper script to run commands inside the FastVGGT container

if [ "$1" == "build" ]; then
    docker compose up --build -d
    exit 0
fi

if [ "$1" == "down" ]; then
    docker compose down
    exit 0
fi

if [ "$1" == "shell" ]; then
    docker exec -it fastvggt_5090 zsh
    exit 0
fi

# Default: run the command passed as arguments
docker exec -it fastvggt_5090 "$@"
