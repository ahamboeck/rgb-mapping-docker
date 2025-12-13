#!/bin/bash

# Allow X11 forwarding
xhost +local:docker > /dev/null 2>&1

# Create output directory if it doesn't exist
mkdir -p output

if [ "$1" == "build" ]; then
    docker compose build
elif [ "$1" == "run" ]; then
    docker compose run --rm kinect
else
    echo "Usage: $0 {build|run}"
    echo "  build: Build the Docker image using docker-compose"
    echo "  run:   Run the container using docker-compose"
fi
