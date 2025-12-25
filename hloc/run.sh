#!/bin/bash
xhost +local:docker
docker compose up -d --build
docker compose exec hloc zsh
