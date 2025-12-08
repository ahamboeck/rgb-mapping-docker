#!/bin/bash
# Run MASt3R demo
# Usage: ./run.sh [extra args]

python3 /tmp_build/mast3r/demo.py \
    --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --server_name 0.0.0.0 \
    "$@"
