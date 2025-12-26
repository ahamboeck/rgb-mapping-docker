#!/bin/bash
# Wrapper for 3DGUT training
# Usage: ./run.sh --config-name apps/colmap_3dgut_mcmc.yaml path=... out_dir=...

# Activate conda env
source /root/miniconda3/etc/profile.d/conda.sh
conda activate 3dgrut

cd /workspace/3dgrut

if [ "$#" -eq 0 ]; then
    echo "Usage: ./run.sh [args]"
    echo "Example: ./run.sh --config-name apps/colmap_3dgut_mcmc.yaml path=/workspace/datasets/Doggo out_dir=/workspace/output/3dgut experiment_name=test"
    exit 1
fi

python train.py "$@"
