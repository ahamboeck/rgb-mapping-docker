#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/VGGT-SLAM"

if [ "$#" -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
