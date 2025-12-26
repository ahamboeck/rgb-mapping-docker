#!/bin/bash
# Run NKSR check
# Usage: ./run.sh

echo "Checking NKSR installation..."
python3 -c "
import torch
import nksr
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'CUDA is available: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('CUDA not available, using CPU')

try:
    reconstructor = nksr.Reconstructor(device)
    print('NKSR Reconstructor initialized successfully on', device)
except Exception as e:
    print('Failed to initialize NKSR Reconstructor:', e)
"
