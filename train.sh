#!/bin/bash
source /liujinxin/conda3/bin/activate 3dpi
cd /liujinxin/code/lhc/wy/openpi

# CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=offline python scripts/train.py pi05_test_right --batch-size=18 --overwrite --exp-name pi05_test_right --model.dtype=bfloat16 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 WANDB_MODE=offline python scripts/train.py pi05_right --batch-size=72 --overwrite --exp-name pi05_test_right --model.dtype=bfloat16 

