#!/bin/bash
source /liujinxin/conda3/bin/activate 
conda activate 3dpi

python scripts/train.py pi05_test_right --batch_size=8 --overwrite --exp-name pi05_test_right  

