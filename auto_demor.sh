#!/bin/bash

# 设置错误处理
set -e

echo "Starting"
echo

echo "===== 1: SVD1 ====="
python Baseline_DeMOR.py --circuit 1

echo
echo "===== 2 ====="
python Baseline_DeMOR.py --circuit 2

echo
echo "===== 3 ====="
python Baseline_DeMOR.py --circuit 3


# echo "===== 4 ====="
# python MFMOR-generatedata.py --circuit 4

# echo
# echo "===== 5 ====="
# python save_svd_mor.py --circuit 5

# echo
# echo "===== 6 ====="
# python save_svd_mor.py --circuit 6

echo
echo "========================================"
echo "All steps completed successfully!"
echo "========================================"