#!/bin/bash

# 设置错误处理
set -e

echo "Starting"
echo

echo "===== 1: mask1 ====="
python mask_try.py --circuit 1

echo "===== 2: mask2 ====="
python mask_try.py --circuit 2

echo "===== 3: mask3 ====="
python mask_try.py --circuit 3

echo "===== 4: mask4 ====="
python mask_try.py --circuit 4

echo
echo "===== 5: mask5 ====="
python mask_try.py --circuit 5

echo
echo "===== 6: mask6 ====="
python mask_try.py --circuit 6

echo
echo "========================================"
echo "All steps completed successfully!"
echo "========================================"