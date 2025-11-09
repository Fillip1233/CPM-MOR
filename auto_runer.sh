#!/bin/bash

# 设置错误处理
set -e

echo "Starting"
echo

echo "===== 1: SVD1 ====="
python save_svd_mor.py --circuit 1

echo
echo "===== 1: SVD2 ====="
python save_svd_mor.py --circuit 2

echo
echo "===== 1: SVD3 ====="
python save_svd_mor.py --circuit 3

# echo
# echo "===== 1: SVD4 ====="
# python train_model.py --mode time_freq --lr 0.0005 --epochs 50 --alpha 1.0 --beta 0.5

# echo
# echo "===== 1: SVD5 ====="
# python active_learning_train.py --cycles 5 --query_ratio 0.1 --epochs_per_cycle 20

# echo
# echo "===== 1: SVD6 ====="
# python ensemble_train.py --n_models 5 --epochs 50

# echo "===== 1: SVD1 ====="
# python test1.py

# echo
# echo "===== 1: SVD2 ====="
# python test2.py

echo
echo "========================================"
echo "All steps completed successfully!"
echo "========================================"