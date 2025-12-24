#!/bin/bash

# 设置错误处理
set -e

echo "Starting"
echo

echo "===== 1 ====="
python train_module_sip.py --cir 1 --module_name tensor_fno_mask --exp_marker top --epoch 100 --bs 8

echo
echo "===== 2 ====="
python train_module_sip.py --cir 2 --module_name tensor_fno_mask --exp_marker top --epoch 100 --bs 8

echo
echo "===== 3 ====="
python train_module_sip.py --cir 3 --module_name tensor_fno_mask --exp_marker top --epoch 100 --bs 8

echo "===== 4 ====="
python train_module_sip.py --cir 4 --module_name tensor_fno_mask --exp_marker top --epoch 100 --bs 8

echo
echo "===== 5 ====="
python train_module_sip.py --cir 5 --module_name tensor_fno_mask --exp_marker top --epoch 100 --bs 8

echo
echo "===== 6 ====="
python train_module_sip.py --cir 6 --module_name tensor_fno_mask --exp_marker top --epoch 100 --bs 8
echo
echo "========================================"
echo "All steps completed successfully!"
echo "========================================"