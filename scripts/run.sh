#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$PROJECT_ROOT/results"

echo "=== 开始训练 Transformer 模型 ==="
python "$PROJECT_ROOT/src/train.py" --seed 42 | tee "$PROJECT_ROOT/results/train_log.txt"

echo "训练完成，结果保存在 $PROJECT_ROOT/results/"
