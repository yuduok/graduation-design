#!/bin/bash
# 快速训练测试 (1 epoch)

cd /Users/yudu/Documents/毕业设计/fine_grained_classification

# 使用CoOp虚拟环境
source /Users/yudu/Documents/毕业设计/CoOp/venv/bin/activate

export PYTHONPATH="/Users/yudu/Documents/毕业设计/CoOp:$PYTHONPATH"

echo "=========================================="
echo "细粒度分类训练 (快速测试)"
echo "=========================================="
echo ""
echo "数据集: Oxford-IIIT Pets"
echo "训练器: DynamicPromptTrainer"
echo "Backbone: RN50"
echo "Shots: 1"
echo "Epochs: 1"
echo ""

python train.py \
  -c configs/dynamic_rn50.yaml \
  --dataset oxford_pets \
  --trainer DynamicPromptTrainer \
  --epochs 1 \
  --batch-size 4 \
  --shots 1 \
  --split 0 \
  --eval-only
