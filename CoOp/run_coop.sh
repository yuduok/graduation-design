#!/bin/zsh
# CoOp 环境激活脚本 (毕业设计版)
# 使用方法: source run_coop.sh

# 激活虚拟环境
source ~/Documents/毕业设计/CoOp/venv/bin/activate

# 设置 PYTHONPATH 以包含 dassl
export PYTHONPATH=$PYTHONPATH:~/Documents/毕业设计/CoOp/dassl

# 设置数据集路径
export DATA=$HOME/Documents/毕业设计/data

echo "✓ CoOp 环境已激活"
echo "  - 工作目录: ~/Documents/毕业设计/CoOp"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - 数据集目录: $DATA"
echo ""
echo "运行示例:"
echo "  # 训练 CoOp (OxfordPets, 1-shot):"
echo "  bash scripts/coop/main.sh oxford_pets rn50 end 16 1 False"
