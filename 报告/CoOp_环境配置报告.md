# CoOp 环境配置与训练测试报告

## 1. 环境配置

### 1.1 项目位置
- **代码目录**: `~/Documents/毕业设计/CoOp/`
- **虚拟环境**: `~/Documents/毕业设计/CoOp/venv/`
- **数据集目录**: `~/Documents/毕业设计/data/oxford_pets/`

### 1.2 依赖版本
- Python: 3.13.3
- PyTorch: 2.10.0 (支持 Apple Silicon MPS 加速)
- torchvision: 0.25.0
- CLIP: OpenAI CLIP

### 1.3 修复的问题
- ✅ PyTorch 2.x 与 Dassl 的 `LRScheduler` 兼容性问题
- ✅ Dassl setup.py 版本解析问题

### 1.4 数据集信息
- **Oxford IIIT-Pets 数据集**
- 类别数: 37 (猫狗品种)
- 训练样本: 37 (1-shot 每类)
- 测试样本: 3,669

## 2. 训练配置

### 2.1 训练参数
- **模型**: CoOp (Context Optimization)
- **骨干网络**: ResNet-50 + CLIP
- **上下文 tokens**: 16
- **学习率**: 0.002 (SGD with cosine scheduler)
- **训练轮次**: 200 epochs
- **1-shot 学习**: 每类仅 1 个训练样本

### 2.2 运行命令
```bash
cd ~/Documents/毕业设计/CoOp
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/Documents/毕业设计/CoOp/dassl

python train.py \
  --root "$HOME/Documents/毕业设计/data" \
  --trainer CoOp \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/CoOp/rn50.yaml \
  --output-dir output/oxford_pets/rn50_1shot \
  --seed 1 \
  DATASET.NUM_SHOTS 1 \
  TRAINER.COOP.N_CTX 16 \
  TRAINER.COOP.CSC False
```

## 3. 训练结果

> 训练进行中...

## 4. 快速启动指南

### 4.1 激活环境
```bash
cd ~/Documents/毕业设计/CoOp
source run_coop.sh
```

### 4.2 数据集准备
按照 `DATASETS.md` 下载并组织数据集到 `~/Documents/毕业设计/data/` 目录。

### 4.3 运行训练
```bash
# 1-shot 训练
bash scripts/coop/main.sh oxford_pets rn50 end 16 1 False

# 完整训练 (200 epochs)
bash scripts/coop/main.sh oxford_pets rn50 end 16 16 False
```

## 5. 相关文件

- 论文: `~/Documents/毕业设计/英文原文.pdf`
- 代码仓库: https://github.com/KaiyangZhou/CoOp
- CoCoOp 代码: https://github.com/KaiyangZhou/CoOp (COCOOP.md)
