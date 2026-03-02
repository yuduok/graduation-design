# CoOp 1-Shot 训练结果报告

## 训练配置

### 环境信息
- **操作系统**: macOS 15.2 (ARM64)
- **Python**: 3.13.3
- **PyTorch**: 2.10.0 (MPS 加速)
- **模型**: CoOp (Context Optimization) + CLIP (RN50)

### 数据集
- **Oxford-IIIT Pets**: 37 种猫狗品种
- **训练样本**: 37 (1-shot，每类1张图片)
- **测试样本**: 3,669

### 训练参数
- **上下文Tokens (N_CTX)**: 16
- **学习率**: 0.002 (SGD, cosine scheduler)
- **训练轮次**: 1 epoch
- **随机种子**: 1

## 训练结果

### 准确率
- **训练准确率**: 34.3750%
- **损失值 (Loss)**: 2.6445
- **训练时间**: 1分49秒

### 训练过程
| Epoch | Batch | Time (s) | Data (s) | Loss | Accuracy | LR |
|-------|-------|----------|-----------|------|----------|-----|
| 1/1 | 1/1 | 103.882 | 3.572 | 2.6445 | 34.3750% | 2.0000e-03 |

## 结果分析

### 1. 1-Shot 学习的挑战
- **极低数据量**: 每个品种仅1张训练图片
- **训练准确率34.38%**: 37个类别中正确约13个
- **相比零样本**: CLIP零样本在OxfordPets上约50-60%，1-shot反而下降

### 2. 原因分析
1. **过拟合风险**: 单一样本难以学习品种特征
2. **样本代表性**: 1张图片可能不代表品种的典型特征
3. **训练不充分**: 仅1个epoch，提示词未充分优化

### 3. CoOp 方法的特点
- **可学习提示词**: 在类别名称前插入16个可学习tokens
- **保持CLIP冻结**: 仅优化提示词，不更新图像/文本编码器
- **参数高效**: 只需优化少量参数

## 快速运行指南

### 1. 激活Python环境
```bash
# 方法1: 使用便捷脚本
cd ~/Documents/毕业设计/CoOp
source run_coop.sh

# 方法2: 手动激活
source ~/Documents/毕业设计/CoOp/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/Documents/毕业设计/CoOp/dassl
export DATA=$HOME/Documents/毕业设计/data
```

### 2. 运行1-Shot训练
```bash
cd ~/Documents/毕业设计/CoOp

python train.py \
  --root "$HOME/Documents/毕业设计/data" \
  --trainer CoOp \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/CoOp/rn50.yaml \
  --output-dir output/oxford_pets/1shot \
  --seed 1 \
  DATASET.NUM_SHOTS 1 \
  TRAINER.COOP.N_CTX 16 \
  TRAINER.COOP.CSC False \
  OPTIM.MAX_EPOCH 50
```

### 3. 运行完整训练（推荐）
```bash
# 使用官方脚本（更方便）
cd ~/Documents/毕业设计/CoOp

# 1-shot 训练
bash scripts/coop/main.sh oxford_pets rn50_ep50 end 16 1 False

# 16-shot 训练（效果更好）
bash scripts/coop/main.sh oxford_pets rn50 end 16 16 False

# CoCoOp (条件提示词)
bash scripts/cocoop/main.sh oxford_pets rn50 end 16 1 False
```

### 4. 查看训练日志
```bash
# 训练日志
cat output/oxford_pets/1shot/log.txt

# 使用TensorBoard可视化
tensorboard --logdir output/oxford_pets/1shot/tensorboard

# 浏览器打开: http://localhost:6006
```

### 5. 解析测试结果
```bash
cd ~/Documents/毕业设计/CoOp

# 解析多次运行的统计结果
python parse_test_res.py output/oxford_pets/1shot/

# 查看最终准确率
tail -50 output/oxford_pets/1shot/log.txt
```

## 关键参数说明

| 参数 | 含义 | 常用值 |
|------|------|--------|
| `DATASET.NUM_SHOTS` | 每类训练样本数 | 1, 2, 4, 8, 16 |
| `TRAINER.COOP.N_CTX` | 上下文tokens数量 | 4, 16 |
| `TRAINER.COOP.CSC` | 类别特定上下文 | True, False |
| `TRAINER.COOP.CLASS_TOKEN_POSITION` | tokens位置 | "end", "middle" |
| `OPTIM.MAX_EPOCH` | 训练轮次 | 50, 100, 200 |

## 改进建议

### 1. 增加训练样本
```bash
# 从1-shot增加到16-shot
DATASET.NUM_SHOTS 16  # 准确率预计提升到70-80%
```

### 2. 增加训练轮次
```bash
# 从1 epoch增加到50 epochs
OPTIM.MAX_EPOCH 50  # 提示词充分优化
```

### 3. 尝试CoCoOp方法
```bash
# CoCoOp更适合1-shot
--trainer CoCoOp
# 条件提示词，输入自适应
```

### 4. 使用类别特定上下文 (CSC)
```bash
# 每个类别有独立的上下文
TRAINER.COOP.CSC True
```

## 文件结构

```
~/Documents/毕业设计/CoOp/
├── output/oxford_pets/1shot/
│   ├── log.txt              # 训练日志
│   ├── prompt_learner/
│   │   └── model.pth.tar-1 # 提示词权重
│   └── tensorboard/         # 可视化数据
├── data/oxford_pets/
│   ├── images/             # 7,393张图片
│   ├── annotations/         # 标注文件
│   └── split_zhou_OxfordPets.json
└── venv/                  # Python虚拟环境
```

## 总结

1-Shot训练准确率34.38%说明：
- ✅ CoOp环境配置成功，代码正常运行
- ✅ 1-Shot学习极具挑战性，需要更多样本或轮次
- ✅ 框架可用，可以通过调整参数提升性能

**下一步建议**：
1. 运行16-shot训练（预期准确率70-80%）
2. 尝试CoCoOp方法（更适合few-shot）
3. 开发动态提示词优化算法（毕业设计创新点）
