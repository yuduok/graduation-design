# 细粒度猫狗分类系统 - 项目完成报告

## 📋 项目概述

**项目名称**: 基于提示词优化的细粒度猫狗分类系统  
**技术栈**: PyTorch 2.4.1 + CLIP + Python 3.8  
**数据集**: Oxford-IIIT Pets (7,390张图片, 37种猫狗品种)  
**当前状态**: ✅ 全部测试通过，训练脚本已修复

---

## ✅ 已完成模块

### 1. 核心模型模块

| 文件 | 功能 | 状态 |
|------|------|------|
| `models/dynamic_prompt.py` | 动态提示词优化 | ✅ 已修复维度错误 |
| `models/custom_clip.py` | 自定义CLIP模型 | ✅ 已优化批量编码 |
| `models/trainer.py` | 训练器 | ✅ 已修复属性错误 |
| `models/breed_semantic.py` | 品种语义增强 | ✅ 测试通过 |

### 2. 训练与评估

| 文件 | 功能 | 状态 |
|------|------|------|
| `train.py` | 主训练脚本 | ✅ 已修复所有配置问题 |
| `evaluate.py` | 评估脚本 | ✅ 可运行 |
| `test_core.py` | 核心功能测试 | ✅ 已通过 |
| `test_demo.py` | Demo功能测试 | ✅ 已通过 |
| `test_full.py` | 端到端测试 | ✅ 已通过 |

### 3. 可视化与部署

| 文件 | 功能 | 状态 |
|------|------|------|
| `demo/pet_classifier_demo.py` | Streamlit演示 | ✅ 已修复导入路径 |
| `web/app.py` | Flask API | ✅ 已修复导入路径 |
| `run_demo.sh` | 启动脚本 | ✅ 完成 |
| `train_quick.sh` | 快速训练脚本 | ✅ 完成 |

### 4. 配置与文档

| 文件 | 功能 | 状态 |
|------|------|------|
| `configs/dynamic_rn50.yaml` | 训练配置 | ✅ 已优化 |
| `README.md` | 使用文档 | ✅ 完成 |

---

## 🔧 最近修复 (2026-03-11)

### 1. Streamlit Demo (`demo/pet_classifier_demo.py`)
- **问题**: 导入路径错误导致 `ImportError: attempted relative import with no known parent package`
- **修复**: 将 `sys.path.insert(0, os.path.join(COOP_PATH, "clip"))` 改为 `sys.path.insert(0, COOP_PATH)`

### 2. Flask API (`web/app.py`)
- **问题**: 缺少 CoOp 路径配置
- **修复**: 添加了正确的路径设置

### 3. 训练脚本 (`train.py`)
- **问题1**: 配置文件中的 `DATASET.SPLIT` 键在默认配置中不存在
- **修复**: 添加了 `extend_cfg()` 函数，扩展配置节点
- **问题2**: CoOp 路径使用相对路径导致模块导入失败
- **修复**: 改为绝对路径 `/Users/yudu/Documents/毕业设计/CoOp`
- **问题3**: 数据集名称不匹配（如 `oxford_pets` vs `OxfordPets`）
- **修复**: 添加了数据集名称映射字典
- **问题4**: 数据集路径错误
- **修复**: 设置 `cfg.DATASET.ROOT = DATA_PATH`（`/Users/yudu/Documents/毕业设计/data`）

### 4. 动态提示词 (`models/dynamic_prompt.py`)
- **问题**: `SoftPromptAdapter.forward` 中多余的 `unsqueeze(0)` 导致维度错误
- **修复**: 移除多余的维度扩展操作
- **问题**: `AdaptivePromptLearner` 缺少 `current_weights` 属性
- **修复**: 初始化 `self.current_weights = None`

### 5. 自定义CLIP (`models/custom_clip.py`)
- **问题**: 循环编码文本导致内存溢出（MPS out of memory）
- **修复**: 改为批量编码，大幅减少内存占用
- **问题**: logit 计算维度不匹配
- **修复**: 修正矩阵乘法维度

### 6. 训练器 (`models/trainer.py`)
- **问题**: `current_epoch` 属性不存在
- **修复**: 改为使用 `self.epoch`

---

## 📁 自动生成的日志文件

### 日志文件位置

训练过程中会在 `output_fgd/oxford_pets/` 目录下自动生成以下文件：

```
output_fgd/
└── oxford_pets/
    ├── log.txt                    # 当前训练日志
    ├── log.txt-2026-03-11-14-11-54  # 历史日志备份
    ├── log.txt-2026-03-11-14-13-22
    └── tensorboard/                # TensorBoard 可视化日志
        └── events.out.tfevents.*
```

### 日志生成机制

这些日志文件由 **CoOp/dassl 框架** 自动生成：

1. **log.txt 日志**
   - 由 `dassl.utils.setup_logger()` 函数创建
   - 路径：`/CoOp/dassl/dassl/utils/logger.py`
   - 逻辑：如果 `log.txt` 已存在，会自动添加时间戳后缀保存历史版本
   - 记录内容：训练进度、损失值、准确率、学习率等

2. **TensorBoard 日志**
   - 由 dassl 框架的 `TensorboardWriter` 自动创建
   - 路径：`output_fgd/oxford_pets/tensorboard/`
   - 记录内容：训练指标曲线、损失变化等
   - 查看方式：`tensorboard --logdir=output_fgd/oxford_pets/tensorboard`

### .gitignore 已配置

已在项目根目录创建 `.gitignore` 文件，自动忽略以下内容：

```
# 训练输出
output_fgd/
*.pth
*.pt

# TensorBoard
tensorboard/
events.out.tfevents.*

# 测试输出
*_output.png

# 数据集
oxford_pets/
```

---

## 🚀 快速使用

### 环境准备

```bash
# 使用CoOp项目已有的虚拟环境
source /Users/yudu/Documents/毕业设计/CoOp/venv/bin/activate

# 安装额外依赖
pip install streamlit flask flask-cors matplotlib seaborn scikit-learn
```

### 测试命令

```bash
cd /Users/yudu/Documents/毕业设计/fine_grained_classification

# 核心功能测试
python test_core.py

# Demo功能测试
python test_demo.py

# 端到端测试（使用真实数据集）
python test_full.py
```

### 启动演示界面

```bash
streamlit run /Users/yudu/Documents/毕业设计/fine_grained_classification/demo/pet_classifier_demo.py
# 访问 http://localhost:8501
```

### 启动API服务

```bash
cd /Users/yudu/Documents/毕业设计/fine_grained_classification/web
source /Users/yudu/Documents/毕业设计/CoOp/venv/bin/activate
python app.py
# API: http://localhost:5001/api/classify
```

### 训练模型

```bash
cd /Users/yudu/Documents/毕业设计/fine_grained_classification

# CPU 训练
python train.py -d oxford_pets -e 50 -b 32 --shots 1

# MPS 训练 (Mac)
python train.py -d oxford_pets -e 50 -b 32 --shots 1 --device mps

# CUDA 训练 (有GPU时)
python train.py -d oxford_pets -e 50 -b 32 --shots 1 --device cuda

# 小批量测试 (内存有限时)
python train.py -d oxford_pets -e 2 -b 4 --shots 1
```

---

## 📊 预期实验结果

| 方法 | 1-shot | 4-shot | 16-shot |
|------|--------|--------|---------|
| Zero-shot CLIP | ~60% | ~60% | ~60% |
| CoOp | ~75% | ~85% | ~90% |
| **Ours (Dynamic)** | ~78% | ~87% | ~91% |

---

## 📁 项目结构

```
fine_grained_classification/
├── train.py                      # 训练入口
├── evaluate.py                   # 评估脚本
├── test_core.py                  # 核心测试
├── test_demo.py                  # Demo测试
├── test_full.py                  # 端到端测试
├── run_demo.sh                   # 启动演示
├── train_quick.sh                # 快速训练
├── PROJECT_STATUS.md             # 本文档
├── README.md                     # 使用说明
│
├── configs/
│   └── dynamic_rn50.yaml         # 训练配置
│
├── models/
│   ├── __init__.py
│   ├── custom_clip.py            # 自定义CLIP (已优化)
│   ├── dynamic_prompt.py         # 动态提示词 (已修复)
│   ├── trainer.py                # 训练器 (已修复)
│   └── breed_semantic.py         # 语义增强
│
├── utils/
│   └── helpers.py                # 工具函数
│
├── demo/
│   └── pet_classifier_demo.py   # Streamlit演示 (已修复)
│
└── web/
    └── app.py                   # Flask API (已修复)
```

---

## 📈 下一步工作

- [x] 核心模块测试通过
- [x] Demo功能测试通过
- [x] 端到端测试通过
- [x] 训练脚本修复完成
- [ ] 运行完整训练实验
  - [ ] 1-shot 实验
  - [ ] 4-shot 实验
  - [ ] 16-shot 实验
- [ ] 对比CoOp/CoCoOp基线方法
- [ ] 绘制学习曲线
- [ ] 生成混淆矩阵
- [ ] 撰写论文实验章节

---

## 💡 创新点总结

1. **动态提示词调整机制** - 根据困难样本自适应修改提示词
2. **难度加权损失** - 对难分类样本给予更高权重
3. **品种语义增强** - 利用品种属性描述增强文本嵌入
4. **可视化决策解释** - 展示预测结果和置信度

---

**更新时间**: 2026-03-11 17:30  
**测试环境**: MacBook Air M2, PyTorch 2.4.1, Python 3.8  
**数据集**: Oxford-IIIT Pets (7,390张图片, 37类)  
**状态**: 全部测试通过 ✅

---

## 🎯 快速开始完整实验

```bash
# 1-shot实验
cd /Users/yudu/Documents/毕业设计/fine_grained_classification
python train.py -d oxford_pets -e 50 -b 32 --shots 1

# 4-shot实验
python train.py -d oxford_pets -e 50 -b 32 --shots 4

# 16-shot实验
python train.py -d oxford_pets -e 50 -b 32 --shots 16
```
