# 基于动态提示词优化的细粒度猫狗分类系统

基于 CLIP + 动态提示词优化的细粒度猫狗品种分类框架，结合多模态信息（图像特征与文本语义），提升模型对相似品种的区分能力。

## 环境安装

**Python 版本**: 3.8

```bash
# 1. 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 CLIP（从 CoOp 目录中引用，无需额外安装）
#    CLIP 和 Dassl 已作为源码包含在 ../CoOp/ 目录中
#    如果独立使用，需手动安装：
#    pip install git+https://github.com/openai/CLIP.git
#    cd ../CoOp/dassl && pip install -e .
```

## 数据准备

下载 Oxford-IIIT Pets 数据集，放置到与本项目平级的 `data/` 目录：

```bash
mkdir -p ../data/oxford_pets && cd ../data/oxford_pets

# 下载并解压
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

目录结构：
```
毕业设计/
├── CoOp/                        # CoOp 框架（含 CLIP + Dassl）
├── data/
│   └── oxford_pets/
│       ├── images/              # 7,390 张图片
│       ├── annotations/
│       └── split_zhou_OxfordPets.json
└── fine_grained_classification/ # 本项目
```

## 项目结构

```
fine_grained_classification/
├── train.py                     # 训练入口（支持 CoOp/CoCoOp/DynamicPromptTrainer）
├── evaluate.py                  # 评估脚本
├── collect_results.py           # 实验结果汇总与图表生成
├── run_experiments.sh           # 自动化批量实验脚本
├── requirements.txt             # Python 依赖
├── configs/
│   └── dynamic_rn50.yaml        # 训练超参配置
├── models/
│   ├── custom_clip.py           # 自定义 CLIP（整合动态提示 + 语义增强）
│   ├── dynamic_prompt.py        # AdaptivePromptLearner + SoftPromptAdapter
│   ├── trainer.py               # DynamicPromptTrainer（注册到 Dassl）
│   └── breed_semantic.py        # 品种属性库（37 品种特征描述）
├── demo/
│   └── pet_classifier_demo.py   # Streamlit 交互演示
├── web/
│   └── app.py                   # Flask REST API 服务
└── utils/
    └── helpers.py               # 可视化与度量工具
```

## 快速开始

### 训练模型

```bash
# 单组实验（输出自动保存到 output_fgd/oxford_pets/{trainer}/shots_{n}/seed_{s}/）
python train.py -d oxford_pets -e 50 -b 32 --shots 1 --trainer DynamicPromptTrainer --device cuda

# 批量运行全部 9 组对比实验（3 方法 × 3 shot）
bash run_experiments.sh cuda

# 汇总结果并生成图表
python collect_results.py --latex --plot
```

### 启动演示

```bash
# Streamlit 界面 → http://localhost:8501
streamlit run demo/pet_classifier_demo.py

# Flask API → http://localhost:5001
cd web && python app.py

# 使用训练好的模型启动 API
python app.py --model ../output_fgd/oxford_pets/DynamicPromptTrainer/shots_1/seed_1/prompt_learner/model-best.pth.tar
```

## 核心方法

### 动态提示词 vs CoOp

| 特性 | CoOp（基线） | 本系统（DynamicPromptTrainer） |
|------|-------------|-------------------------------|
| 提示词生成 | 所有图片共享同一组可学习 ctx 向量 | 每张图片经 SoftPromptAdapter 生成独特偏移 |
| 训练信号 | 均匀 Cross-Entropy loss | 难度加权 loss：难样本梯度放大 2-4x |
| 适应机制 | 无 | SoftPromptAdapter (512→64→512 双层 MLP) |

### 工作流程

```
输入图片 → [冻结] CLIP 视觉编码器 → image_features
         → [可训练] SoftPromptAdapter MLP → ctx 偏移
         → 基础 ctx + 偏移 → 图片条件提示词
         → [冻结] CLIP 文本编码器 → 相似度计算
         → DifficultyWeightCalculator → 加权 CE loss
         → 反向传播（仅更新 ctx + MLP 参数）
```

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| AdaptivePromptLearner | `models/dynamic_prompt.py` | 图像条件提示词生成 |
| SoftPromptAdapter | `models/dynamic_prompt.py` | 双层 MLP 生成 ctx 偏移 |
| DifficultyWeightCalculator | `models/dynamic_prompt.py` | 基于原型距离的难度加权 |
| BreedAttributeDatabase | `models/breed_semantic.py` | 37 品种属性库（毛发/面部/体型） |
| SemanticEnhancer | `models/breed_semantic.py` | 多模板语义增强 |

## 训练配置

主要超参数（`configs/dynamic_rn50.yaml`）：

| 参数 | 值 |
|------|-----|
| Backbone | CLIP RN50 |
| 优化器 | SGD (lr=0.002) |
| 学习率调度 | Cosine Annealing + warmup 1 epoch |
| 训练轮次 | 50 |
| 可学习 ctx | 4 tokens，初始化为 "a photo of a" |
| SoftPromptAdapter | 512→64→512 MLP |

## 实验结果

| 方法 | 1-shot | 4-shot | 16-shot |
|------|--------|--------|---------|
| Zero-shot CLIP | ~81% | ~81% | ~81% |
| CoOp | - | - | - |
| CoCoOp | - | - | - |
| **Ours (Dynamic)** | - | - | - |

> 待训练完成后由 `python collect_results.py` 自动填充。

## 参考资料

- [CLIP: Connecting Text and Images](https://github.com/openai/CLIP)
- [CoOp: Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [CoCoOp: Conditional Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
