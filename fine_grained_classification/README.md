# 基于动态提示词优化的细粒度猫狗分类系统

基于 CLIP + 动态提示词优化的细粒度猫狗品种分类框架，结合多模态信息（图像特征与文本语义），提升模型对相似品种的区分能力。

## 环境安装

**本地 Python 版本**: 3.8+
**云端 Python 版本**: 3.9+（推荐）

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac: source venv/bin/activate
                        # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 CLIP + Dassl
#    CLIP 和 Dassl 已作为源码包含在 ../CoOp/ 目录中
cd ../CoOp/dassl && pip install -e .
```

### 云端部署

```bash
# 1. 拉取代码
git clone <your-repo-url>
cd graduation-design

# 2. 安装依赖（根据云端 Python 版本选择合适的 torch 版本）
pip install -r fine_grained_classification/requirements.txt

# 3. 安装 Dassl（完整版，包含 dassl.data 模块）
cd CoOp/dassl
pip install -e .

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
python train.py -d oxford_pets -e 50 -b 16 --shots 1 --trainer DynamicPromptTrainer --device cuda

# 批量运行全部 9 组对比实验（3 方法 × 3 shot）
bash run_experiments.sh cuda

# 汇总结果并生成图表
python collect_results.py --latex --plot
```

### 启动演示

```bash
# Streamlit 界面 → http://localhost:8501
streamlit run demo/pet_classifier_demo.py

# Flask API → http://localhost:5001（zero-shot 模式）
cd web && python app.py

# 使用训练好的模型启动 API（动态提示词推理模式）
python app.py --model ../output_fgd/oxford_pets/DynamicPromptTrainer/shots_1/seed_1/prompt_learner/model-best.pth.tar
```

> 加载训练模型后，Web API 会自动切换为**动态提示词推理模式**：对每张上传图片，
> 通过 `SoftPromptAdapter` 生成图像条件化的提示词再编码，而非使用固定文本模板。
> API 响应中的 `mode` 字段标明当前模式（`"dynamic_prompt"` 或 `"zero_shot"`）。

### 跨设备使用模型

在远端 CUDA 服务器上训练的模型可以直接在本地 Mac (CPU/MPS) 上使用。PyTorch 模型文件
保存的是纯数值张量，与训练设备无关。加载时会自动通过 `map_location` 映射到当前设备。

```bash
# 将远端训练好的模型拷贝到本地
scp remote:/path/to/model-best.pth.tar ./output_fgd/

# 本地 Mac 上直接启动（自动使用 CPU）
python web/app.py --model ./output_fgd/model-best.pth.tar
```

## 核心方法

### 动态提示词 vs CoOp vs CoCoOp

| 特性 | CoOp | CoCoOp | **本系统（DynamicPromptTrainer）** |
|------|------|--------|----------------------------------|
| 提示词类型 | 静态可学习 ctx | 图像条件偏移 | 图像条件偏移 + 难度自适应加权 |
| 核心参数 | `ctx` | `ctx` + `meta_net` | `ctx` + `SoftPromptAdapter` + `DynamicPromptOptimizer` + `class_adaptive_factors` |
| 是否感知图像 | 否 | 是 | 是 |
| 是否感知难度 | 否 | 否 | **是（独有）** |
| 损失函数 | 标准 CE | 标准 CE | **加权 CE（困难样本权重更大）** |
| 提示词层数 | 单层静态 | 单层偏移 | **双层（MLP 偏移 + 类别自适应因子）** |

### 三项核心创新

1. **困难样本自适应加权** — `DifficultyWeightCalculator` 基于特征空间距离和误分类历史，动态调整每个样本的损失权重（误分类样本 ×2，远离类中心样本额外 +0.5×distance）
2. **双层提示词调整** — 在 `SoftPromptAdapter` 的图像条件偏移之上叠加 `class_adaptive_factors`，为不同类别学习不同的提示词缩放
3. **动量更新类别原型** — 训练过程中以 momentum=0.9 持续跟踪每个类别的特征中心，为难度评估提供稳定参考

### 工作流程

```
输入图片 → [冻结] CLIP 视觉编码器 → image_features
         → [可训练] SoftPromptAdapter MLP → ctx 偏移
         → 基础 ctx × class_adaptive_factors + 偏移 → 图片条件提示词
         → [冻结] CLIP 文本编码器 → 初始 logits
         → DifficultyWeightCalculator(原型距离 + 误分类反馈) → 难度权重
         → 加权 CE loss → 反向传播（仅更新 ctx + MLP + adaptive_factors）
```

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| AdaptivePromptLearner | `models/dynamic_prompt.py` | 整合双层提示词生成（偏移 + 类别因子） |
| SoftPromptAdapter | `models/dynamic_prompt.py` | 512→64→512 双层 MLP 生成图像条件偏移 |
| DynamicPromptOptimizer | `models/dynamic_prompt.py` | 难度权重计算 + 类别自适应因子 |
| DifficultyWeightCalculator | `models/dynamic_prompt.py` | 基于原型距离 + 误分类历史的加权 |
| BreedAttributeDatabase | `models/breed_semantic.py` | 37 品种属性库（毛发/面部/体型） |
| SemanticEnhancer | `models/breed_semantic.py` | 多模板语义增强 |

## 训练配置

主要超参数（`configs/dynamic_rn50.yaml`）：

| 参数 | 值 |
|------|-----|
| Backbone | CLIP RN50 |
| 优化器 | SGD (lr=0.001) | 动态提示词专用学习率（原0.002过高） |
| 学习率调度 | Cosine Annealing + warmup 1 epoch |
| 训练轮次 | 50 |
| 可学习 ctx | 4 tokens，初始化为 "a photo of a" |
| SoftPromptAdapter | 512→64→512 MLP |

## 实验结果

| 方法 | 1-shot | 4-shot | 16-shot |
|------|--------|--------|---------|
| Zero-shot CLIP | ~81% | ~81% | ~81% |
| CoOp | 83.3% | 87.9% | 88.3% |
| CoCoOp | 88.0% | 89.0% | 90.0% |
| **Ours (Dynamic)** | 待训练 | 待训练 | 待训练 |

> 训练中，预期 16-shot 达到 ~89-90%。

## 参考资料

- [CLIP: Connecting Text and Images](https://github.com/openai/CLIP)
- [CoOp: Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [CoCoOp: Conditional Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
