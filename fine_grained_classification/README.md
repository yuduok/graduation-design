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
# 单个模型训练（可自定义 trainer / shots / epochs）
python train.py -d oxford_pets -t DynamicPromptTrainer --shots 16 -e 60 --device cuda

# 参数说明：
#   -t, --trainer    模型类型: DynamicPromptTrainer, CoOp, CoCoOp
#   --shots          Few-shot 数量: 1, 2, 4, 8, 16
#   -e, --epochs     训练轮次
#   -d, --dataset    数据集: oxford_pets
#   --device         设备: cuda, cpu
#   -b, --batch-size 批大小（默认16）

# 批量运行对比实验
bash run_experiments.sh                    # 默认: 100,90,80,70,50 epochs
bash run_experiments.sh cuda 16 "100,90,80,70,50"  # 自定义 epochs 列表

# 收集结果
python collect_results.py                   # 按 shot 对比
python collect_results.py --epochs           # 按 epochs 对比（支持不同 epoch 设置的结果）
python collect_results.py --latex --plot     # 生成 LaTeX 表格和图表
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
| 提示词类型 | 静态可学习 ctx | 图像条件偏移 | 图像条件偏移 + 可学习难度加权 |
| 核心参数 | `ctx` | `ctx` + `meta_net` | `ctx` + `SoftPromptAdapter` + `DifficultyWeightCalculator` + `class_adaptive_factors` |
| 是否感知图像 | 否 | 是 | 是 |
| 是否感知难度 | 否 | 否 | **是（可学习）** |
| 损失函数 | 标准 CE | 标准 CE | **加权 CE（困难样本权重可学习）** |
| 提示词层数 | 单层静态 | 单层偏移 | **双层（MLP 偏移 + 类别自适应因子）** |

### 核心创新

1. **可学习难度加权** — `DifficultyWeightCalculator` 使用可学习温度参数，让模型自动学习何时关注困难样本：
   - `temperature`: 控制权重对置信度变化的敏感度
   - `confidence_scale`: 控制权重放大程度
   - `wrong_weight`: 错误预测样本的额外权重
   - 移除 `detach()`，让梯度回传使这些参数可学习

2. **双层提示词调整** — 在 `SoftPromptAdapter` 的图像条件偏移之上叠加 `class_adaptive_factors`，为不同类别学习不同的提示词缩放

3. **两阶段前向传播** — 训练时先计算基础 logits 获取预测，再用预测计算难度权重生成自适应提示词

### 工作流程

```
输入图片 → [冻结] CLIP 视觉编码器 → image_features
         → [可训练] SoftPromptAdapter MLP → ctx 偏移
         → 基础 ctx × class_adaptive_factors + 偏移 → 图片条件提示词
         → [冻结] CLIP 文本编码器 → 初始 logits
         → DifficultyWeightCalculator(可学习温度) → 难度权重
         → 加权 CE loss → 反向传播（更新 ctx + MLP + 温度参数）
```

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| AdaptivePromptLearner | `models/dynamic_prompt.py` | 整合双层提示词生成（偏移 + 类别因子） |
| SoftPromptAdapter | `models/dynamic_prompt.py` | 512→32→512 MLP 生成图像条件偏移（vis_dim//16） |
| DifficultyWeightCalculator | `models/dynamic_prompt.py` | 可学习难度权重计算器（温度参数） |
| DynamicPromptOptimizer | `models/dynamic_prompt.py` | 难度权重计算 + 类别自适应因子 |
| BreedAttributeDatabase | `models/breed_semantic.py` | 37 品种属性库（毛发/面部/体型） |
| SemanticEnhancer | `models/breed_semantic.py` | 多模板语义增强 |

## 训练配置

主要超参数（`configs/dynamic_rn50.yaml`）：

| 参数 | 值 |
|------|-----|
| Backbone | CLIP RN50 |
| 优化器 | SGD (lr=0.002) | 与 CoCoOp 保持一致 |
| 学习率调度 | Cosine Annealing + warmup 1 epoch |
| 训练轮次 | 50-100（根据 shot 数） |
| 可学习 ctx | 4 tokens，初始化为 "a photo of a" |
| SoftPromptAdapter | 512→32→512 MLP（vis_dim//16） |
| 难度权重 | 可学习温度参数，初始 temperature=0.1 |
| 精度 | fp16 | 与 CoCoOp 一致，加速训练 |

## 实验结果

### 2026-04-13 实验结果（OxfordPets）- 自适应 Epoch 策略

采用自适应训练策略：**Few-shot 数越少，训练 Epoch 越多**（1-shot=100ep, 2-shot=80ep, 4-shot=60ep, 8-shot=40ep, 16-shot=20ep）

|| 方法 | 1-shot (ep100) | 2-shot (ep80) | 4-shot (ep60) | 8-shot (ep40) | 16-shot (ep20) |
||------|----------------|----------------|----------------|----------------|-----------------|
|| Zero-shot CLIP | ~81% | ~81% | ~81% | ~81% | ~81% |
|| CoOp | 80.9% | 82.6% | 87.2% | 86.8% | 89.3% |
|| CoCoOp | 86.8% | 83.5% | 89.1% | 88.6% | 89.6% |
|| **Ours (Dynamic)** | **86.0%** | **85.9%** | **89.5%** | **89.2%** | **89.8%** |

**结论**：
|- DynamicPromptTrainer 在 2/4/8/16-shot 上均优于 CoCoOp
|- DynamicPromptTrainer 在 1-shot 上略低于 CoCoOp（-0.8%）
|- **整体显著优于 CoOp**（1-shot 提升 +5.1%，4-shot 提升 +2.3%，16-shot 提升 +0.5%）
|- **最佳结果**：16-shot 达到 **89.8%**，超越所有基线方法

### 对比分析（Ours vs CoCoOp）

|| Shot | Ours | CoCoOp | 差异 | 结论 |
||------|------|--------|------|------|
|| 1-shot | 86.0% | 86.8% | -0.8% | 略低，但显著优于 CoOp |
|| 2-shot | 85.9% | 83.5% | +2.4% | **大幅领先** |
|| 4-shot | 89.5% | 89.1% | +0.4% | **领先** |
|| 8-shot | 89.2% | 88.6% | +0.6% | **领先** |
|| 16-shot | 89.8% | 89.6% | +0.2% | **最佳结果** |

> 2026-04-13 更新：使用自适应 Epoch 策略，重新训练后结果显著改善。可学习难度权重模块生效。

## 更新日志

### 2026-04-13：自适应 Epoch 策略 + 最终实验结果

**训练策略优化**：
1. **自适应 Epoch 配置** — Few-shot 数越少，训练 Epoch 越多：
   - 1-shot → 100 epochs（最需要充分学习）
   - 2-shot → 80 epochs
   - 4-shot → 60 epochs
   - 8-shot → 40 epochs
   - 16-shot → 20 epochs（数据充足，避免过拟合）

**最终实验结果**：
| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|--------|--------|--------|--------|--------|---------|
| DynamicPromptTrainer | 86.0% | 85.9% | 89.5% | 89.2% | **89.8%** |
| CoCoOp | 86.8% | 83.5% | 89.1% | 88.6% | 89.6% |
| CoOp | 80.9% | 82.6% | 87.2% | 86.8% | 89.3% |

**关键发现**：
- DynamicPromptTrainer 在 2/4/8/16-shot 上全面超越 CoCoOp
- 最佳结果：16-shot 达到 **89.8%**
- 1-shot 差距缩小至 -0.8%（之前为 -1.9%）

### 2026-04-02：可学习难度权重优化

**修复内容**：
1. **两阶段前向传播修复** — `CustomCLIPDynamic.forward()` 现在正确实现两阶段前向：
   - 阶段1：使用基础提示词计算初始 logits（无梯度）
   - 阶段2：使用初始 logits 计算难度权重，生成自适应提示词

2. **SoftPromptAdapter 优化** — 隐藏层维度改为 `vis_dim // 16`（32 for RN50），与 CoCoOp 一致

3. **可学习难度权重** — `DifficultyWeightCalculator` 改为 `nn.Module`，添加可学习参数：
   - `temperature`: 温度参数（初始 0.1）
   - `confidence_scale`: 置信度缩放（初始 2.0）
   - `wrong_weight`: 错误预测权重（初始 2.0）
   - 移除 `detach()`，让梯度回传

4. **超参数调整** — 与 CoCoOp 保持一致：
   - LR: 0.002
   - PREC: fp16

### 2026-03-28：训练稳定性修复

| Bug | 原因 | 修复 |
|-----|------|------|
| 权重范围无限制 | 困难权重可能达到 3-4 倍，对小样本冲击过大 | 添加 `weight = max(0.5, min(2.0, weight))` 限制 |
| 学习率过高 | 0.002 对动态提示词参数过大 | 降为 0.001，后调整为 0.002 |

### 2026-03-27：核心模块激活修复

| Bug | 原因 | 修复 |
|-----|------|------|
| 难度权重从未计算 | `predictions` 始终为 None | 改为两阶段前向 |
| `class_adaptive_factors` 未参与计算 | 未与 ctx 相乘 | 在 forward 中相乘 |

## 参考资料

- [CLIP: Connecting Text and Images](https://github.com/openai/CLIP)
- [CoOp: Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [CoCoOp: Conditional Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
