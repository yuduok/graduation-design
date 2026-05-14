# 基于动态提示词优化的细粒度分类系统

基于 CLIP + 动态提示词优化的细粒度分类框架，支持多数据集（Oxford-IIIT Pets / Stanford Cars），结合多模态信息（图像特征与文本语义），提升模型对相似类别的区分能力。

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

### Oxford-IIIT Pets 数据集

下载 Oxford-IIIT Pets 数据集，放置到与本项目平级的 `data/` 目录：

```bash
mkdir -p ../data/oxford_pets && cd ../data/oxford_pets

# 下载并解压
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

### Stanford Cars 数据集

下载 Stanford Cars 数据集，放置到与本项目平级的 `data/` 目录：

```bash
mkdir -p ../data/stanford_cars && cd ../data/stanford_cars

# 下载并解压
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat
tar -xzf cars_train.tgz
tar -xzf cars_test.tgz
tar -xzf car_devkit.tgz
```

目录结构：

```
毕业设计/
├── CoOp/                        # CoOp 框架（含 CLIP + Dassl）
├── data/
│   ├── oxford_pets/
│   │   ├── images/              # 7,390 张图片
│   │   ├── annotations/
│   │   └── split_zhou_OxfordPets.json
│   └── stanford_cars/
│       ├── cars_train/
│       ├── cars_test/
│       ├── devkit/
│       └── cars_test_annos_withlabels.mat
└── fine_grained_classification/ # 本项目
```

## 项目结构

```
fine_grained_classification/
├── train.py                     # 训练入口（支持 CoOp/CoCoOp/DynamicPromptTrainer）
├── compare_models.py            # 模型对比评估脚本
├── evaluate.py                  # 评估脚本
├── evaluate_security_ttc.py     # TTC 对抗安全评估脚本（支持双数据集）
├── collect_results.py           # Oxford Pets 实验结果汇总与图表生成
├── collect_results_stanford_cars.py # Stanford Cars 实验结果汇总与图表生成
├── run_experiments.sh           # Oxford Pets 自动化批量实验脚本
├── run_experiments_stanford_cars.sh # Stanford Cars 自动化批量实验脚本
├── modal_train.py               # Modal 云端 GPU 训练脚本
├── modal_upload.py              # Modal 数据集快速上传脚本
├── requirements.txt             # Python 依赖
├── SECURITY.md                  # 安全验证文档
├── configs/
│   ├── dynamic_rn50.yaml        # RN50 训练超参配置（Oxford Pets）
│   ├── dynamic_vitb16.yaml      # ViT-B/16 训练超参配置
│   └── stanford_cars_rn50.yaml  # RN50 训练超参配置（Stanford Cars）
├── models/
│   ├── __init__.py              # 模型模块导出
│   ├── custom_clip.py           # 自定义 CLIP（整合动态提示 + 语义增强）
│   ├── dynamic_prompt.py        # AdaptivePromptLearner + SoftPromptAdapter
│   ├── trainer.py               # DynamicPromptTrainer（注册到 Dassl）
│   ├── breed_semantic.py        # 品种属性库（Oxford Pets 37 品种特征描述）
│   ├── adversarial_defense.py   # 对抗性防御模块（TTC）
│   └── robust_custom_clip.py    # 鲁棒 CLIP 模型
├── demo/
│   ├── pet_classifier_demo.py   # Streamlit 交互演示（Oxford Pets）
│   └── car_classifier_demo.py   # Streamlit 交互演示（Stanford Cars）
├── web/
│   ├── app.py                   # Flask REST API 服务（Oxford Pets，端口 5001）
│   ├── app_stanford_cars.py     # Flask REST API 服务（Stanford Cars，端口 5002）
│   └── static/
│       └── index.html           # 前端演示页面
├── utils/
│   └── helpers.py               # 可视化与度量工具
├── output_fgd/                  # 实验输出目录
│   ├── oxford_pets/
│   │   ├── CoOp/                # CoOp 实验结果
│   │   ├── CoCoOp/              # CoCoOp 实验结果
│   │   ├── DynamicPromptTrainer/ # DynamicPrompt 实验结果
│   │   └── experiment_summary.json # 实验结果摘要
│   └── stanford_cars/
│       ├── CoOp/
│       ├── CoCoOp/
│       ├── DynamicPromptTrainer/
│       └── experiment_summary.json
├── comparison_results/          # 模型对比结果
│   ├── accuracy_comparison.png
│   ├── confidence_distribution.png
│   ├── top_k_accuracies.png
│   └── comparison_summary.json
├── security_results/            # 对抗防御实验结果
│   ├── ttc_dynamic_prompt_oxford_pets.json
│   └── ttc_dynamic_prompt_stanford_cars.json
├── thesis/                      # 毕业论文 LaTeX 源文件
│   ├── main.tex
│   ├── chapters/
│   └── thesis_figures/
└── thesis_figures/              # 论文图表
```

## 快速开始

### 训练模型

#### Oxford Pets

```bash
# 单个模型训练（可自定义 trainer / shots / epochs）
python train.py -d oxford_pets -t DynamicPromptTrainer --shots 16 -e 60 --device cuda

# 批量运行对比实验
bash run_experiments.sh                    # 默认: 100,80,60,40,20 epochs
bash run_experiments.sh cuda 16 "100,80,60,40,20"  # 自定义 epochs 列表
```

#### Stanford Cars

```bash
# 单个模型训练
python train.py -d stanford_cars -t DynamicPromptTrainer --shots 16 -e 60 --device cuda

# 批量运行对比实验
bash run_experiments_stanford_cars.sh                    # 默认: 100,80,60,40,20 epochs
bash run_experiments_stanford_cars.sh cuda 16 "100,80,60,40,20"  # 自定义 epochs 列表
```

**参数说明**：
```
  -t, --trainer    模型类型: DynamicPromptTrainer, CoOp, CoCoOp
  --shots          Few-shot 数量: 1, 2, 4, 8, 16
  -e, --epochs     训练轮次
  -d, --dataset    数据集: oxford_pets, stanford_cars
  --device         设备: cuda, cpu
  -b, --batch-size 批大小（默认16）
```

#### 模型对比评估

```bash
# Oxford Pets
python compare_models.py --dataset oxford_pets --backbone RN50

# Stanford Cars
python compare_models.py --dataset stanford_cars --backbone RN50

# 收集结果
python collect_results.py --base-dir output_fgd/oxford_pets    # Oxford Pets
python collect_results_stanford_cars.py --base-dir output_fgd/stanford_cars  # Stanford Cars
python collect_results.py --latex --plot     # 生成 LaTeX 表格和图表
```

**对比脚本功能**：

- 准确率对比（整体准确率）
- Top-K 准确率（Top-1, Top-3, Top-5）
- 置信度分布分析（正确/错误预测的置信度对比）
- 可视化图表输出：
  - `accuracy_comparison.png` - 准确率柱状图对比
  - `confidence_distribution.png` - 预测置信度分布对比
  - `top_k_accuracies.png` - Top-K 准确率对比
- `comparison_summary.json` - 结果汇总文件

#### TTC 安全评估

```bash
# Oxford Pets TTC 评估
python evaluate_security_ttc.py \
  --dataset oxford_pets \
  --shots 16 \
  --seed 1 \
  --device cuda

# Stanford Cars TTC 评估
python evaluate_security_ttc.py \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --device cuda
```

### 启动演示

#### 1. Flask API 服务（推荐）

##### Oxford Pets API

```bash
cd web

# 使用训练好的模型启动（动态提示词推理模式）
python app.py --shot 16 --port 5001

# 或指定具体模型路径
python app.py --model ../output_fgd/oxford_pets/DynamicPromptTrainer/shots_16/seed_1/prompt_learner/model.pth.tar-20 --port 5001

# Zero-shot 模式（不加载训练模型）
python app.py --port 5001
```

API 服务启动后：

- 前端页面：`http://localhost:5001/`
- API 文档：见页面底部或访问 `/api/health`

**Oxford Pets API 端点**：

| 端点                  | 方法   | 功能                                |
| :------------------ | :--- | :-------------------------------- |
| `/api/classify`     | POST | 单图分类（支持对比模式）                      |
| `/api/compare`      | POST | 多模型对比（DynamicPrompt vs Zero-shot） |
| `/api/breeds`       | GET  | 获取所有品种列表                          |
| `/api/breed/<name>` | GET  | 获取品种详情（属性/提示词模板）                  |
| `/api/experiments`  | GET  | 获取实验结果摘要                          |
| `/api/model-info`   | GET  | 获取当前模型信息                          |
| `/api/health`       | GET  | 健康检查                              |

##### Stanford Cars API

```bash
cd web

# 使用训练好的模型启动（动态提示词推理模式）
python app_stanford_cars.py --shot 16 --port 5002

# 或指定具体模型路径
python app_stanford_cars.py --model ../output_fgd/stanford_cars/DynamicPromptTrainer/shots_16/seed_1/prompt_learner/model.pth.tar-20 --port 5002

# Zero-shot 模式（不加载训练模型）
python app_stanford_cars.py --port 5002
```

API 服务启动后：

- 前端页面：`http://localhost:5002/`
- API 文档：见页面底部或访问 `/api/health`

**Stanford Cars API 端点**：

| 端点                  | 方法   | 功能                                |
| :------------------ | :--- | :-------------------------------- |
| `/api/classify`     | POST | 单图分类（支持对比模式）                      |
| `/api/compare`      | POST | 多模型对比（DynamicPrompt vs Zero-shot） |
| `/api/models`       | GET  | 获取所有车型列表                          |
| `/api/model/<name>` | GET  | 获取车型详情（年份/品牌）                      |
| `/api/experiments`  | GET  | 获取实验结果摘要                          |
| `/api/model-info`   | GET  | 获取当前模型信息                          |
| `/api/health`       | GET  | 健康检查                              |

#### 2. Streamlit 交互演示

```bash
# Oxford Pets 演示 → http://localhost:8501
streamlit run demo/pet_classifier_demo.py

# Stanford Cars 演示 → http://localhost:8501
streamlit run demo/car_classifier_demo.py
```

功能标签页：

- **🔍 分类演示** - 上传图片，选择提示词模式，对比 Zero-shot
- **📊 研究结果** - 实验数据表格、方法对比、自适应 Epoch 策略
- **📚 知识库** - 类别属性浏览（Oxford Pets: 毛发/面部/体型/性格；Stanford Cars: 年份/品牌/型号）
- **🔌 API 文档** - 完整接口说明

> 加载训练模型后，Web API 会自动切换为**动态提示词推理模式**：对每张上传图片，
> 通过 `SoftPromptAdapter` 生成图像条件化的提示词再编码，而非使用固定文本模板。
> API 响应中的 `mode` 字段标明当前模式（`"dynamic_prompt"` 或 `"zero_shot"`）。

### 跨设备使用模型

在远端 CUDA 服务器上训练的模型可以直接在本地 Mac (CPU/MPS) 上使用。PyTorch 模型文件
保存的是纯数值张量，与训练设备无关。加载时会自动通过 `map_location` 映射到当前设备。

```bash
# 将远端训练好的模型拷贝到本地
scp remote:/path/to/model-best.pth.tar ./output_fgd/

# 本地 Mac 上直接启动 Oxford Pets（自动使用 CPU）
python web/app.py --model ./output_fgd/oxford_pets/model-best.pth.tar

# 本地 Mac 上直接启动 Stanford Cars（自动使用 CPU）
python web/app_stanford_cars.py --model ./output_fgd/stanford_cars/model-best.pth.tar
```

## 核心方法

### 动态提示词 vs CoOp vs CoCoOp

| 特性     | CoOp      | CoCoOp             | **本系统（DynamicPromptTrainer）**                                                         |
| :----- | :-------- | :----------------- | :------------------------------------------------------------------------------------ |
| 提示词类型  | 静态可学习 ctx | 图像条件偏移             | 图像条件偏移 + 可学习难度加权                                                                      |
| 核心参数   | `ctx`     | `ctx` + `meta_net` | `ctx` + `SoftPromptAdapter` + `DifficultyWeightCalculator` + `class_adaptive_factors` |
| 是否感知图像 | 否         | 是                  | 是                                                                                     |
| 是否感知难度 | 否         | 否                  | **是（可学习）**                                                                            |
| 损失函数   | 标准 CE     | 标准 CE              | **加权 CE（困难样本权重可学习）**                                                                  |
| 提示词层数  | 单层静态      | 单层偏移               | **双层（MLP 偏移 + 类别自适应因子）**                                                              |

### 核心创新

1. **可学习难度加权** — `DifficultyWeightCalculator` 使用可学习温度参数，让模型自动学习何时关注困难样本：
   - `temperature`: 控制权重对置信度变化的敏感度
   - `confidence_scale`: 控制权重放大程度
   - `wrong_weight`: 错误预测样本的额外权重
   - 移除 `detach()`，让梯度回传使这些参数可学习
2. **双层提示词调整** — 在 `SoftPromptAdapter` 的图像条件偏移之上叠加 `class_adaptive_factors`，为不同类别学习不同的提示词缩放
3. **两阶段前向传播** — 训练时先计算基础 logits 获取预测，再用预测计算难度权重生成自适应提示词
4. **品种语义增强** — 37 品种属性数据库（毛发/面部/体型/性格），支持多模板文本生成
5. **多数据集支持** — 支持 Oxford Pets（37类猫狗）和 Stanford Cars（196类汽车）

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

| 模块                         | 文件                         | 说明                                    |
| :------------------------- | :------------------------- | :------------------------------------ |
| AdaptivePromptLearner      | `models/dynamic_prompt.py` | 整合双层提示词生成（偏移 + 类别因子）                  |
| SoftPromptAdapter          | `models/dynamic_prompt.py` | 512→32→512 MLP 生成图像条件偏移（vis_dim//16） |
| DifficultyWeightCalculator | `models/dynamic_prompt.py` | 可学习难度权重计算器（温度参数）                      |
| DynamicPromptOptimizer     | `models/dynamic_prompt.py` | 难度权重计算 + 类别自适应因子                      |
| BreedAttributeDatabase     | `models/breed_semantic.py` | 37 品种属性库（毛发/面部/体型）                    |
| SemanticEnhancer           | `models/breed_semantic.py` | 多模板语义增强                               |
| TextEncoder                | `models/custom_clip.py`    | CLIP 文本编码器封装                          |
| DynamicPromptTrainer       | `models/trainer.py`        | Dassl 注册的训练器                          |
| CarClassifierAPI           | `web/app_stanford_cars.py` | Stanford Cars Flask API（分批编码避免OOM）    |
| PetClassifierAPI           | `web/app.py`               | Oxford Pets Flask API                      |

## 训练配置

主要超参数（`configs/dynamic_rn50.yaml`）：

| 参数                | 值                                 | <br />           |
| :---------------- | :-------------------------------- | :--------------- |
| Backbone          | CLIP RN50                         | <br />           |
| 优化器               | SGD (lr=0.002)                    | 与 CoCoOp 保持一致    |
| 学习率调度             | Cosine Annealing + warmup 1 epoch | <br />           |
| 训练轮次              | 50-100（根据 shot 数）                 | <br />           |
| 可学习 ctx           | 4 tokens，初始化为 "a photo of a"      | <br />           |
| SoftPromptAdapter | 512→32→512 MLP（vis_dim//16）      | <br />           |
| 难度权重              | 可学习温度参数，初始 temperature=0.1        | <br />           |
| 精度                | fp16                              | 与 CoCoOp 一致，加速训练 |

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
\|- DynamicPromptTrainer 在 2/4/8/16-shot 上均优于 CoCoOp
\|- DynamicPromptTrainer 在 1-shot 上略低于 CoCoOp（-0.8%）
\|- **整体显著优于 CoOp**（1-shot 提升 +5.1%，4-shot 提升 +2.3%，16-shot 提升 +0.5%）
\|- **最佳结果**：16-shot 达到 **89.8%**，超越所有基线方法

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

### 2026-05-14：多数据集支持（Stanford Cars）

**新增功能**：

1. **Stanford Cars 数据集支持** — 新增 196 类汽车型号细粒度分类
   - 新增 `configs/stanford_cars_rn50.yaml` 配置文件
   - 新增 `run_experiments_stanford_cars.sh` 批量训练脚本
   - 支持 `train.py -d stanford_cars` 直接训练
2. **汽车分类 Demo** — 新增 `demo/car_classifier_demo.py`
   - Streamlit 交互界面，支持 196 类汽车型号识别
   - 车型知识库（年份/品牌/型号）
3. **Stanford Cars Flask API** — 新增 `web/app_stanford_cars.py`
   - 独立的 REST API 服务（端口 5002）
   - 分批编码文本特征（每批 50 类），避免 196 类显存溢出
   - 车型信息自动解析（年份/品牌）
   - API 端点：`/api/models`、`/api/model/<name>`、`/api/classify`、`/api/compare`
4. **TTC 安全评估扩展** — `evaluate_security_ttc.py` 支持双数据集
   - `--dataset oxford_pets` / `--dataset stanford_cars`
   - 自动根据数据集命名输出文件
   - `forward()` 方法支持分批编码，避免 OOM
5. **结果收集脚本** — 新增 `collect_results_stanford_cars.py`
   - 自动扫描 `output_fgd/stanford_cars/` 目录
   - 生成对比表格、LaTeX 代码、柱状图、学习曲线
6. **安全文档更新** — `SECURITY.md` 新增 Stanford Cars 使用说明
7. **项目文档更新** — 根目录 `README.md` 和 `PROJECT_STATUS.md` 全面更新

### 2026-04-13：自适应 Epoch 策略 + 最终实验结果

**训练策略优化**：

1. **自适应 Epoch 配置** — Few-shot 数越少，训练 Epoch 越多：
   - 1-shot → 100 epochs（最需要充分学习）
   - 2-shot → 80 epochs
   - 4-shot → 60 epochs
   - 8-shot → 40 epochs
   - 16-shot → 20 epochs（数据充足，避免过拟合）

**最终实验结果**：

| Method               | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot   |
| :------------------- | :----- | :----- | :----- | :----- | :-------- |
| DynamicPromptTrainer | 86.0%  | 85.9%  | 89.5%  | 89.2%  | **89.8%** |
| CoCoOp               | 86.8%  | 83.5%  | 89.1%  | 88.6%  | 89.6%     |
| CoOp                 | 80.9%  | 82.6%  | 87.2%  | 86.8%  | 89.3%     |

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

| Bug     | 原因                      | 修复                                          |
| :------ | :---------------------- | :------------------------------------------ |
| 权重范围无限制 | 困难权重可能达到 3-4 倍，对小样本冲击过大 | 添加 `weight = max(0.5, min(2.0, weight))` 限制 |
| 学习率过高   | 0.002 对动态提示词参数过大        | 降为 0.001，后调整为 0.002                         |

### 2026-03-27：核心模块激活修复

| Bug                            | 原因                     | 修复            |
| :----------------------------- | :--------------------- | :------------ |
| 难度权重从未计算                       | `predictions` 始终为 None | 改为两阶段前向       |
| `class_adaptive_factors` 未参与计算 | 未与 ctx 相乘              | 在 forward 中相乘 |

## 参考资料

- [CLIP: Connecting Text and Images](https://github.com/openai/CLIP)
- [CoOp: Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [CoCoOp: Conditional Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [CLIP-Test-time-Counterattacks](https://github.com/ldhl-hk/CLIP-Test-time-Counterattacks)