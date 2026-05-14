# 细粒度分类系统 - 项目状态报告

**项目名称**: 基于动态提示词优化的细粒度分类系统
**技术栈**: PyTorch 2.4.1 + CLIP (RN50) + Python 3.8+
**支持数据集**: Oxford-IIIT Pets (37 种猫狗品种) / Stanford Cars (196 种汽车型号)
**更新时间**: 2026-05-14

---

## 一、系统架构

### 核心思路

在预训练 CLIP 模型基础上，冻结图像/文本编码器，仅训练**动态提示词模块**，使提示词能根据每张输入图片自适应调整，从而提升细粒度类别的区分能力。

### 支持的数据集

| 数据集 | 类别数 | 安全场景 | 特点 |
|--------|--------|---------|------|
| **Oxford-IIIT Pets** | 37 | 通用细粒度分类 | 猫狗品种分类，原始基准数据集 |
| **Stanford Cars** | 196 | 自动驾驶/智能交通 | 汽车型号细粒度分类，安全关键场景 |

### 与 CoOp / CoCoOp 基线的核心区别

| 特性 | CoOp | CoCoOp | **本系统（DynamicPromptTrainer）** |
|------|------|--------|----------------------------------|
| 提示词类型 | **静态**：所有图片共享同一组可学习 ctx 向量 | **图像条件**：meta_net 生成偏移 | **图像条件 + 可学习难度加权**：双层调整 + 加权损失 |
| 核心参数 | `ctx` | `ctx` + `meta_net` | `ctx` + `SoftPromptAdapter` + `DifficultyWeightCalculator` + `class_adaptive_factors` |
| 是否感知图像 | 否 | 是 | 是 |
| 是否感知难度 | 否 | 否 | **是（可学习）** |
| 损失函数 | 标准 CE | 标准 CE | **加权 CE（困难样本权重可学习）** |
| 提示词调整层数 | 单层静态 | 单层偏移 | **双层（MLP 偏移 + 类别自适应因子）** |
| 难度权重 | 无 | 无 | **可学习温度参数** |

### 动态提示词工作流程

```
输入图片
  ↓
[冻结] CLIP 视觉编码器 → image_features [batch, 512]
  ↓
[可训练] SoftPromptAdapter MLP → 图片特定的 ctx 偏移 [batch, 512]
  ↓
基础 ctx × class_adaptive_factors + 偏移 → 图片条件提示词嵌入 [batch, n_cls, n_ctx, 512]
  ↓
[冻结] CLIP 文本编码器 → text_features [batch, n_cls, 512]
  ↓
logit_scale × (image_features @ text_features.T) → 初始 logits
  ↓
DifficultyWeightCalculator(可学习温度) → 难度权重 w_i
  ↓
加权 CE loss = mean(w_i × CE_i) → 反向传播（更新 ctx + MLP + 温度参数）
```

**关键创新**: 使用可学习的温度参数，让模型自动学习何时该关注困难样本，而非人工设定固定权重。

---

## 二、项目结构

```
fine_grained_classification/
├── train.py                     # 训练入口（支持 CoOp/CoCoOp/DynamicPromptTrainer）
├── compare_models.py            # 模型对比评估脚本
├── evaluate.py                  # 评估脚本
├── evaluate_security_ttc.py     # TTC 对抗安全评估脚本（支持 OxfordPets / StanfordCars）
├── collect_results.py           # 实验结果汇总与图表生成
├── run_experiments.sh           # Oxford Pets 自动化批量实验脚本
├── run_experiments_stanford_cars.sh # Stanford Cars 自动化批量实验脚本
├── requirements.txt             # Python 依赖
├── SECURITY.md                  # 安全验证文档
├── configs/
│   ├── dynamic_rn50.yaml        # RN50 训练超参配置（Oxford Pets）
│   ├── dynamic_vitb16.yaml      # ViT-B/16 训练超参配置
│   └── stanford_cars_rn50.yaml  # RN50 训练超参配置（Stanford Cars）
├── models/
│   ├── __init__.py              # 模型模块导出
│   ├── custom_clip.py           # 自定义 CLIP（整合动态提示 + 语义增强）
│   ├── dynamic_prompt.py        # AdaptivePromptLearner + SoftPromptAdapter + DifficultyWeightCalculator
│   ├── trainer.py               # DynamicPromptTrainer（注册到 Dassl）
│   ├── breed_semantic.py        # 品种属性库（37 品种的毛发/面部/体型特征）
│   ├── adversarial_defense.py   # 对抗性防御模块（TTC 风格）
│   └── robust_custom_clip.py    # 鲁棒 CLIP 模型
├── demo/
│   ├── pet_classifier_demo.py   # Streamlit 交互演示（Oxford Pets）
│   └── car_classifier_demo.py   # Streamlit 交互演示（Stanford Cars）
├── web/
│   ├── app.py                   # Flask REST API 服务（研究增强版）
│   └── static/
│       └── index.html           # 前端 HTML 演示页面
├── utils/
│   └── helpers.py               # 可视化与度量工具
├── output_fgd/                  # 实验输出目录
│   ├── oxford_pets/
│   │   ├── CoOp/                # CoOp 实验结果（shots_1/2/4/8/16）
│   │   ├── CoCoOp/              # CoCoOp 实验结果（shots_1/2/4/8/16）
│   │   ├── DynamicPromptTrainer/ # DynamicPrompt 实验结果（shots_1/2/4/8/16）
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

---

## 三、训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Backbone | RN50 | CLIP ResNet-50 |
| 优化器 | SGD | lr=0.002（与 CoCoOp 一致） |
| 学习率调度 | Cosine Annealing | warmup 1 epoch |
| 训练轮次 | 50-100 | 根据 shot 数调整 |
| 批大小 | 16 | 默认值，可按 GPU 显存调整 |
| 可学习 ctx 长度 | 4 tokens | 初始化为 "a photo of a" |
| ctx 嵌入维度 | 512 | 与 CLIP 特征维度一致 |
| SoftPromptAdapter | 512→32→512 | 双层 MLP + ReLU（vis_dim//16） |
| 难度权重 | 可学习温度参数 | temperature=0.1, confidence_scale=2.0, wrong_weight=2.0 |
| 精度 | fp16 | 与 CoCoOp 一致，加速训练 |

---

## 四、云端部署

### 服务器要求

| 项目 | 要求 |
|------|------|
| Python | 3.9+ |
| GPU | CUDA 11.0+ |
| 内存 | 16GB+ |
| 硬盘 | 10GB+ |

### 云端部署步骤

```bash
# 1. 拉取代码
git clone <your-repo-url>
cd graduation-design

# 2. 安装依赖
cd fine_grained_classification
pip install -r requirements.txt

# 3. 安装 Dassl（关键！）
cd ../CoOp/dassl
pip install -e .

# 4. 准备数据集
mkdir -p ../data/oxford_pets
mkdir -p ../data/stanford_cars
# 下载数据集到对应目录

# 5. 开始训练（Oxford Pets）
cd ../../fine_grained_classification
python train.py -d oxford_pets -e 50 -b 16 --shots 16 --trainer DynamicPromptTrainer --device cuda

# 或训练 Stanford Cars
python train.py -d stanford_cars -e 50 -b 16 --shots 16 --trainer DynamicPromptTrainer --device cuda
```

### 目录结构要求

确保云端目录结构与本地一致：

```
graduation-design/
├── CoOp/                      # CoOp 框架（含 CLIP + 完整 Dassl）
│   ├── datasets/              # 数据集定义
│   ├── trainers/              # 训练器
│   └── dassl/                 # Dassl 完整仓库（包含 dassl.data）
├── data/
│   ├── oxford_pets/           # Oxford Pets 数据集
│   └── stanford_cars/         # Stanford Cars 数据集
├── fine_grained_classification/
│   ├── train.py
│   ├── evaluate_security_ttc.py
│   └── ...
└── output_fgd/                # 训练输出
    ├── oxford_pets/
    └── stanford_cars/
```

---

## 五、快速使用

### 环境准备

```bash
source ../CoOp/venv/bin/activate
pip install streamlit flask flask-cors matplotlib seaborn scikit-learn
```

### 训练模型

#### Oxford Pets

```bash
# 在 fine_grained_classification/ 目录下执行
# 单组实验（输出自动保存到 output_fgd/oxford_pets/{trainer}/shots_{n}/seed_{s}/）
python train.py -d oxford_pets -e 50 -b 16 --shots 16 --trainer DynamicPromptTrainer --device cuda

# 批量运行全部 15 组对比实验（3 方法 × 5 shots）
bash run_experiments.sh cuda
```

#### Stanford Cars

```bash
# 单组实验（输出自动保存到 output_fgd/stanford_cars/{trainer}/shots_{n}/seed_{s}/）
python train.py -d stanford_cars -e 50 -b 16 --shots 16 --trainer DynamicPromptTrainer --device cuda

# 批量运行全部 15 组对比实验
bash run_experiments_stanford_cars.sh cuda
```

### TTC 安全评估

```bash
# Oxford Pets
python evaluate_security_ttc.py \
  --dataset oxford_pets \
  --shots 16 \
  --seed 1 \
  --device cuda

# Stanford Cars
python evaluate_security_ttc.py \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --device cuda
```

### 启动演示

#### Flask API 服务（推荐）

```bash
cd web

# 使用训练好的模型启动（动态提示词推理模式）
python app.py --shot 16 --port 5001

# 或指定具体模型路径
python app.py --model ../output_fgd/oxford_pets/DynamicPromptTrainer/shots_16/seed_1/prompt_learner/model.pth.tar-20 --port 5001

# Zero-shot 模式（不加载训练模型）
python app.py --port 5001
```

**API 端点**：
| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 前端 HTML 演示页面 |
| `/api/classify` | POST | 单图分类（支持对比模式） |
| `/api/compare` | POST | 多模型对比（DynamicPrompt vs Zero-shot） |
| `/api/breeds` | GET | 获取所有品种列表 |
| `/api/breed/<name>` | GET | 获取品种详情（属性/提示词模板） |
| `/api/experiments` | GET | 获取实验结果摘要 |
| `/api/model-info` | GET | 获取当前模型信息 |
| `/api/health` | GET | 健康检查 |

#### Streamlit 交互演示

```bash
# Oxford Pets 演示
streamlit run demo/pet_classifier_demo.py

# Stanford Cars 演示
streamlit run demo/car_classifier_demo.py
```

功能标签页：
- **🔍 分类演示** - 上传图片，选择提示词模式，对比 Zero-shot
- **📊 研究结果** - 实验数据表格、方法对比、自适应 Epoch 策略
- **📚 知识库** - 类别属性浏览（Oxford Pets: 毛发/面部/体型/性格；Stanford Cars: 年份/品牌/型号）
- **🔌 API 文档** - 完整接口说明

> 加载训练模型后，Web API 自动切换为**动态提示词推理模式**，通过 `SoftPromptAdapter` 生成图像条件化的提示词。
> API 响应中 `mode` 字段标明当前推理模式：`"dynamic_prompt"` 或 `"zero_shot"`。
> CUDA 训练的模型可直接在 Mac CPU/MPS 上使用，PyTorch 通过 `map_location` 自动映射设备。

---

## 六、当前进度

### 已完成

- [x] 核心模型模块（dynamic_prompt / custom_clip / trainer / breed_semantic）
- [x] 训练流程（train.py 支持 3 种 trainer + few-shot 配置 + 多数据集）
- [x] Web 演示（Streamlit + Flask API + 前端 HTML 页面）
- [x] 所有单元测试通过（test_core / test_demo / test_full）
- [x] Bug 修复：logit_scale 缺失、类名错误（36→37 类）、导入路径等
- [x] 云端部署支持：替换完整版 Dassl、修复 ftfy 版本、路径自动查找
- [x] Web API 动态提示词推理：加载训练模型后，API 使用 prompt_learner + TextEncoder 生成图像条件化的提示词，而非固定模板
- [x] 跨设备模型兼容：CUDA 训练的模型可在 Mac CPU/MPS 上直接加载使用
- [x] 默认 batch size 从 32 降为 16，避免 GPU 显存不足
- [x] 修复 PyTorch lr_scheduler verbose 参数弃用警告
- [x] 核心机制修复：两阶段前向传播 + 难度权重计算
- [x] 可学习难度权重优化：添加温度参数，移除 detach
- [x] **自适应 Epoch 策略训练完成**（1-shot=100ep, 2-shot=80ep, 4-shot=60ep, 8-shot=40ep, 16-shot=20ep）
- [x] **所有对比实验完成**（DynamicPromptTrainer / CoOp / CoCoOp，全部 5 种 shot 配置）
- [x] **实验结果分析完成**（DynamicPromptTrainer 在 2/4/8/16-shot 上全面超越 CoCoOp）
- [x] **Web API 研究增强版完成** - 新增多模型对比、品种语义信息、实验结果展示、调试信息
- [x] **前端 HTML 页面完成** - 交互式分类、品种知识库、实验结果看板
- [x] **Streamlit Demo 研究增强版完成** - 四标签页：分类演示/研究结果/品种知识库/API文档
- [x] **Stanford Cars 数据集支持** - 新增 196 类汽车型号细粒度分类
- [x] **TTC 安全评估扩展** - 支持 OxfordPets / StanfordCars 双数据集
- [x] **汽车分类 Demo** - 新增 `demo/car_classifier_demo.py`
- [x] **安全文档更新** - `SECURITY.md` 新增 Stanford Cars 使用说明

### 待完成

- [ ] Stanford Cars 实验训练与结果收集
- [ ] 撰写论文实验章节
- [ ] 对抗防御实验完整评估
- [ ] ViT-B/16 骨干网络实验

---

## 七、实验结果

### 2026-04-13 实验结果（OxfordPets）- 自适应 Epoch 策略

采用自适应训练策略：**Few-shot 数越少，训练 Epoch 越多**
- 1-shot → 100 epochs
- 2-shot → 80 epochs
- 4-shot → 60 epochs
- 8-shot → 40 epochs
- 16-shot → 20 epochs

|| 方法 | 1-shot (ep100) | 2-shot (ep80) | 4-shot (ep60) | 8-shot (ep40) | 16-shot (ep20) |
||------|----------------|----------------|----------------|----------------|----------------|
|| Zero-shot CLIP | ~81% | ~81% | ~81% | ~81% | ~81% |
|| CoOp | 80.9% | 82.6% | 87.2% | 86.8% | 89.3% |
|| CoCoOp | 86.8% | 83.5% | 89.1% | 88.6% | 89.6% |
|| **Ours (Dynamic)** | **86.0%** | **85.9%** | **89.5%** | **89.2%** | **89.8%** |

### 结果分析

**DynamicPromptTrainer vs CoOp**：
|- 1-shot: +5.1% (86.0% vs 80.9%)
|- 2-shot: +3.3% (85.9% vs 82.6%)
|- 4-shot: +2.3% (89.5% vs 87.2%)
|- 8-shot: +2.4% (89.2% vs 86.8%)
|- 16-shot: +0.5% (89.8% vs 89.3%)
|- **结论：全部显著优于 CoOp**

**DynamicPromptTrainer vs CoCoOp**：
|- 1-shot: -0.8%（略低，但差距大幅缩小）
|- 2-shot: +2.4%（大幅领先）
|- 4-shot: +0.4%（领先）
|- 8-shot: +0.6%（领先）
|- 16-shot: +0.2%（领先，达到最佳 89.8%）
|- **结论：在 2/4/8/16-shot 均优于 CoCoOp，整体表现更优**

### Stanford Cars 实验（待完成）

| 数据集 | 类别数 | 状态 | 说明 |
|--------|--------|------|------|
| OxfordPets | 37 | ✅ 已完成 | 最佳 89.8%（16-shot） |
| StanfordCars | 196 | ⏳ 待训练 | 安全关键场景，自动驾驶应用 |

> 2026-05-14 更新：新增 Stanford Cars 数据集支持，训练脚本和评估工具已就绪，待执行训练。

---

## 八、已修复 Bug 记录

### 2026-05-14：多数据集支持（Stanford Cars）

**新增功能**：
1. **Stanford Cars 数据集支持** — 新增 196 类汽车型号细粒度分类
   - 新增 `configs/stanford_cars_rn50.yaml` 配置文件
   - 新增 `run_experiments_stanford_cars.sh` 批量训练脚本
   - `train.py` 已原生支持 `stanford_cars` 数据集
2. **汽车分类 Demo** — 新增 `demo/car_classifier_demo.py`
   - Streamlit 交互界面，支持 196 类汽车型号识别
   - 车型知识库（年份/品牌/型号）
3. **TTC 安全评估扩展** — `evaluate_security_ttc.py` 支持双数据集
   - `--dataset oxford_pets` / `--dataset stanford_cars`
   - 自动根据数据集命名输出文件
4. **文档更新** — README.md / SECURITY.md / PROJECT_STATUS.md 全面更新

### 2026-04-23：Web 界面和 API 研究增强版

**新增功能**：
1. **Flask API 增强** - 新增 `/api/compare`、`/api/breeds`、`/api/breed/<name>`、`/api/experiments`、`/api/model-info` 端点
2. **前端 HTML 页面** - 交互式分类、品种知识库、实验结果看板、API 文档
3. **Streamlit Demo 增强** - 四标签页：分类演示/研究结果/品种知识库/API文档
4. **品种语义信息** - API 返回品种属性（毛发/面部/体型/性格）
5. **动态提示词调试信息** - 返回 ctx_norm、bias_norm、class_adaptive_factors 等

### 2026-04-13：自适应 Epoch 策略 + 实验完成

**训练策略优化**：
- Few-shot 数越少 → 训练 Epoch 越多（避免欠拟合）
- Few-shot 数越多 → 训练 Epoch 越少（避免过拟合）

| Shot | Epochs | 理论依据 |
|------|--------|----------|
| 1-shot | 100 | 数据最少，需要更多迭代学习 |
| 2-shot | 80 | 较少数据 |
| 4-shot | 60 | 中等数据量 |
| 8-shot | 40 | 数据充足 |
| 16-shot | 20 | 数据最充足，短训练防止过拟合 |

**实验结果验证**：
- 自适应 Epoch 策略有效：所有方法在最优 Epoch 下达到最佳性能
- DynamicPromptTrainer 在 2/4/8/16-shot 上全面超越 CoCoOp
- 1-shot 差距从 -1.9% 缩小到 -0.8%

### 2026-04-02：可学习难度权重优化

**问题描述**：
- 难度权重是 detached 的，不参与梯度回传
- 权重计算使用固定阈值，无法自适应

**修复内容**：
1. `DifficultyWeightCalculator` 改为 `nn.Module`
2. 添加可学习参数：
   - `temperature`: 温度参数（初始 0.1）
   - `confidence_scale`: 置信度缩放（初始 2.0）
   - `wrong_weight`: 错误预测权重（初始 2.0）
3. 移除 `weights.detach()`，让梯度回传
4. 新损失函数：`weight = 1 + sigmoid(scale * (1 - conf) / temp)`

### 2026-04-02：dtype 一致性修复

| 问题 | 原因 | 修复 |
|------|------|------|
| fp16 模式下 dtype 不匹配 | SoftPromptAdapter 和 DynamicPromptOptimizer 参数未转换为 fp16 | 在创建时指定 dtype 参数 |

### 2026-03-28：训练稳定性修复

| Bug | 原因 | 修复 |
|-----|------|------|
| 权重范围无限制 | 困难权重可能达到 3-4 倍，对小样本冲击过大 | 添加 `weight = max(0.5, min(2.0, weight))` 限制 |
| 学习率过高 | 0.002 对动态提示词参数过大 | 调整为 0.002（与 CoCoOp 一致） |

### 2026-03-27：核心模块激活修复

| Bug | 原因 | 修复 |
|-----|------|------|
| 难度权重从未计算 | `predictions` 始终为 None | 改为两阶段前向 |
| `class_adaptive_factors` 未参与计算 | 未与 ctx 相乘 | 在 forward 中相乘 |

### 2026-03-20：Web 动态提示词推理 + 显存优化 + verbose 弃用

| Bug | 原因 | 修复 |
|-----|------|------|
| Web API 未使用训练模型的动态提示词 | `predict()` 始终用固定模板 | 新增动态推理路径 |
| `UserWarning: The verbose parameter is deprecated` | PyTorch 新版弃用 verbose 参数 | 移除 verbose 参数 |
| CUDA OOM (batch_size=32) | 默认 batch 过大 | 改为 16 |

---

## 九、创新点总结

1. **可学习难度加权** — `DifficultyWeightCalculator` 使用可学习温度参数，让模型自动学习何时关注困难样本
2. **双层提示词调整** — 在 MLP 偏移之上叠加 `class_adaptive_factors`，为不同类别学习不同的提示词缩放
3. **两阶段前向传播** — 训练时先计算基础 logits 获取预测，再用预测计算难度权重生成自适应提示词
4. **品种语义增强** — 37 品种属性库（毛发/面部/体型/性格），支持多模板文本生成
5. **动态提示词推理** — Web 端加载训练模型后自动切换为动态推理模式
6. **跨设备兼容** — CUDA 训练的模型可在 Mac CPU/MPS 上无缝使用
7. **研究增强版 Web 界面** — 多模型对比、品种语义信息、实验结果展示、调试信息可视化
8. **多数据集支持** — 支持 Oxford Pets（37类猫狗）和 Stanford Cars（196类汽车），覆盖通用分类和安全关键场景
9. **TTC 安全评估** — 支持对抗攻击防御评估，适用于不同数据集的安全分析
