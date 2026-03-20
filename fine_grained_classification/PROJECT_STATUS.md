# 细粒度猫狗分类系统 - 项目状态报告

**项目名称**: 基于提示词优化的细粒度猫狗分类系统
**技术栈**: PyTorch 2.4.1 + CLIP (RN50) + Python 3.8+
**数据集**: Oxford-IIIT Pets (7,390 张图片, 37 种猫狗品种)
**更新时间**: 2026-03-20

---

## 一、系统架构

### 核心思路

在预训练 CLIP 模型基础上，冻结图像/文本编码器，仅训练**动态提示词模块**，使提示词能根据每张输入图片自适应调整，从而提升细粒度品种的区分能力。

### 与 CoOp / CoCoOp 基线的核心区别

| 特性 | CoOp | CoCoOp | **本系统（DynamicPromptTrainer）** |
|------|------|--------|----------------------------------|
| 提示词类型 | **静态**：所有图片共享同一组可学习 ctx 向量 | **图像条件**：meta_net 生成偏移 | **图像条件 + 难度自适应**：双层调整 + 加权损失 |
| 核心参数 | `ctx` | `ctx` + `meta_net` | `ctx` + `SoftPromptAdapter` + `DynamicPromptOptimizer` + `class_adaptive_factors` |
| 是否感知图像 | 否 | 是 | 是 |
| 是否感知难度 | 否 | 否 | **是（独有）** |
| 损失函数 | 标准 CE | 标准 CE | **加权 CE（困难样本权重更大）** |
| 提示词调整层数 | 单层静态 | 单层偏移 | **双层（MLP 偏移 + 类别自适应因子）** |
| 原型追踪 | 无 | 无 | 动量更新类原型，计算样本到类中心距离 |

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
logit_scale × (image_features @ text_features.T) → logits
  ↓
DifficultyWeightCalculator → 难度权重 w_i
  ↓
加权 CE loss = mean(w_i × CE_i) → 反向传播（仅更新 ctx + MLP + adaptive_factors）
```

**关键创新**: 同一只"波斯猫"，不同姿态/角度的图片会产生不同的提示词偏移，让模型关注该图片中最有区分力的特征。

---

## 二、项目结构

```
fine_grained_classification/
├── train.py                     # 训练入口（支持 CoOp/CoCoOp/DynamicPromptTrainer）
├── evaluate.py                  # 评估脚本
├── collect_results.py           # 实验结果汇总与图表生成
├── run_experiments.sh           # 自动化批量实验脚本
├── configs/
│   └── dynamic_rn50.yaml        # 训练超参配置
├── models/
│   ├── custom_clip.py           # 自定义 CLIP（整合动态提示 + 语义增强）
│   ├── dynamic_prompt.py        # AdaptivePromptLearner + SoftPromptAdapter
│   ├── trainer.py               # DynamicPromptTrainer（注册到 Dassl）
│   └── breed_semantic.py        # 品种属性库（37 品种的毛发/面部/体型特征）
├── demo/
│   └── pet_classifier_demo.py   # Streamlit 交互演示
├── web/
│   └── app.py                   # Flask REST API 服务
└── utils/
    └── helpers.py               # 可视化与度量工具
```

---

## 三、训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Backbone | RN50 | CLIP ResNet-50 |
| 优化器 | SGD | lr=0.002 |
| 学习率调度 | Cosine Annealing | warmup 1 epoch |
| 训练轮次 | 50 | - |
| 批大小 | 16 | 默认值，可按 GPU 显存调整 |
| 可学习 ctx 长度 | 4 tokens | 初始化为 "a photo of a" |
| ctx 嵌入维度 | 512 | 与 CLIP 特征维度一致 |
| SoftPromptAdapter | 512→64→512 | 双层 MLP + ReLU |
| 难度权重 | 距离系数 0.5 + 误分类 ×2 | 动量原型更新(0.9) |

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
# 下载数据集到 ../data/oxford_pets/

# 5. 开始训练
cd ../../fine_grained_classification
python train.py -d oxford_pets -e 50 -b 16 --shots 1 --trainer DynamicPromptTrainer --device cuda
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
│   └── oxford_pets/           # 数据集
├── fine_grained_classification/
│   ├── train.py
│   └── ...
└── output_fgd/                # 训练输出
```

---

## 五、快速使用

### 环境准备

```bash
source ../CoOp/venv/bin/activate
pip install streamlit flask flask-cors matplotlib seaborn scikit-learn
```

### 训练模型

```bash
# 在 fine_grained_classification/ 目录下执行
# 单组实验（输出自动保存到 output_fgd/oxford_pets/{trainer}/shots_{n}/seed_{s}/）
python train.py -d oxford_pets -e 50 -b 16 --shots 1 --trainer DynamicPromptTrainer --device cuda

# 批量运行全部 9 组对比实验
bash run_experiments.sh cuda
```

### 启动演示

```bash
# Streamlit 界面 → http://localhost:8501
streamlit run demo/pet_classifier_demo.py

# Flask API → http://localhost:5001（zero-shot 模式）
cd web && python app.py

# 使用训练模型启动 API（动态提示词推理模式）
python app.py --model ../output_fgd/oxford_pets/DynamicPromptTrainer/shots_1/seed_1/prompt_learner/model-best.pth.tar
```

> 加载训练模型后，Web API 自动切换为**动态提示词推理模式**，通过 `SoftPromptAdapter` 生成图像条件化的提示词。
> API 响应中 `mode` 字段标明当前推理模式：`"dynamic_prompt"` 或 `"zero_shot"`。
> CUDA 训练的模型可直接在 Mac CPU 上使用，PyTorch 通过 `map_location` 自动映射设备。

---

## 五、当前进度

### 已完成

- [x] 核心模型模块（dynamic_prompt / custom_clip / trainer / breed_semantic）
- [x] 训练流程（train.py 支持 3 种 trainer + few-shot 配置）
- [x] Web 演示（Streamlit + Flask API）
- [x] 所有单元测试通过（test_core / test_demo / test_full）
- [x] Bug 修复：logit_scale 缺失、类名错误（36→37 类）、导入路径等
- [x] 云端部署支持：替换完整版 Dassl、修复 ftfy 版本、路径自动查找
- [x] Web API 动态提示词推理：加载训练模型后，API 使用 prompt_learner + TextEncoder 生成图像条件化的提示词，而非固定模板
- [x] 跨设备模型兼容：CUDA 训练的模型可在 Mac CPU/MPS 上直接加载使用
- [x] 默认 batch size 从 32 降为 16，避免 GPU 显存不足
- [x] 修复 PyTorch lr_scheduler verbose 参数弃用警告

### 待完成

- [ ] 云端运行完整训练实验（9 组：3 方法 × 3 shot）
  - [ ] DynamicPromptTrainer: 1-shot / 4-shot / 16-shot
  - [ ] CoOp 基线: 1-shot / 4-shot / 16-shot
  - [ ] CoCoOp 基线: 1-shot / 4-shot / 16-shot
- [ ] 将训练好的模型同步回本地
- [ ] 生成对比表格、学习曲线、混淆矩阵
- [ ] 撰写论文实验章节

### 预期实验结果

| 方法 | 1-shot | 4-shot | 16-shot |
|------|--------|--------|---------|
| Zero-shot CLIP | ~81% | ~81% | ~81% |
| CoOp | ~75% | ~85% | ~90% |
| CoCoOp | ~76% | ~86% | ~90% |
| **Ours (Dynamic)** | ~78% | ~87% | ~91% |

---

## 六、已修复 Bug 记录

### 2026-03-20：Web 动态提示词推理 + 显存优化 + verbose 弃用

| Bug | 原因 | 修复 |
|-----|------|------|
| Web API 未使用训练模型的动态提示词 | `predict()` 始终用固定模板 `"a photo of a {name}"` | 新增动态推理路径：`prompt_learner(image_features)` → `TextEncoder` → 相似度计算 |
| `UserWarning: The verbose parameter is deprecated` | PyTorch 新版弃用 `_LRScheduler` 的 `verbose` 参数 | `lr_scheduler.py` 中 `super().__init__()` 不再传递 `verbose` |
| CUDA OOM (batch_size=32) | 默认 batch 过大，GPU 显存不足 | `run_experiments.sh` + `evaluate.py` 默认 batch size 从 32 改为 16 |
| 模型加载 strict 不匹配 | `token_prefix`/`token_suffix` 是 buffer 不需要加载 | `load_trained_model()` 加载时过滤固定 token 并使用 `strict=False` |

### 2026-03-17：云端部署 dassl 模块缺失

| Bug | 原因 | 修复 |
|-----|------|------|
| `ModuleNotFoundError: No module named 'dassl.data'` | 原 CoOp/dassl 目录不完整，缺少 `dassl/data` 模块 | 用完整的 Dassl.pytorch 仓库替换，删除内部 .git 目录 |
| `ftfy==6.3` 版本不存在 | pip 无法找到该版本 | 降级为 `ftfy==6.2.3` |
| 路径硬编码 | train.py 使用本地绝对路径 `/Users/yudu/...` | 改为自动查找 CoOp 目录（相对于项目根目录） |

修复步骤：
```bash
# 本地操作
cd CoOp
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git temp_dassl
rm -rf dassl
mv temp_dassl dassl
rm -rf dassl/.git  # 删除嵌套的 git 目录

# 云端操作
git pull
cd CoOp/dassl && pip install -e .
```

### 2026-03-14：Web 界面准确率修复

| Bug | 原因 | 修复 |
|-----|------|------|
| 概率显示 ~3% | `logit_scale`(×100) 未乘 | `web/app.py` + `demo/pet_classifier_demo.py` 添加 `logit_scale * logits` |
| 类名错误 | `Great_Dane` 应为 `great_pyrenees`；`German_Shorthaired_Pointer` 应为 `german_shorthaired` | 修正为数据集一致的 37 个类名 |
| 缺少 1 个类 | 36 类，缺 `american_pit_bull_terrier` | 补齐 37 类 |

修复后零样本实测：200 张随机图片 → **81.0%** Top-1 准确率，Top-1 概率通常 60-99%。

### 2026-03-11：训练流程修复

- `train.py`: 配置扩展 `extend_cfg()`、数据集名称映射、路径修正
- `dynamic_prompt.py`: 移除多余 `unsqueeze(0)`、初始化 `current_weights`
- `custom_clip.py`: 批量文本编码（解决 MPS 内存溢出）、矩阵乘法维度修正
- `trainer.py`: `current_epoch` → `self.epoch`
- `demo/pet_classifier_demo.py` + `web/app.py`: 导入路径修复

---

## 七、创新点总结

1. **图像条件动态提示词** — 通过 SoftPromptAdapter 使每张图片获得独特的提示词偏移（区别于 CoOp 的静态提示和 CoCoOp 的单层偏移）
2. **双层提示词调整** — 在 MLP 偏移之上叠加 `class_adaptive_factors`，为不同类别的提示词学习独立缩放（CoOp/CoCoOp 均无此机制）
3. **困难样本自适应加权** — DifficultyWeightCalculator 基于动量原型距离（momentum=0.9）和误分类历史动态加权损失函数
4. **品种语义增强** — 37 品种属性库（毛发/面部/体型/性格），支持多模板文本生成
5. **动态提示词推理** — Web 端加载训练模型后自动切换为动态推理模式，每张图片实时生成图像条件化的提示词
6. **跨设备兼容** — CUDA 训练的模型可在 Mac CPU/MPS 上无缝使用
