# 细粒度猫狗分类系统 - 项目状态报告

**项目名称**: 基于提示词优化的细粒度猫狗分类系统
**技术栈**: PyTorch 2.4.1 + CLIP (RN50) + Python 3.8
**数据集**: Oxford-IIIT Pets (7,390 张图片, 37 种猫狗品种)
**更新时间**: 2026-03-14

---

## 一、系统架构

### 核心思路

在预训练 CLIP 模型基础上，冻结图像/文本编码器，仅训练**动态提示词模块**，使提示词能根据每张输入图片自适应调整，从而提升细粒度品种的区分能力。

### 与 CoOp 基线的核心区别

| 特性 | CoOp（基线） | 本系统（DynamicPromptTrainer） |
|------|-------------|-------------------------------|
| 提示词生成 | **静态**：所有图片共享同一组可学习 ctx 向量 | **图像条件**：每张图片经 SoftPromptAdapter 生成独特偏移 |
| 训练信号 | 均匀 Cross-Entropy loss | **难度加权 loss**：难样本梯度放大 2-4x |
| 适应机制 | 无 | SoftPromptAdapter (512→64→512 双层 MLP) |
| 原型追踪 | 无 | 动量更新类原型，计算样本到类中心距离 |

### 动态提示词工作流程

```
输入图片
  ↓
[冻结] CLIP 视觉编码器 → image_features [batch, 512]
  ↓
[可训练] SoftPromptAdapter MLP → 图片特定的 ctx 偏移 [batch, 512]
  ↓
基础 ctx（可学习）+ 偏移 → 图片条件提示词嵌入 [batch, n_cls, n_ctx, 512]
  ↓
[冻结] CLIP 文本编码器 → text_features [batch, n_cls, 512]
  ↓
logit_scale × (image_features @ text_features.T) → logits
  ↓
DifficultyWeightCalculator → 难度权重 w_i
  ↓
加权 CE loss = mean(w_i × CE_i) → 反向传播（仅更新 ctx + MLP）
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
| 批大小 | 32 | - |
| 可学习 ctx 长度 | 4 tokens | 初始化为 "a photo of a" |
| ctx 嵌入维度 | 512 | 与 CLIP 特征维度一致 |
| SoftPromptAdapter | 512→64→512 | 双层 MLP + ReLU |
| 难度权重 | 距离系数 0.5 + 误分类 ×2 | 动量原型更新(0.9) |

---

## 四、快速使用

### 环境准备

```bash
source ../CoOp/venv/bin/activate
pip install streamlit flask flask-cors matplotlib seaborn scikit-learn
```

### 训练模型

```bash
# 在 fine_grained_classification/ 目录下执行
# 单组实验（输出自动保存到 output_fgd/oxford_pets/{trainer}/shots_{n}/seed_{s}/）
python train.py -d oxford_pets -e 50 -b 32 --shots 1 --trainer DynamicPromptTrainer --device cuda

# 批量运行全部 9 组对比实验
bash run_experiments.sh cuda
```

### 汇总结果

```bash
python collect_results.py --latex --plot
```

### 启动演示

```bash
# Streamlit 界面 → http://localhost:8501
streamlit run demo/pet_classifier_demo.py

# Flask API → http://localhost:5001
cd web && python app.py
# 使用训练模型: python app.py --model ../output_fgd/oxford_pets/DynamicPromptTrainer/shots_1/seed_1/prompt_learner/model-best.pth.tar
```

---

## 五、当前进度

### 已完成

- [x] 核心模型模块（dynamic_prompt / custom_clip / trainer / breed_semantic）
- [x] 训练流程（train.py 支持 3 种 trainer + few-shot 配置）
- [x] Web 演示（Streamlit + Flask API）
- [x] 所有单元测试通过（test_core / test_demo / test_full）
- [x] Bug 修复：logit_scale 缺失、类名错误（36→37 类）、导入路径等

### 待完成

- [ ] 运行完整训练实验（9 组：3 方法 × 3 shot）
  - [ ] DynamicPromptTrainer: 1-shot / 4-shot / 16-shot
  - [ ] CoOp 基线: 1-shot / 4-shot / 16-shot
  - [ ] CoCoOp 基线: 1-shot / 4-shot / 16-shot
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

1. **图像条件动态提示词** — 通过 SoftPromptAdapter 使每张图片获得独特的提示词偏移
2. **难度加权损失** — DifficultyWeightCalculator 基于原型距离和误分类状态动态加权
3. **品种语义增强** — 37 品种属性库（毛发/面部/体型/性格），支持多模板文本生成
4. **端到端可视化** — Streamlit 界面 + Flask API，支持自定义提示词模板实时对比
