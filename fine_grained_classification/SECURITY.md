# 对抗性防御安全文档

## 概述

本项目实现了基于 **Test-time Counterattack (TTC)** 的对抗性防御机制，源自论文：

> **"CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP"** (CVPR 2025)
> Songlong Xing, Zhengyu Zhao, Nicu Sebe

## 核心原理

### 对抗性攻击

对抗性攻击通过向输入图像添加人类视觉难以察觉的微小扰动，使深度学习模型产生错误预测。典型的攻击方法包括：

- **PGD (Projected Gradient Descent)**: 迭代式攻击，通过多步梯度更新最大化分类损失
- **FGSM (Fast Gradient Sign Method)**: 单步攻击，使用梯度的符号方向进行快速扰动

### Test-time Counterattack (TTC) 防御

TTC 的核心思想是：**利用 CLIP 预训练视觉编码器在推理时对抗对抗性图像**。

#### 关键洞察

1. **特征差异检测**: 对抗性图像与原始图像在 CLIP 特征空间中存在显著差异
2. **自适应防御**: 根据特征差异比率 (tau) 自动判断图像是否被攻击，并自适应调整防御强度
3. **无需训练**: 作为一种测试时防御方法，TTC 不需要额外的对抗性训练

#### 算法流程

```
1. 对输入图像 X 计算原始特征 f_orig
2. 生成小扰动 delta（反向攻击方向）
3. 计算扰动后特征 f_att
4. 计算特征差异比率: tau = ||f_att - f_orig|| / ||f_orig||
5. 根据 tau 值决定防御强度:
   - tau > tau_threshold: 应用较强防御
   - tau <= tau_threshold: 应用轻量防御
6. 返回防御后的图像
```

## 文件结构

```
models/
├── adversarial_defense.py   # 核心防御模块
│   ├── TestTimeCounterattack      # TTC防御实现
│   ├── AdversarialDetector         # 对抗性检测器
│   ├── RobustPromptLearner         # 鲁棒提示学习器
│   └── create_defense_system       # 防御系统工厂函数
│
├── robust_custom_clip.py      # 集成防御的CLIP模型
│   ├── RobustCustomCLIP             # 动态提示+防御
│   ├── RobustCustomCLIPCoCoOp       # CoCoOp+防御
│   └── build_robust_clip            # 模型构建函数
│
└── custom_clip.py            # 原始CLIP模型（基础）

evaluate_security.py          # 安全评估脚本
```

## 使用方法

### 1. 基础使用

```python
import torch
from models.robust_custom_clip import build_robust_clip
from models.adversarial_defense import create_defense_system

# 防御配置
defense_config = {
    "eps": 4.0 / 255.0,      # 防御扰动上界
    "num_steps": 2,            # 防御迭代步数
    "step_size": 1.0 / 255.0,  # 防御步长
    "tau_threshold": 0.2,      # 特征差异阈值
    "beta": 2.0                # 防御强度系数
}

# 构建带防御的模型
model = build_robust_clip(
    cfg, classnames, clip_model,
    model_type="dynamic",
    defense_config=defense_config
)

# 推理（自动检测并防御）
output, defense_info = model(image)

# 或强制使用防御模式
output, defense_info = model(image, defense_mode="defend")
```

### 2. 设置防御模式

```python
# 自动模式（推荐）：自动检测并防御
model.set_defense_mode("auto")

# 始终防御：对所有输入应用防御
model.set_defense_mode("defend")

# 正常模式：不应用防御
model.set_defense_mode("normal")

# 启用/禁用防御
model.set_defense_enabled(True)  # 启用
model.set_defense_enabled(False) # 禁用
```

### 3. 检测对抗性图像

```python
# 检测图像是否为对抗性
is_adversarial, confidence, tau = model.detect_adversarial(image)
print(f"Is adversarial: {is_adversarial}")
print(f"Confidence: {confidence}")
print(f"Tau value: {tau}")
```

### 4. 安全评估

```bash
python evaluate_security.py \
    --dataset oxford_pets \
    --epsilon 4.0/255.0 \
    --defense-eps 4.0/255.0 \
    --defense-steps 2 \
    --output-dir security_results
```

## 配置参数

### TestTimeCounterattack 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eps` | 4.0/255.0 | 防御扰动上界（像素值归一化到[0,1]） |
| `num_steps` | 2 | PGD迭代步数 |
| `step_size` | 1.0/255.0 | 每步扰动大小 |
| `tau_threshold` | 0.2 | 特征差异阈值，用于判断是否为对抗性图像 |
| `beta` | 2.0 | 防御强度系数，控制不同步之间权重的指数增长 |
| `norm` | "l_inf" | 扰动范数类型，可选 "l_inf" 或 "l_2" |

### AdversarialDetector 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eps` | 4.0/255.0 | 检测用轻量扰动幅度 |
| `num_steps` | 2 | 检测用迭代步数 |
| `step_size` | 1.0/255.0 | 检测用步长 |
| `tau_threshold` | 0.2 | 检测阈值 |

## 防御机制详解

### 特征差异比率 (Tau)

Tau 是 TTC 的核心指标：

```
tau = ||f_noisy - f_orig|| / ||f_orig||
```

- **正常图像**: tau 值较小（通常 < 0.2）
- **对抗性图像**: tau 值较大（通常 > 0.2）

### 自适应权重机制

TTC 在多个防御步骤中使用加权平均：

```
weights[i] = exp(scheme_sign * i * beta)
```

- 当 tau > tau_threshold（可能是对抗性图像）时，使用较大的 scheme_sign，增大后期步骤权重
- 当 tau <= tau_threshold（正常图像）时，仅使用初始扰动（单次RN）

## 注意事项

### 1. 计算开销

- 防御机制会增加推理时间（通常约 2-3 倍）
- 可以通过设置 `defense_mode="auto"` 在仅检测到攻击时应用防御

### 2. 干净样本准确率

- 防御机制可能轻微降低干净样本上的准确率
- 这是对抗鲁棒性与分类精度之间的权衡

### 3. 阈值调整

- `tau_threshold` 需要根据具体数据集和攻击强度调整
- 建议范围: 0.1 ~ 0.3

### 4. 防御强度

- `eps` 和 `num_steps` 影响防御强度
- 较强防御可能增加计算开销和干净样本准确率下降

## 参考

- 原始论文: https://arxiv.org/abs/2503.03613
- CLIP-Test-time-Counterattacks 代码库
