# 模型安全验证文档

## 概述

本项目当前采用的安全验证方案，是**直接复用 `CLIP-Test-time-Counterattacks` 的 TTC 原始实现**，并将其桥接到当前细粒度分类模型上进行测试。

入口脚本：

```bash
fine_grained_classification/evaluate_security_ttc.py
```

该方案的目标不是在项目内重新实现 TTC，而是尽量保持论文代码路径不变，只替换以下两部分：

- 数据集来源：支持 `OxfordPets` 和 `StanfordCars` 测试集
- 被测模型：改为当前项目训练得到的 `DynamicPromptTrainer` checkpoint

## 支持的数据集

| 数据集 | 类别数 | 安全场景 | 说明 |
|--------|--------|---------|------|
| **OxfordPets** | 37 | 通用细粒度分类 | 猫狗品种分类（原始基准） |
| **StanfordCars** | 196 | 自动驾驶/智能交通 | 汽车型号细粒度分类 |

## 方法说明

桥接脚本的执行流程如下：

1. 加载指定数据集的测试集（`OxfordPets` 或 `StanfordCars`）
2. 加载 `DynamicPromptTrainer` 的 `prompt_learner` checkpoint
3. 将测试图像从 Dassl/CoOp 的归一化张量反归一化回 `[0, 1]`
4. 按 TTC 原始代码中的 `clip_img_preprocessing` 重新送入 CLIP 编码器
5. 在当前细粒度分类模型上生成 PGD 对抗样本
6. 对干净样本和对抗样本分别应用 TTC counterattack
7. 输出 clean / adv / TTC 后的准确率与 `tau` 统计

核心特点：

- TTC 的 `tau` 计算逻辑沿用原始实现
- TTC 的加权 counterattack 逻辑沿用原始实现
- 评估对象是你当前项目训练出的细粒度分类模型
- 支持多数据集评估，便于对比不同场景下的鲁棒性

## 相关文件

```text
fine_grained_classification/
├── evaluate_security_ttc.py    # TTC 评估脚本（支持 OxfordPets / StanfordCars）
├── SECURITY.md
├── security_results/
│   ├── ttc_dynamic_prompt_oxford_pets.json
│   └── ttc_dynamic_prompt_stanford_cars.json
└── output_fgd/
    ├── oxford_pets/
    │   └── DynamicPromptTrainer/
    └── stanford_cars/
        └── DynamicPromptTrainer/

CLIP-Test-time-Counterattacks/
└── code/
    ├── attacks.py
    ├── func.py
    └── test_time_counterattack.py
```

## 使用方法

### 1. 最小测试

先用小批量和少量 batch 验证流程能否跑通：

```bash
python fine_grained_classification/evaluate_security_ttc.py \
  --device cpu \
  --batch-size 2 \
  --num-workers 0 \
  --max-batches 1
```

### 2. Oxford Pets 完整评估

```bash
python fine_grained_classification/evaluate_security_ttc.py \
  --dataset oxford_pets \
  --shots 16 \
  --seed 1 \
  --device cuda \
  --ttc-eps 0.011764705882352941 \
  --ttc-numsteps 2 \
  --tau-thres 0.6 \
  --beta 1.0
```

### 3. Stanford Cars 完整评估

```bash
python fine_grained_classification/evaluate_security_ttc.py \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --device cuda \
  --ttc-eps 0.011764705882352941 \
  --ttc-numsteps 2 \
  --tau-thres 0.6 \
  --beta 1.0
```

如果不显式指定 `--model-path`，脚本会自动在以下目录寻找 checkpoint：

```bash
# Oxford Pets
fine_grained_classification/output_fgd/oxford_pets/DynamicPromptTrainer/shots_{shots}/seed_{seed}/prompt_learner/

# Stanford Cars
fine_grained_classification/output_fgd/stanford_cars/DynamicPromptTrainer/shots_{shots}/seed_{seed}/prompt_learner/
```

## 参数说明

### 攻击参数

| 参数 | 含义 |
|------|------|
| `--epsilon` | PGD 攻击扰动上界 |
| `--alpha` | PGD 每步步长 |
| `--num-steps` | PGD 迭代步数 |

### TTC 参数

| 参数 | 含义 |
|------|------|
| `--ttc-eps` | TTC 反攻击扰动上界 |
| `--ttc-stepsize` | TTC 每步步长 |
| `--ttc-numsteps` | TTC 迭代步数 |
| `--tau-thres` | `tau` 判断阈值 |
| `--beta` | TTC 多步加权强度 |

### 运行参数

| 参数 | 含义 |
|------|------|
| `--dataset` | 数据集：`oxford_pets` / `stanford_cars` |
| `--device` | 运行设备：`cpu` / `cuda` / `mps` |
| `--batch-size` | 测试 batch size |
| `--num-workers` | DataLoader worker 数 |
| `--max-batches` | 仅测试前若干个 batch，便于调试 |
| `--model-path` | 手动指定 checkpoint 路径 |
| `--output` | 结果 JSON 输出路径（默认自动根据数据集命名） |

## 输出结果

结果自动保存到：

```bash
# Oxford Pets 默认输出
fine_grained_classification/security_results/ttc_dynamic_prompt_oxford_pets.json

# Stanford Cars 默认输出
fine_grained_classification/security_results/ttc_dynamic_prompt_stanford_cars.json
```

输出字段包括：

- `clean_acc`：干净样本准确率
- `clean_ttc_acc`：干净样本经过 TTC 后的准确率
- `adv_acc`：对抗样本准确率
- `adv_ttc_acc`：对抗样本经过 TTC 后的准确率
- `adv_ttc_gain`：TTC 对对抗样本带来的准确率提升
- `clean_ttc_delta`：TTC 对干净样本准确率造成的变化
- `mean_clean_tau`：干净样本平均 `tau`
- `mean_adv_tau`：对抗样本平均 `tau`
- `num_samples`：实际评估样本数
- `dataset`：评估使用的数据集名称

## 运行建议

- 首次运行先加 `--max-batches 1` 或 `--max-batches 10`，确认流程无误
- CPU 可以跑通，但速度较慢，完整测试建议使用 `cuda`
- 如果只想验证某个特定模型，直接传 `--model-path`
- 若需要论文实验，建议固定 `shots`、`seed`、攻击参数和 TTC 参数，分别保存结果文件
- **Stanford Cars** 数据集具有更强的安全场景关联（自动驾驶），推荐用于安全分析

## 当前实现说明

为适配当前环境，已做必要处理：

- TTC 评估脚本中显式兼容当前 PyTorch 的 checkpoint 加载方式
- `CLIP-Test-time-Counterattacks/code/attacks.py` 做了最小修改，使其不依赖强制 `.cuda()`，并在未安装 `autoattack` 时不影响 PGD 路径
- 支持多数据集自动切换，通过 `--dataset` 参数控制
- 输出文件名自动根据数据集名称生成，避免覆盖

这些修改不改变当前采用的 TTC 核心评估逻辑。

## 参考

- 论文：`CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP`
- 代码目录：`CLIP-Test-time-Counterattacks/`