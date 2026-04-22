# 毕业设计 - 基于动态提示词优化的细粒度猫狗分类

杭州电子科技大学 网络安全学院 毕业设计项目

**题目**: 基于动态提示词优化的细粒度猫狗分类系统
**技术栈**: PyTorch + CLIP + 动态提示词学习
**数据集**: Oxford-IIIT Pets (37 种猫狗品种)

---

## 仓库结构

```
graduation-design/
│
├── README.md                          # 本文件：仓库结构说明
│
├── CoOp/                              # CoOp 框架（第三方依赖）
│   ├── README.md                      # CoOp 官方文档
│   ├── clip/                          # CLIP 模型实现（修改版）
│   │   ├── __init__.py
│   │   ├── clip.py                    # CLIP 核心代码
│   │   └── simple_tokenizer.py        # BPE 分词器
│   ├── dassl/                         # Dassl 深度学习框架
│   │   ├── README.md
│   │   ├── dassl/                     # 核心库
│   │   │   ├── engine/                # 训练引擎
│   │   │   ├── data/                  # 数据集管理
│   │   │   ├── model/                 # 模型定义
│   │   │   ├── optim/                 # 优化器
│   │   │   └── utils/                 # 工具函数
│   │   └── setup.py                   # 安装脚本
│   ├── trainers/                      # CoOp/CoCoOp 训练器
│   │   ├── coop.py
│   │   └── cocoop.py
│   └── datasets/                      # 数据集定义
│
├── data/                              # 数据集目录（需手动下载）
│   └── oxford_pets/                   # Oxford-IIIT Pets 数据集
│       ├── images/                    # 7,390 张宠物图片
│       ├── annotations/               # 标注文件
│       └── split_zhou_OxfordPets.json # 数据集划分
│
├── fine_grained_classification/       # 本项目核心代码
│   ├── README.md                      # 详细使用文档
│   ├── PROJECT_STATUS.md              # 项目状态报告
│   ├── requirements.txt               # Python 依赖列表
│   │
│   ├── train.py                       # 训练入口脚本
│   ├── compare_models.py              # 模型对比评估
│   ├── evaluate.py                    # 评估脚本
│   ├── collect_results.py             # 实验结果汇总与图表生成
│   ├── run_experiments.sh             # 批量实验脚本
│   ├── evaluate_security_ttc.py       # 对抗防御评估
│   ├── export_thesis_results.py       # 论文结果导出
│   ├── generate_thesis_figures.py     # 论文图表生成
│   │
│   ├── configs/                       # 训练配置文件
│   │   ├── dynamic_rn50.yaml          # RN50 超参配置
│   │   └── dynamic_vitb16.yaml        # ViT-B/16 超参配置
│   │
│   ├── models/                        # 模型定义
│   │   ├── __init__.py                # 模块导出
│   │   ├── custom_clip.py             # 自定义 CLIP（动态提示 + 语义增强）
│   │   ├── dynamic_prompt.py          # 动态提示词核心模块
│   │   │                              #   - AdaptivePromptLearner
│   │   │                              #   - SoftPromptAdapter
│   │   │                              #   - DifficultyWeightCalculator
│   │   ├── trainer.py                 # DynamicPromptTrainer（Dassl 注册）
│   │   ├── breed_semantic.py          # 品种语义增强（37品种属性库）
│   │   ├── adversarial_defense.py     # 对抗性防御（TTC）
│   │   └── robust_custom_clip.py      # 鲁棒 CLIP 模型
│   │
│   ├── demo/                          # 演示系统
│   │   └── pet_classifier_demo.py     # Streamlit 交互演示
│   │
│   ├── web/                           # Web API 服务
│   │   ├── app.py                     # Flask REST API（研究增强版）
│   │   └── static/
│   │       └── index.html             # 前端 HTML 演示页面
│   │
│   ├── utils/                         # 工具函数
│   │   └── helpers.py                 # 可视化与度量工具
│   │
│   ├── output_fgd/                    # 实验输出目录
│   │   └── oxford_pets/
│   │       ├── CoOp/                  # CoOp 实验结果
│   │       │   └── shots_{1,2,4,8,16}/
│   │       │       └── seed_1/
│   │       │           └── prompt_learner/
│   │       │               ├── checkpoint
│   │       │               └── model.pth.tar-{epoch}
│   │       ├── CoCoOp/                # CoCoOp 实验结果
│   │       │   └── shots_{1,2,4,8,16}/
│   │       │       └── seed_1/
│   │       │           └── prompt_learner/
│   │       ├── DynamicPromptTrainer/  # DynamicPrompt 实验结果
│   │       │   └── shots_{1,2,4,8,16}/
│   │       │       └── seed_1/
│   │       │           └── prompt_learner/
│   │       └── experiment_summary.json # 实验结果摘要
│   │
│   ├── comparison_results/            # 模型对比结果
│   │   ├── accuracy_comparison.png
│   │   ├── confidence_distribution.png
│   │   ├── top_k_accuracies.png
│   │   └── comparison_summary.json
│   │
│   ├── security_results/              # 对抗防御实验结果
│   │   └── ttc_dynamic_prompt_oxford_pets.json
│   │
│   ├── thesis/                        # 毕业论文 LaTeX 源文件
│   │   ├── main.tex                   # 主文件
│   │   ├── preamble.tex               # 导言区
│   │   ├── chapter1.tex ~ chapter6.tex # 各章节
│   │   ├── abstract.tex               # 摘要
│   │   ├── thanks.tex                 # 致谢
│   │   ├── ref.bib                    # 参考文献
│   │   ├── HDU-Bachelor-Thesis.cls    # 论文模板类
│   │   ├── chapters/                  # 章节文件
│   │   └── thesis_figures/            # 论文插图
│   │
│   └── thesis_figures/                # 论文图表（复制）
│       ├── adaptive_epochs.png
│       ├── comparison_vs_cocoop.png
│       ├── method_comparison.png
│       └── method_comparison_line.png
│
└── CLIP-Test-time-Counterattacks/     # 对抗防御参考代码（第三方）
    └── ...
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <your-repo-url>
cd graduation-design

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
cd fine_grained_classification
pip install -r requirements.txt

# 安装 Dassl（关键步骤）
cd ../CoOp/dassl
pip install -e .
```

### 2. 准备数据集

```bash
mkdir -p ../data/oxford_pets && cd ../data/oxford_pets

# 下载 Oxford-IIIT Pets 数据集
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

### 3. 训练模型

```bash
cd ../../fine_grained_classification

# 训练 DynamicPromptTrainer（16-shot）
python train.py -d oxford_pets -t DynamicPromptTrainer --shots 16 -e 20 --device cuda

# 批量运行全部对比实验
bash run_experiments.sh cuda
```

### 4. 启动演示

```bash
# 方式一：Flask API 服务（推荐）
cd web
python app.py --shot 16 --port 5001
# 访问 http://localhost:5001/

# 方式二：Streamlit 交互演示
cd demo
streamlit run pet_classifier_demo.py
# 访问 http://localhost:8501
```

---

## 核心特性

- **动态提示词优化**: 图像条件化提示词 + 可学习难度加权
- **多方法对比**: CoOp / CoCoOp / DynamicPromptTrainer
- **品种语义增强**: 37 品种属性数据库（毛发/面部/体型/性格）
- **自适应 Epoch 策略**: Few-shot 数越少，训练 Epoch 越多
- **研究增强版 Web 界面**: 多模型对比、品种知识库、实验结果展示

---

## 实验结果

在 Oxford-IIIT Pets 数据集上的准确率对比：

| 方法 | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|--------|---------|
| CoOp | 80.9% | 82.6% | 87.2% | 86.8% | 89.3% |
| CoCoOp | 86.8% | 83.5% | 89.1% | 88.6% | 89.6% |
| **DynamicPrompt (Ours)** | **86.0%** | **85.9%** | **89.5%** | **89.2%** | **89.8%** |

最佳结果：**89.8%**（16-shot），超越 CoOp (+0.5%) 和 CoCoOp (+0.2%)

---

## 详细文档

- [fine_grained_classification/README.md](fine_grained_classification/README.md) - 详细使用文档
- [fine_grained_classification/PROJECT_STATUS.md](fine_grained_classification/PROJECT_STATUS.md) - 项目状态报告

---

## 参考资料

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [CoOp: Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
- [CoCoOp: Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
