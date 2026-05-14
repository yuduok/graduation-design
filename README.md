# 毕业设计 - 基于动态提示词优化的细粒度分类系统

杭州电子科技大学 网络安全学院 毕业设计项目

**题目**: 基于动态提示词优化的细粒度分类系统
**技术栈**: PyTorch + CLIP + 动态提示词学习
**支持数据集**: Oxford-IIIT Pets (37 种猫狗品种) / Stanford Cars (196 种汽车型号)
**安全场景**: 通用细粒度分类 / 自动驾驶车辆识别

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
│   ├── oxford_pets/                   # Oxford-IIIT Pets 数据集
│   │   ├── images/                    # 7,390 张宠物图片
│   │   ├── annotations/               # 标注文件
│   │   └── split_zhou_OxfordPets.json # 数据集划分
│   └── stanford_cars/                 # Stanford Cars 数据集
│       ├── cars_train/                # 训练图像
│       ├── cars_test/                 # 测试图像
│       ├── devkit/                    # 开发工具包
│       └── cars_test_annos_withlabels.mat # 测试标注
│
├── fine_grained_classification/       # 本项目核心代码
│   ├── README.md                      # 详细使用文档
│   ├── PROJECT_STATUS.md              # 项目状态报告
│   ├── SECURITY.md                    # 安全验证文档
│   ├── requirements.txt               # Python 依赖列表
│   │
│   ├── train.py                       # 训练入口脚本
│   ├── compare_models.py              # 模型对比评估
│   ├── evaluate.py                    # 评估脚本
│   ├── evaluate_security_ttc.py       # 对抗防御评估（支持双数据集）
│   ├── collect_results.py             # Oxford Pets 结果汇总
│   ├── collect_results_stanford_cars.py # Stanford Cars 结果汇总
│   ├── run_experiments.sh             # Oxford Pets 批量实验
│   ├── run_experiments_stanford_cars.sh # Stanford Cars 批量实验
│   ├── export_thesis_results.py       # 论文结果导出
│   ├── generate_thesis_figures.py     # 论文图表生成
│   │
│   ├── configs/                       # 训练配置文件
│   │   ├── dynamic_rn50.yaml          # RN50 超参配置（Oxford Pets）
│   │   ├── dynamic_vitb16.yaml        # ViT-B/16 超参配置
│   │   └── stanford_cars_rn50.yaml    # RN50 超参配置（Stanford Cars）
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
│   │   ├── pet_classifier_demo.py     # Streamlit 交互演示（Oxford Pets）
│   │   └── car_classifier_demo.py     # Streamlit 交互演示（Stanford Cars）
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
│   │   ├── oxford_pets/
│   │   │   ├── CoOp/                  # CoOp 实验结果
│   │   │   ├── CoCoOp/                # CoCoOp 实验结果
│   │   │   ├── DynamicPromptTrainer/  # DynamicPrompt 实验结果
│   │   │   └── experiment_summary.json
│   │   └── stanford_cars/
│   │       ├── CoOp/
│   │       ├── CoCoOp/
│   │       ├── DynamicPromptTrainer/
│   │       └── experiment_summary.json
│   │
│   ├── comparison_results/            # 模型对比结果
│   │   ├── accuracy_comparison.png
│   │   ├── confidence_distribution.png
│   │   ├── top_k_accuracies.png
│   │   └── comparison_summary.json
│   │
│   ├── security_results/              # 对抗防御实验结果
│   │   ├── ttc_dynamic_prompt_oxford_pets.json
│   │   └── ttc_dynamic_prompt_stanford_cars.json
│   │
│   ├── modal_train.py                 # Modal 云端训练脚本
│   ├── modal_upload.py                # Modal 数据集上传脚本
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

#### Oxford-IIIT Pets

```bash
mkdir -p ../data/oxford_pets && cd ../data/oxford_pets

# 下载 Oxford-IIIT Pets 数据集
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

#### Stanford Cars

```bash
mkdir -p ../data/stanford_cars && cd ../data/stanford_cars

# 下载 Stanford Cars 数据集
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat
tar -xzf cars_train.tgz
tar -xzf cars_test.tgz
tar -xzf car_devkit.tgz
```

### 3. 训练模型

#### Oxford Pets

```bash
cd ../../fine_grained_classification

# 训练 DynamicPromptTrainer（16-shot）
python train.py -d oxford_pets -t DynamicPromptTrainer --shots 16 -e 20 --device cuda

# 批量运行全部对比实验
bash run_experiments.sh cuda

# 收集实验结果
python collect_results.py --latex --plot
```

#### Stanford Cars

```bash
# 训练 DynamicPromptTrainer（16-shot）
python train.py -d stanford_cars -t DynamicPromptTrainer --shots 16 -e 50 --device cuda

# 批量运行全部对比实验
bash run_experiments_stanford_cars.sh cuda

# 收集实验结果
python collect_results_stanford_cars.py --latex --plot
```

### 4. TTC 安全评估

```bash
# Oxford Pets 安全评估
python evaluate_security_ttc.py --dataset oxford_pets --shots 16 --device cuda

# Stanford Cars 安全评估
python evaluate_security_ttc.py --dataset stanford_cars --shots 16 --device cuda
```

### 5. 启动演示

```bash
# 方式一：Flask API 服务（推荐）
cd web
python app.py --shot 16 --port 5001
# 访问 http://localhost:5001/

# 方式二：Streamlit 交互演示（Oxford Pets）
streamlit run demo/pet_classifier_demo.py
# 访问 http://localhost:8501

# 方式三：Streamlit 交互演示（Stanford Cars）
streamlit run demo/car_classifier_demo.py
# 访问 http://localhost:8501
```

---

## 核心特性

- **动态提示词优化**: 图像条件化提示词 + 可学习难度加权
- **多方法对比**: CoOp / CoCoOp / DynamicPromptTrainer
- **多数据集支持**: Oxford Pets（37类猫狗）/ Stanford Cars（196类汽车）
- **品种语义增强**: 37 品种属性数据库（毛发/面部/体型/性格）
- **自适应 Epoch 策略**: Few-shot 数越少，训练 Epoch 越多
- **研究增强版 Web 界面**: 多模型对比、品种知识库、实验结果展示
- **对抗安全评估**: TTC 测试时反攻击防御评估
- **云端训练支持**: Modal 平台一键云端 GPU 训练

---

## 实验结果

### Oxford-IIIT Pets

| 方法 | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|--------|---------|
| CoOp | 80.9% | 82.6% | 87.2% | 86.8% | 89.3% |
| CoCoOp | 86.8% | 83.5% | 89.1% | 88.6% | 89.6% |
| **DynamicPrompt (Ours)** | **86.0%** | **85.9%** | **89.5%** | **89.2%** | **89.8%** |

最佳结果：**89.8%**（16-shot），超越 CoOp (+0.5%) 和 CoCoOp (+0.2%)

### Stanford Cars（待完成）

| 方法 | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|--------|---------|
| CoOp | - | - | - | - | - |
| CoCoOp | - | - | - | - | - |
| **DynamicPrompt (Ours)** | - | - | - | - | - |

---

## 云端训练（Modal）

```bash
cd fine_grained_classification

# 安装 Modal
pip install modal
modal setup

# 上传数据集
modal run modal_upload.py --dataset-dir /path/to/stanford_cars --remote-path /stanford_cars

# 云端训练
modal run modal_train.py --dataset stanford_cars --trainer DynamicPromptTrainer --shots 16 --epochs 50

# 下载结果
modal volume get fgc-output / output_fgd/modal/
```

---

## 详细文档

- [fine_grained_classification/README.md](fine_grained_classification/README.md) - 详细使用文档
- [fine_grained_classification/PROJECT_STATUS.md](fine_grained_classification/PROJECT_STATUS.md) - 项目状态报告
- [fine_grained_classification/SECURITY.md](fine_grained_classification/SECURITY.md) - 安全验证文档

---

## 参考资料

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [CoOp: Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
- [CoCoOp: Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [CLIP-Test-time-Counterattacks](https://github.com/ldhl-hk/CLIP-Test-time-Counterattacks)