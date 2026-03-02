# 细粒度猫狗分类系统

基于 CLIP + 动态提示词优化的细粒度猫狗品种分类

## 项目结构

```
fine_grained_classification/
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── configs/
│   └── dynamic_rn50.yaml  # 训练配置
├── models/
│   ├── custom_clip.py     # 自定义CLIP模型
│   ├── dynamic_prompt.py  # 动态提示词模块
│   ├── trainer.py         # 训练器
│   └── breed_semantic.py  # 品种语义增强
├── utils/
│   └── helpers.py         # 工具函数
├── demo/
│   └── pet_classifier_demo.py  # Streamlit演示
└── web/
    └── app.py             # Flask API服务
```

## 快速开始

### 1. 环境安装

```bash
pip install torch torchvision
pip install clip
pip install streamlit
pip install flask flask-cors
pip install open-clip-torch  # 或使用原版CLIP
```

### 2. 数据准备

下载 Oxford-IIIT Pets 数据集：

```bash
# 下载数据集
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# 解压
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

放置结构：
```
data/
├── images/
│   ├── Abyssinian_1.jpg
│   ├── ...
└── annotations/
    ├── annotations.txt
    └── ...
```

### 3. 训练模型

```bash
cd /Users/yudu/Documents/毕业设计/fine_grained_classification

# 基本训练
python train.py -c configs/dynamic_rn50.yaml

# 指定few-shot数量
python train.py -c configs/dynamic_rn50.yaml --shots 1
python train.py -c configs/dynamic_rn50.yaml --shots 4
python train.py -c configs/dynamic_rn50.yaml --shots 16

# 训练CoOp模型
python train.py -c configs/dynamic_rn50.yaml -t CoOp

# 使用不同的backbone
python train.py -c configs/dynamic_rn50.yaml --backbone ViT-B/32
```

### 4. 评估模型

```bash
python evaluate.py --model-dir output_fgd/oxford_pets/DynamicPromptTrainer/rn50
```

### 5. 启动演示

```bash
cd demo
streamlit run pet_classifier_demo.py
```

访问 `http://localhost:8501` 查看演示界面。

### 6. 启动API服务

```bash
cd web
python app.py
```

访问 `http://localhost:5000/api/health` 测试API。

## 配置说明

主要配置项（`configs/dynamic_rn50.yaml`）：

```yaml
TRAINER:
  DYNAMIC:
    CTX_INIT: "a photo of a"  # 初始提示词
    N_CTX: 16                  # 可学习token数量
    USE_DYNAMIC: true          # 使用动态优化
    USE_ADAPTIVE: true         # 使用自适应调整
    USE_DIFFICULTY_WEIGHT: true # 使用难度权重
    ALPHA: 0.1                 # 学习率
    BETA: 0.01                 # 正则化系数
    USE_SEMANTIC_ENHANCEMENT: false  # 语义增强（可选）
```

## 核心模块

### 1. 动态提示词优化 (`models/dynamic_prompt.py`)

- `DynamicPromptOptimizer`: 根据困难样本自适应调整提示词
- `DifficultyWeightCalculator`: 计算样本难度权重
- `SoftPromptAdapter`: 使用MLP动态生成提示词偏移
- `AdaptivePromptLearner`: 自适应提示词学习器

### 2. 品种语义增强 (`models/breed_semantic.py`)

- `BreedAttributeDatabase`: 品种属性数据库（37种猫狗品种）
- `SemanticEnhancer`: 语义增强模块

### 3. 自定义CLIP (`models/custom_clip.py`)

- `CustomCLIPDynamic`: 动态提示版本
- `CustomCLIPCoCoOp`: CoCoOp风格版本

## 创新点

1. **动态提示词调整机制**: 根据困难样本自适应修改提示词
2. **难度加权损失**: 对难分类样本给予更高权重
3. **品种语义增强**: 利用品种属性描述增强文本嵌入
4. **可视化决策解释**: 展示预测结果和置信度

## 实验结果

| 方法 | 1-shot | 4-shot | 16-shot |
|------|--------|--------|---------|
| Zero-shot CLIP | xx% | xx% | xx% |
| CoOp | xx% | xx% | xx% |
| CoCoOp | xx% | xx% | xx% |
| **Ours (Dynamic)** | xx% | xx% | xx% |

## 参考资料

- [CLIP: Connecting Text and Images](https://github.com/openai/CLIP)
- [CoOp: Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [CoCoOp: Conditional Context Optimization](https://github.com/KaiyangZhou/CoOp)
- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
