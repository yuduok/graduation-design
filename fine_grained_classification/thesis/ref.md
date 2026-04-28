# 论文引用核对与框架梳理

## 1. 实际编译结构

`main.tex` 当前实际包含的文件是：

- `abstract.tex`
- `chapter1.tex`
- `chapter2.tex`
- `chapter3.tex`
- `chapter4.tex`
- `chapter5.tex`
- `chapter6_visualization.tex`
- `chapter6.tex`

说明：

- `chapters/abstract.tex` 和 `chapters/ch1_intro.tex` 目前是目录中的备用/旧版文件，不在 `main.tex` 的实际编译链路里。

## 2. 术语替换情况

已将正文中出现的 `DifficultyWeightCalculator` 统一改为中文表述：

- 中文正文统一为：`难度权重计算器`
- 英文摘要中改为：`difficulty weight calculator`

## 3. 正文中的引用位置（谁引用了谁）

下面按正文出现位置列出 `\cite{}` 的实际引用点。

### `chapter1.tex`

- 第 5 行：`parkhi2012cats`，`atabuzzaman2025zero`
- 第 7 行：`snell2017prototypical`，`finn2017model`
- 第 9 行：`wang2021fgmi`
- 第 18 行：`krizhevsky2012imagenet`，`simonyan2015very`，`he2016deep`
- 第 20 行：`parkhi2012cats`，`oxfordpetsweb`，`cuizhang2023classification`，`wang2024enhancing`
- 第 22 行：`clip2021`
- 第 24 行：`gu2023prompt`，`chen2023vlp`，`coop2022`，`cocoop2022`，`liu2025fgmpl`
- 第 32 行：`parkhi2012cats`
- 第 34 行：`cuizhang2023classification`，`wang2024enhancing`
- 第 36 行：`atabuzzaman2025zero`
- 第 40 行：`clip2021`
- 第 40 行：`chen2023vlp`，`gu2023prompt`
- 第 42 行：`coop2022`
- 第 44 行：`cocoop2022`
- 第 46 行：`qu2025proapo`，`liu2025fgmpl`，`choi2025multimodal`
- 第 56 行：`qu2025proapo`，`liu2025fgmpl`，`choi2025multimodal`

### `chapter2.tex`

- 第 5 行：`clip2021`
- 第 7 行：`clip2021`，`chen2023vlp`
- 第 9 行：`clip2021`
- 第 15 行：`coop2022`
- 第 23 行：`coop2022`
- 第 29 行：`cocoop2022`
- 第 37 行：`cocoop2022`
- 第 41 行：`tipadapter2022`，`clipadapter2022`
- 第 47 行：`xing2025ttc`
- 第 61 行：`lin2017focal`

### `chapter3.tex`

- 第 122 行：`xu2023fewshotvit`

### `chapter4.tex`

- 第 11 行：`xing2025ttc`

### `chapter5.tex`

- 第 13 行：`parkhi2012cats`

### 其他已编译章节

- `abstract.tex`、`chapter6_visualization.tex`、`chapter6.tex` 中当前没有 `\cite{}` 引用。

## 4. 参考文献条目被引用情况（谁被引用了）

下面按 `ref.bib` 条目检查“是否被正文引用”。

| BibKey | 是否被引用 | 被引用位置 |
|---|---|---|
| `clip2021` | 是 | `chapter1.tex:22,40`; `chapter2.tex:5,7,9` |
| `coop2022` | 是 | `chapter1.tex:24,42`; `chapter2.tex:15,23` |
| `cocoop2022` | 是 | `chapter1.tex:24,44`; `chapter2.tex:29,37` |
| `he2016deep` | 是 | `chapter1.tex:18` |
| `parkhi2012cats` | 是 | `chapter1.tex:5,20,32`; `chapter5.tex:13` |
| `wang2024enhancing` | 是 | `chapter1.tex:20,34` |
| `gu2023prompt` | 是 | `chapter1.tex:24,40` |
| `cuizhang2023classification` | 是 | `chapter1.tex:20,34` |
| `atabuzzaman2025zero` | 是 | `chapter1.tex:5,36` |
| `choi2025multimodal` | 是 | `chapter1.tex:46,56` |
| `xing2025ttc` | 是 | `chapter2.tex:47`; `chapter4.tex:11` |
| `liu2025fgmpl` | 是 | `chapter1.tex:24,46,56` |
| `qu2025proapo` | 是 | `chapter1.tex:46,56` |
| `chen2023vlp` | 是 | `chapter1.tex:24,40`; `chapter2.tex:7` |
| `oxfordpetsweb` | 是 | `chapter1.tex:20` |
| `xu2023fewshotvit` | 是 | `chapter3.tex:122` |
| `tipadapter2022` | 是 | `chapter2.tex:41` |
| `lin2017focal` | 是 | `chapter2.tex:61` |
| `snell2017prototypical` | 是 | `chapter1.tex:7` |
| `finn2017model` | 是 | `chapter1.tex:7` |
| `clipadapter2022` | 是 | `chapter2.tex:41` |
| `krizhevsky2012imagenet` | 是 | `chapter1.tex:18` |
| `simonyan2015very` | 是 | `chapter1.tex:18` |
| `wang2021fgmi` | 是 | `chapter1.tex:9` |

结论：

- `ref.bib` 当前共有 24 条文献条目。
- 这 24 条条目全部都已经在正文中被 `\cite{}` 引用。
- 目前没有发现“写进 `ref.bib` 但正文完全没有引用”的条目。

## 5. 论文整体框架与模块梳理

按当前论文结构，系统核心模块可以概括为：

- 冻结的 CLIP 主干
- 图像编码器
- 文本编码器
- 基础可学习上下文提示
- 类别自适应因子
- 图像条件提示偏移分支（两层 MLP）
- 难度权重计算器
- 加权交叉熵损失
- 自适应训练轮数策略
- TTC 测试时防御模块
- Streamlit 可视化演示界面

如果只画“方法主干架构图”，建议突出以下 7 个主模块：

1. 输入图像
2. 冻结 CLIP 视觉编码器
3. 基础提示上下文 + 类别名称
4. 图像条件提示偏移分支
5. 类别自适应因子
6. 文本编码器与图文相似度分类
7. 难度权重计算器 + 加权损失

## 6. 生图提示词

下面这版提示词适合生成“论文方法总架构图/技术路线图”。

### 中文提示词

绘制一张本科毕业论文风格的深度学习系统架构图，主题是“基于提示词优化的细粒度猫狗分类方法”。整体风格要求学术、规范、简洁、适合直接放入论文。画面采用横向流程图布局，白色背景，蓝灰色和浅橙色为主色，模块边框清晰，箭头明确，字体规整，具有中文论文插图风格。

图中包含以下模块，并按数据流从左到右排列：

1. 输入图像（猫狗宠物照片）
2. 冻结的 CLIP 视觉编码器
3. 提取图像特征
4. 基础可学习上下文提示，与类别名称拼接
5. 图像条件提示偏移分支（两层 MLP，512→32→512）
6. 类别自适应因子，对上下文进行缩放
7. 动态提示生成
8. CLIP 文本编码器
9. 图文特征相似度计算
10. 分类 logits 输出
11. 难度权重计算器，根据真实类别概率、预测置信度、误分类情况生成样本权重
12. 加权交叉熵损失
13. 训练反馈箭头回到提示相关模块

图中额外用侧边模块或下方注释标出：

- 自适应训练轮数策略（1-shot 到 16-shot）
- TTC 测试时防御模块（作为推理阶段安全扩展）
- Streamlit 可视化界面（作为系统展示层）

要求：

- 突出“提示生成 + 难样本加权”是核心创新
- 主干网络冻结，只更新提示相关参数
- 图中使用少量数学符号点缀，例如 softmax、logits、weighted loss
- 风格接近高质量中文学术论文插图、AI 论文方法图、信息图式神经网络结构图
- 版式清爽，模块层次分明，适合打印和答辩展示

### 英文提示词

Create a clean academic system architecture diagram for an undergraduate thesis on fine-grained cat and dog classification based on prompt optimization. Use a horizontal pipeline layout, white background, blue-gray and light orange color palette, clear module boxes, thin arrows, neat typography, and a formal paper-ready style.

Include these modules from left to right:
input pet image, frozen CLIP visual encoder, image feature extraction, learnable prompt context plus class names, image-conditional prompt offset branch with a two-layer MLP (512 to 32 to 512), class-adaptive scaling factors, dynamic prompt generation, CLIP text encoder, image-text similarity computation, classification logits, difficulty weight calculator, and weighted cross-entropy loss.

Show that the difficulty weight calculator uses true-class probability, prediction confidence, and misclassification status to generate sample weights. Add a feedback path from weighted loss back to the prompt-related trainable modules. Clearly emphasize that the CLIP backbone is frozen and only prompt-related parameters are updated.

Also add side modules or annotations for adaptive epoch scheduling across 1-shot to 16-shot settings, TTC test-time defense as an inference-time security extension, and a Streamlit visualization interface as the demo layer.

The figure should emphasize that the main innovation is the combination of dynamic prompting and hard-sample weighting. Style should resemble a high-quality deep learning method figure from a Chinese academic thesis: minimal, precise, structured, elegant, and presentation-ready.
