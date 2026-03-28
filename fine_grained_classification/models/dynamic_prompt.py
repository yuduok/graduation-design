"""
动态提示词优化模块
Dynamic Prompt Optimizer for Fine-Grained Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class DifficultyWeightCalculator:
    """
    难度权重计算模块
    - 特征空间中远离中心的样本权重更高
    - 被频繁误分类的样本权重更高
    """
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.breed_prototypes = {}
        self.misclassification_history = {}
        
    def update_prototypes(self, features, labels):
        """更新每个类别的原型（特征中心）"""
        prototypes = {}
        for feat, label in zip(features, labels):
            label_idx = label.item()
            if label_idx not in prototypes:
                prototypes[label_idx] = []
            prototypes[label_idx].append(feat)
        
        for label_idx, feats in prototypes.items():
            center = torch.mean(torch.stack(feats), dim=0)
            if label_idx in self.breed_prototypes:
                # 动量更新
                self.breed_prototypes[label_idx] = (
                    self.momentum * self.breed_prototypes[label_idx] + 
                    (1 - self.momentum) * center
                )
            else:
                self.breed_prototypes[label_idx] = center.detach()
    
    def compute_weights(self, features, labels, predictions=None):
        """
        计算样本难度权重
        Args:
            features: 图像特征 [batch_size, feature_dim]
            labels: 真实标签 [batch_size]
            predictions: 预测结果 [batch_size, num_classes] (可选)
        Returns:
            weights: 每个样本的权重 [batch_size]
        """
        weights = []

        for i, (feat, label) in enumerate(zip(features, labels)):
            label_idx = label.item()

            # 基础权重为1
            weight = 1.0

            # 1. 特征距离权重：特征空间中距离类别中心越远，权重越高
            if label_idx in self.breed_prototypes:
                distance = torch.norm(feat - self.breed_prototypes[label_idx])
                weight += 0.5 * torch.sigmoid(distance - 1.0)

            # 2. 误分类权重（如果有 predictions）
            if predictions is not None:
                pred_idx = predictions[i].argmax().item()
                if label_idx != pred_idx:
                    weight *= 1.5

                    # 记录误分类历史
                    if label_idx not in self.misclassification_history:
                        self.misclassification_history[label_idx] = {}
                    if pred_idx not in self.misclassification_history[label_idx]:
                        self.misclassification_history[label_idx][pred_idx] = 0
                    self.misclassification_history[label_idx][pred_idx] += 1

            # 限制权重范围
            weight = max(0.5, min(2.0, weight))
            weights.append(weight)

        return torch.tensor(weights, device=features.device)


class DynamicPromptOptimizer(nn.Module):
    """
    动态提示词优化器
    - 根据困难样本自适应调整提示词
    - 困难样本加权机制
    - 自适应学习率调整
    """
    def __init__(self, n_ctx=16, ctx_dim=512, alpha=0.1, beta=0.01, 
                 use_difficulty_weight=True, momentum=0.9):
        super().__init__()
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.alpha = alpha  # 学习率
        self.beta = beta    # 正则化系数
        self.use_difficulty_weight = use_difficulty_weight
        
        # 可学习的上下文 tokens
        self.ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim) * 0.02)
        
        # 难度权重计算器
        if use_difficulty_weight:
            self.difficulty_calculator = DifficultyWeightCalculator(momentum)
        else:
            self.difficulty_calculator = None
        
        # 类别特定的自适应因子
        self.class_adaptive_factors = nn.Parameter(torch.ones(1, n_ctx, ctx_dim))
        
    def forward(self, features, labels=None, predictions=None):
        """
        前向传播和自适应更新
        Args:
            features: 图像特征 [batch_size, feature_dim]
            labels: 真实标签 (训练时需要)
            predictions: 预测结果 (训练时需要)
        Returns:
            ctx: 上下文向量
            weights: 难度权重 (训练时) 或 None (推理时)
        """
        if self.training and labels is not None:
            if self.use_difficulty_weight and self.difficulty_calculator:
                # 更新原型
                self.difficulty_calculator.update_prototypes(features, labels)
                # 计算权重（如果有 predictions 用 predictions，否则仅用原型距离）
                weights = self.difficulty_calculator.compute_weights(
                    features, labels, predictions
                )
                return self.ctx, weights

        return self.ctx, None
    
    def get_adaptive_context(self, batch_size, n_cls):
        """
        获取自适应的上下文（带类别自适应因子）
        """
        ctx = self.ctx.unsqueeze(0)  # [1, n_ctx, ctx_dim]
        
        # 应用类别自适应因子（简单实现）
        ctx = ctx * self.class_adaptive_factors
        
        # 扩展到批次和类别维度
        ctx_expanded = ctx.unsqueeze(0).expand(batch_size, n_cls, -1, -1)
        # [batch_size, n_cls, n_ctx, ctx_dim]
        
        return ctx_expanded


class SoftPromptAdapter(nn.Module):
    """
    软提示词适配器
    - 使用轻量级MLP动态生成提示词偏移
    - 根据输入图像特征自适应调整
    """
    def __init__(self, vis_dim=512, ctx_dim=512, hidden_dim=64):
        super().__init__()
        self.vis_dim = vis_dim
        self.ctx_dim = ctx_dim
        
        # 轻量级元网络
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, hidden_dim)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_dim, ctx_dim))
        ]))
        
        # 初始化
        nn.init.kaiming_normal_(self.meta_net[0].weight)
        nn.init.zeros_(self.meta_net[0].bias)
        nn.init.kaiming_normal_(self.meta_net[2].weight)
        nn.init.zeros_(self.meta_net[2].bias)
    
    def forward(self, image_features, base_ctx, n_cls):
        """
        根据图像特征动态调整提示词
        Args:
            image_features: [batch_size, vis_dim]
            base_ctx: [n_ctx, ctx_dim] 基础上下文
            n_cls: 类别数量
        Returns:
            adaptive_ctx: [batch_size, n_cls, n_ctx, ctx_dim] 自适应上下文
        """
        batch_size = image_features.shape[0]
        
        # 计算图像特征对应的偏移
        bias = self.meta_net(image_features)  # [batch_size, ctx_dim]
        bias = bias.unsqueeze(1)  # [batch_size, 1, ctx_dim]
        
        # 基础上下文扩展
        ctx = base_ctx.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_ctx, ctx_dim]
        
        # 添加偏移
        ctx_shifted = ctx + bias  # [batch_size, n_ctx, ctx_dim]
        
        # 为每个类别复制相同的自适应上下文
        ctx_shifted = ctx_shifted.unsqueeze(1).expand(-1, n_cls, -1, -1)
        # [batch_size, n_cls, n_ctx, ctx_dim]
        
        return ctx_shifted


class AdaptivePromptLearner(nn.Module):
    """
    自适应提示词学习器（结合动态优化和软提示）
    - 基础上下文：可学习的静态提示
    - 动态调整：基于图像特征的自适应偏移
    - 困难样本：加权关注
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.dtype = clip_model.dtype
        
        # 从配置获取参数
        n_ctx = getattr(cfg.TRAINER.DYNAMIC, 'N_CTX', 16)
        ctx_init = getattr(cfg.TRAINER.DYNAMIC, 'CTX_INIT', "a photo of a")
        vis_dim = clip_model.visual.output_dim
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # 初始化上下文
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            from clip import clip
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.ctx = nn.Parameter(ctx_vectors)
        self.n_ctx = n_ctx
        
        # 当前难度权重（用于损失计算）
        self.current_weights = None
        
        # 动态优化器
        use_dynamic = getattr(cfg.TRAINER.DYNAMIC, 'USE_DYNAMIC', True)
        use_adaptive = getattr(cfg.TRAINER.DYNAMIC, 'USE_ADAPTIVE', True)
        
        if use_dynamic:
            self.dynamic_optimizer = DynamicPromptOptimizer(
                n_ctx=n_ctx, 
                ctx_dim=ctx_dim,
                alpha=getattr(cfg.TRAINER.DYNAMIC, 'ALPHA', 0.1),
                beta=getattr(cfg.TRAINER.DYNAMIC, 'BETA', 0.01),
                use_difficulty_weight=getattr(cfg.TRAINER.DYNAMIC, 'USE_DIFFICULTY_WEIGHT', True)
            )
        else:
            self.dynamic_optimizer = None
        
        if use_adaptive:
            hidden_dim = getattr(cfg.TRAINER.DYNAMIC, 'ADAPTIVE_HIDDEN_DIM', 64)
            self.soft_prompt_adapter = SoftPromptAdapter(
                vis_dim=vis_dim,
                ctx_dim=ctx_dim,
                hidden_dim=hidden_dim
            )
        else:
            self.soft_prompt_adapter = None
        
        # Tokenize类名
        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
    
    def forward(self, image_features, labels=None, predictions=None):
        """
        前向传播
        Args:
            image_features: [batch_size, vis_dim]
            labels: (可选) 用于动态优化
            predictions: (可选) 用于动态优化
        Returns:
            prompts: [batch_size, n_cls, n_tkn, ctx_dim]
        """
        # 获取基础上下文
        base_ctx = self.ctx  # [n_ctx, ctx_dim]

        # 应用类别自适应因子（如果有动态优化器）
        if self.dynamic_optimizer is not None:
            # class_adaptive_factors: [1, n_ctx, ctx_dim]
            base_ctx = base_ctx * self.dynamic_optimizer.class_adaptive_factors.squeeze(0)

            # 训练模式：计算难度权重
            if self.training:
                _ctx, weights = self.dynamic_optimizer(image_features, labels, predictions)
                if weights is not None:
                    self.current_weights = weights
                else:
                    self.current_weights = None
            else:
                self.current_weights = None
        else:
            self.current_weights = None

        # 如果启用自适应调整
        if self.soft_prompt_adapter is not None:
            ctx_shifted = self.soft_prompt_adapter(image_features, base_ctx, self.n_cls)
        else:
            batch_size = image_features.shape[0]
            ctx = base_ctx.unsqueeze(0).unsqueeze(0)  # [1, 1, n_ctx, ctx_dim]
            ctx_shifted = ctx.expand(batch_size, self.n_cls, -1, -1)

        # 构建完整提示词
        prefix = self.token_prefix.unsqueeze(0)  # [1, n_cls, 1, ctx_dim]
        suffix = self.token_suffix.unsqueeze(0)  # [1, n_cls, *, ctx_dim]

        batch_size = ctx_shifted.shape[0]
        prefix = prefix.expand(batch_size, -1, -1, -1)
        suffix = suffix.expand(batch_size, -1, -1, -1)

        # 拼接: [SOS, context tokens, CLS, EOS]
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=2)
        # [batch_size, n_cls, n_tkn, ctx_dim]

        return prompts
