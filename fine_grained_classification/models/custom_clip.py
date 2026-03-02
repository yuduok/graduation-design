"""
自定义CLIP模型 - 整合动态提示和语义增强
Custom CLIP with Dynamic Prompt and Semantic Enhancement
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import importlib


def load_clip_to_cpu(cfg):
    """加载CLIP模型到CPU"""
    try:
        import clip
    except ImportError:
        raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    """CLIP文本编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 提取EOT token的特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIPDynamic(nn.Module):
    """
    自定义CLIP模型 - 动态提示 + 语义增强
    Custom CLIP with Dynamic Prompt Learning and Semantic Enhancement
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.dtype = clip_model.dtype
        
        # 导入动态提示学习器
        from .dynamic_prompt import AdaptivePromptLearner
        self.prompt_learner = AdaptivePromptLearner(cfg, classnames, clip_model)
        
        # 文本编码器
        self.text_encoder = TextEncoder(clip_model)
        
        # 图像编码器
        self.image_encoder = clip_model.visual
        
        # Logit scale
        self.logit_scale = clip_model.logit_scale
        
        # Tokenized prompts (用于确定序列长度)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 语义增强模块（可选）
        use_semantic = getattr(cfg.TRAINER.DYNAMIC, 'USE_SEMANTIC_ENHANCEMENT', False)
        if use_semantic:
            from .breed_semantic import SemanticEnhancer
            use_attr_db = getattr(cfg.TRAINER.DYNAMIC, 'USE_ATTRIBUTE_DATABASE', True)
            self.semantic_enhancer = SemanticEnhancer(clip_model, use_attribute_database=use_attr_db)
        else:
            self.semantic_enhancer = None
        
        # 困难负样本权重
        self.hard_negative_weights = None
    
    def forward(self, image, label=None):
        """
        前向传播
        Args:
            image: 输入图像 [batch_size, 3, H, W]
            label: 标签 (训练时可选)
        Returns:
            如果训练（有label）: logits [batch_size, n_cls]
            如果推理（无label）: logits [batch_size, n_cls]
        """
        # 1. 编码图像
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 2. 生成动态提示词
        if self.training and label is not None:
            # 训练模式：需要label和predictions
            prompts = self.prompt_learner(image_features, label, predictions=None)
        else:
            # 推理模式
            prompts = self.prompt_learner(image_features)
        
        # 3. 编码文本
        # prompts shape: [batch_size, n_cls, n_tkn, ctx_dim]
        batch_size, n_cls, n_tkn, _ = prompts.shape
        logits = []
        
        for i in range(batch_size):
            # 对每张图像，获取其对应的n_cls个提示词
            batch_prompts = prompts[i]  # [n_cls, n_tkn, ctx_dim]
            text_features = self.text_encoder(batch_prompts, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算logits
            logit_scale = self.logit_scale.exp()
            logit = logit_scale * image_features[i] @ text_features.T
            logits.append(logit)
        
        logits = torch.stack(logits)  # [batch_size, n_cls]
        
        # 4. 语义增强（如果启用）
        if self.semantic_enhancer is not None:
            logits = self._apply_semantic_enhancement(image_features, logits)
        
        # 5. 困难负样本加权（如果有）
        if self.hard_negative_weights is not None:
            logits = self._apply_hard_negative_weights(logits)
        
        return logits
    
    def _apply_semantic_enhancement(self, image_features, logits):
        """
        应用语义增强
        Args:
            image_features: [batch_size, feat_dim]
            logits: [batch_size, n_cls]
        Returns:
            enhanced_logits: [batch_size, n_cls]
        """
        # 这里可以添加语义增强的逻辑
        # 例如：使用品种相似度调整logits
        return logits
    
    def _apply_hard_negative_weights(self, logits):
        """
        应用困难负样本权重
        Args:
            logits: [batch_size, n_cls]
        Returns:
            weighted_logits: [batch_size, n_cls]
        """
        if self.hard_negative_weights is None:
            return logits
        
        # 对logits进行加权
        weighted_logits = logits * self.hard_negative_weights
        return weighted_logits
    
    def get_prompt_tokens(self, image_features):
        """
        获取提示词tokens（用于可视化）
        Args:
            image_features: [batch_size, feat_dim]
        Returns:
            prompt_tokens: [batch_size, n_cls, n_tkn, ctx_dim]
        """
        return self.prompt_learner(image_features)
    
    def get_difficulty_weights(self):
        """获取当前batch的难度权重（用于损失计算）"""
        return self.prompt_learner.current_weights


class CustomCLIPCoCoOp(nn.Module):
    """
    自定义CLIP模型 - CoCoOp风格（条件提示）
    Custom CLIP in CoCoOp Style.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.dtype = clip_model.dtype
        
        # 导入动态提示学习器
        from .dynamic_prompt import AdaptivePromptLearner
        self.prompt_learner = AdaptivePromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 文本编码器
        self.text_encoder = TextEncoder(clip_model)
        
        # 图像编码器
        self.image_encoder = clip_model.visual
        
        # Logit scale
        self.logit_scale = clip_model.logit_scale
    
    def forward(self, image, label=None):
        """
        前向传播 - CoCoOp风格
        Args:
            image: 输入图像 [batch_size, 3, H, W]
            label: 标签 (训练时可选)
        Returns:
            如果训练: loss
            如果推理: logits [batch_size, n_cls]
        """
        # 1. 编码图像
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 2. 生成动态提示词（条件于图像特征）
        predictions = None
        if self.training and label is not None:
            # 先用基础提示词计算预测结果
            base_prompts = self.prompt_learner(image_features)
            logits = self._compute_logits(image_features, base_prompts)
            predictions = logits
        
        # 生成自适应提示词
        prompts = self.prompt_learner(image_features, label, predictions)
        
        # 3. 计算最终logits
        final_logits = self._compute_logits(image_features, prompts)
        
        # 4. 如果训练，返回loss
        if self.training and label is not None:
            loss = F.cross_entropy(final_logits, label)
            return loss
        
        return final_logits
    
    def _compute_logits(self, image_features, prompts):
        """计算logits"""
        batch_size = image_features.shape[0]
        logits = []
        
        for i in range(batch_size):
            batch_prompts = prompts[i]  # [n_cls, n_tkn, ctx_dim]
            text_features = self.text_encoder(batch_prompts, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logit_scale = self.logit_scale.exp()
            logit = logit_scale * image_features[i] @ text_features.T
            logits.append(logit)
        
        logits = torch.stack(logits)  # [batch_size, n_cls]
        return logits


def build_custom_clip(cfg, classnames, clip_model, model_type="dynamic"):
    """
    构建自定义CLIP模型
    Args:
        cfg: 配置
        classnames: 类别名称
        clip_model: CLIP模型
        model_type: "dynamic" (动态提示) 或 "cocoop" (CoCoOp风格)
    Returns:
        model: 自定义CLIP模型
    """
    if model_type == "dynamic":
        model = CustomCLIPDynamic(cfg, classnames, clip_model)
    elif model_type == "cocoop":
        model = CustomCLIPCoCoOp(cfg, classnames, clip_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
