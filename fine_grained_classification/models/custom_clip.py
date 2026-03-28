"""
自定义CLIP模型 - 整合动态提示和语义增强
Custom CLIP with Dynamic Prompt and Semantic Enhancement
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import importlib

# 使用CoOp的clip模块
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


def load_clip_to_cpu(cfg):
    """加载CLIP模型到CPU"""
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
    
    def forward(self, image, label=None):
        """
        前向传播
        Args:
            image: 输入图像 [batch_size, 3, H, W]
            label: 标签 (训练时可选)
        Returns:
            logits [batch_size, n_cls]
        """
        # 1. 编码图像
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. 生成动态提示词（单阶段前向，稳定训练）
        if self.training and label is not None:
            # 训练模式：生成提示词并更新原型
            prompts = self.prompt_learner(image_features, label, predictions=None)
            
            # 更新类别原型（不计算权重，保持训练稳定）
            if (self.prompt_learner.dynamic_optimizer is not None
                and self.prompt_learner.dynamic_optimizer.difficulty_calculator is not None):
                self.prompt_learner.dynamic_optimizer.difficulty_calculator.update_prototypes(
                    image_features, label
                )
            
            logits = self._encode_and_compute_logits(image_features, prompts)
        else:
            # 推理模式
            prompts = self.prompt_learner(image_features)
            logits = self._encode_and_compute_logits(image_features, prompts)

        # 语义增强（如果启用）
        if self.semantic_enhancer is not None:
            logits = self._apply_semantic_enhancement(image_features, logits)

        return logits

    def _encode_and_compute_logits(self, image_features, prompts):
        """编码提示词并计算 logits"""
        batch_size, n_cls, n_tkn, _ = prompts.shape

        prompts_flat = prompts.view(-1, n_tkn, prompts.size(-1))
        tokenized_prompts_batch = self.tokenized_prompts.unsqueeze(0).expand(
            batch_size, -1, -1
        ).reshape(batch_size * n_cls, -1)

        text_features = self.text_encoder(prompts_flat, tokenized_prompts_batch)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(batch_size, n_cls, -1)

        logit_scale = self.logit_scale.exp()
        image_features_3d = image_features.unsqueeze(1)
        logits = logit_scale * (image_features_3d @ text_features.transpose(1, 2)).squeeze(1)

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
