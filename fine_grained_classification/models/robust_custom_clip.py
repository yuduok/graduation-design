"""
鲁棒CLIP模型 - 集成对抗性防御
Robust CLIP Model with Adversarial Defense Integration

该模块将对抗性防御机制集成到自定义CLIP模型中，
使模型能够在推理时抵御对抗性攻击。
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from .custom_clip import TextEncoder, load_clip_to_cpu, build_custom_clip
from .adversarial_defense import (
    TestTimeCounterattack,
    AdversarialDetector,
    create_defense_system
)


class RobustCustomCLIP(nn.Module):
    """
    鲁棒自定义CLIP模型

    在 CustomCLIPDynamic 基础上集成对抗性防御机制
    支持三种模式:
    - "normal": 正常推理模式
    - "defend": 始终应用防御
    - "auto": 自动检测并防御（推荐）
    """

    def __init__(self, cfg, classnames, clip_model, defense_config=None):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.dtype = clip_model.dtype

        from .dynamic_prompt import AdaptivePromptLearner
        self.prompt_learner = AdaptivePromptLearner(cfg, classnames, clip_model)

        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        use_semantic = getattr(cfg.TRAINER.DYNAMIC, 'USE_SEMANTIC_ENHANCEMENT', False)
        if use_semantic:
            from .breed_semantic import SemanticEnhancer
            use_attr_db = getattr(cfg.TRAINER.DYNAMIC, 'USE_ATTRIBUTE_DATABASE', True)
            self.semantic_enhancer = SemanticEnhancer(clip_model, use_attribute_database=use_attr_db)
        else:
            self.semantic_enhancer = None

        if defense_config is None:
            defense_config = {}
        self.defense_system = create_defense_system(clip_model, defense_config)

        self.defense_mode = "auto"
        self.defense_enabled = True

    def forward(self, image, label=None, defense_mode=None):
        """
        前向传播

        Args:
            image: 输入图像 [batch_size, 3, H, W]
            label: 标签 (训练时可选)
            defense_mode: 防御模式 ("normal", "defend", "auto")
        Returns:
            logits [batch_size, n_cls]
            defense_info: 防御信息字典（如果应用了防御）
        """
        if defense_mode is not None:
            self.defense_mode = defense_mode

        defended_image = image
        defense_info = None

        if self.defense_enabled and self.defense_mode != "normal":
            if self.defense_mode == "auto":
                defended_image, defense_info = self.defense_system(image, mode="auto")
            elif self.defense_mode == "defend":
                defended_image = self.defense_system(image, mode="defend")
                defense_info = {"mode": "defend"}
        else:
            defense_info = {"mode": "normal", "defense_applied": False}

        image_features = self.image_encoder(defended_image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.training and label is not None:
            with torch.no_grad():
                base_prompts = self.prompt_learner(image_features)
                base_logits = self._encode_and_compute_logits(image_features, base_prompts)

            if (self.prompt_learner.dynamic_optimizer is not None
                and self.prompt_learner.dynamic_optimizer.difficulty_calculator is not None):
                self.prompt_learner.dynamic_optimizer.difficulty_calculator.update_prototypes(
                    image_features, label
                )

            prompts = self.prompt_learner(image_features, label, predictions=base_logits)
            logits = self._encode_and_compute_logits(image_features, prompts)
        else:
            prompts = self.prompt_learner(image_features)
            logits = self._encode_and_compute_logits(image_features, prompts)

        if self.semantic_enhancer is not None:
            logits = self._apply_semantic_enhancement(image_features, logits)

        if defense_info is not None:
            if defense_info.get("mode") != "normal":
                defense_info["defense_applied"] = True
                if "tau" in defense_info:
                    defense_info["tau_mean"] = defense_info["tau"].mean().item()
        else:
            defense_info = {"defense_applied": False}

        return logits, defense_info

    def _encode_and_compute_logits(self, image_features, prompts):
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
        return logits

    def set_defense_mode(self, mode):
        """设置防御模式"""
        valid_modes = ["normal", "defend", "auto"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid defense mode: {mode}. Must be one of {valid_modes}")
        self.defense_mode = mode
        print(f"[RobustCustomCLIP] Defense mode set to: {mode}")

    def set_defense_enabled(self, enabled):
        """启用/禁用防御"""
        self.defense_enabled = enabled
        print(f"[RobustCustomCLIP] Defense enabled: {enabled}")

    def detect_adversarial(self, image):
        """
        检测图像是否为对抗性图像

        Args:
            image: 输入图像
        Returns:
            is_adversarial: 布尔张量
            confidence: 置信度
            tau: 特征差异比率
        """
        return self.defense_system.detect(image)


class RobustCustomCLIPCoCoOp(nn.Module):
    """
    鲁棒 CoCoOp 风格模型

    在 CustomCLIPCoCoOp 基础上集成对抗性防御机制
    """

    def __init__(self, cfg, classnames, clip_model, defense_config=None):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.dtype = clip_model.dtype

        from .dynamic_prompt import AdaptivePromptLearner
        self.prompt_learner = AdaptivePromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        if defense_config is None:
            defense_config = {}
        self.defense_system = create_defense_system(clip_model, defense_config)

        self.defense_mode = "auto"
        self.defense_enabled = True

    def forward(self, image, label=None, defense_mode=None):
        """前向传播"""
        if defense_mode is not None:
            self.defense_mode = defense_mode

        defended_image = image
        defense_info = None

        if self.defense_enabled and self.defense_mode != "normal":
            if self.defense_mode == "auto":
                defended_image, defense_info = self.defense_system(image, mode="auto")
            elif self.defense_mode == "defend":
                defended_image = self.defense_system(image, mode="defend")
                defense_info = {"mode": "defend"}
        else:
            defense_info = {"mode": "normal", "defense_applied": False}

        image_features = self.image_encoder(defended_image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        predictions = None
        if self.training and label is not None:
            base_prompts = self.prompt_learner(image_features)
            logits = self._compute_logits(image_features, base_prompts)
            predictions = logits

        prompts = self.prompt_learner(image_features, label, predictions)
        final_logits = self._compute_logits(image_features, prompts)

        if self.training and label is not None:
            loss = F.cross_entropy(final_logits, label)
            return loss, {"defense_info": defense_info}

        if defense_info is not None:
            if defense_info.get("mode") != "normal":
                defense_info["defense_applied"] = True

        return final_logits, defense_info

    def _compute_logits(self, image_features, prompts):
        batch_size = image_features.shape[0]
        logits = []

        for i in range(batch_size):
            batch_prompts = prompts[i]
            text_features = self.text_encoder(batch_prompts, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logit = logit_scale * image_features[i] @ text_features.T
            logits.append(logit)

        logits = torch.stack(logits)
        return logits

    def set_defense_mode(self, mode):
        valid_modes = ["normal", "defend", "auto"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid defense mode: {mode}")
        self.defense_mode = mode

    def set_defense_enabled(self, enabled):
        self.defense_enabled = enabled


def build_robust_clip(cfg, classnames, clip_model, model_type="dynamic", defense_config=None):
    """
    构建鲁棒CLIP模型

    Args:
        cfg: 配置
        classnames: 类别名称
        clip_model: CLIP模型
        model_type: "dynamic" 或 "cocoop"
        defense_config: 防御配置字典

    Returns:
        model: 鲁棒CLIP模型
    """
    if model_type == "dynamic":
        model = RobustCustomCLIP(cfg, classnames, clip_model, defense_config)
    elif model_type == "cocoop":
        model = RobustCustomCLIPCoCoOp(cfg, classnames, clip_model, defense_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
