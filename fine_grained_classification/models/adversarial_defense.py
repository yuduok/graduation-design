"""
对抗性防御模块 - Test-time Counterattack (TTC) 风格防御
Adversarial Defense Module - Test-time Counterattack Style Protection

基于论文: "CLIP is Strong Enough to Fight Back: Test-time Counterattacks
towards Zero-shot Adversarial Robustness of CLIP" (CVPR 2025)

该模块实现了在推理时对抗对抗性扰动的防御机制。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as safe


class TestTimeCounterattack(nn.Module):
    """
    测试时反攻击防御模块

    核心思想：利用CLIP预训练视觉编码器在推理时对抗对抗性图像，
    通过比较原始图像和扰动图像的特征差异来检测和防御攻击。

    防御策略：
    1. 对输入图像应用小扰动（反向攻击）
    2. 比较原始特征与扰动后特征的差异比率 (tau)
    3. 根据差异程度自适应调整防御强度
    """

    def __init__(
        self,
        clip_visual_encoder,
        eps=4.0 / 255.0,
        num_steps=2,
        step_size=1.0 / 255.0,
        tau_threshold=0.2,
        beta=2.0,
        norm="l_inf"
    ):
        super().__init__()
        self.clip_visual = clip_visual_encoder
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.tau_threshold = tau_threshold
        self.beta = beta
        self.norm = norm

        self.lower_limit = 0.0
        self.upper_limit = 1.0

    def _clamp(self, X):
        return torch.max(
            torch.min(X, self.upper_limit), self.lower_limit
        )

    def _compute_tau(self, orig_feat, noisy_feat):
        """
        计算特征差异比率 tau

        tau = ||noisy_feat - orig_feat|| / ||orig_feat||

        tau较大表示图像可能受到攻击
        """
        diff = noisy_feat - orig_feat
        diff_ratio = diff.norm(dim=-1) / (orig_feat.norm(dim=-1) + 1e-10)
        return diff_ratio

    def _compute_ori_features(self, images, prompt_token=None):
        """计算原始图像的特征表示"""
        with torch.no_grad():
            features = self.clip_visual(images, prompt_token)
        return features

    def _apply_counterattack(self, images, prompt_token=None):
        """
        应用反攻击扰动

        关键洞察：如果图像是对抗性的，特征差异会较大。
        通过在特征空间中反向优化，可以减少对抗性影响。
        """
        X = images.detach().clone()

        delta = torch.zeros_like(X)
        if self.norm == "l_inf":
            delta.uniform_(-self.eps, self.eps)
        elif self.norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps

        delta = self._clamp(delta - X)
        delta.requires_grad = True

        X_ori_reps = self._compute_ori_features(X, prompt_token)
        X_ori_norm = X_ori_reps.norm(dim=-1)

        deltas_per_step = []
        deltas_per_step.append(delta.data.clone())

        for _ in range(self.num_steps):
            prompted_images = X + delta
            X_att_reps = self._compute_ori_features(prompted_images, prompt_token)

            l2_loss = ((X_att_reps - X_ori_reps) ** 2).sum(1).sum()

            grad = torch.autograd.grad(l2_loss, delta)[0]

            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = X[:, :, :, :]

            if self.norm == "l_inf":
                d = torch.clamp(
                    d + self.step_size * torch.sign(g),
                    min=-self.eps,
                    max=self.eps
                )
            elif self.norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * self.step_size).view(d.size(0), -1).renorm(
                    p=2, dim=0, maxnorm=self.eps
                ).view_as(d)

            d = self._clamp(d - x)
            delta.data[:, :, :, :] = d
            deltas_per_step.append(delta.data.clone())

        Delta = torch.stack(deltas_per_step, dim=1)

        diff_ratio = self._compute_tau(
            X_ori_reps,
            self._compute_ori_features(X + delta, prompt_token)
        )

        scheme_sign = (self.tau_threshold - diff_ratio).sign()

        weights = torch.arange(self.num_steps + 1, device=X.device)
        weights = weights.unsqueeze(0).expand(X.size(0), -1).float()
        weights = torch.exp(scheme_sign.view(-1, 1) * weights * self.beta)
        weights = weights / weights.sum(dim=1, keepdim=True)

        weights_hard = torch.zeros_like(weights)
        weights_hard[:, 0] = 1.0
        weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
        weights = weights.view(X.size(0), self.num_steps + 1, 1, 1, 1)

        Delta = (weights * Delta).sum(dim=1)

        return Delta

    def forward(self, images, prompt_token=None):
        """
        前向传播：应用防御机制

        Args:
            images: 输入图像 [batch_size, 3, H, W]
            prompt_token: 可选的提示token

        Returns:
            defended_images: 防御后的图像
            tau: 特征差异比率（用于判断是否为对抗性图像）
        """
        X = images.detach().clone()

        X_ori_reps = self._compute_ori_features(X, prompt_token)

        delta = self._apply_counterattack(X, prompt_token)

        X_att_reps = self._compute_ori_features(
            self._clamp(X + delta), prompt_token
        )
        tau = self._compute_tau(X_ori_reps, X_att_reps)

        defended_images = self._clamp(X + delta)

        return defended_images, tau


class AdversarialDetector(nn.Module):
    """
    对抗性检测器

    用于检测输入图像是否为对抗性图像
    基于特征差异比率进行判断
    """

    def __init__(
        self,
        clip_visual_encoder,
        eps=4.0 / 255.0,
        num_steps=2,
        step_size=1.0 / 255.0,
        tau_threshold=0.2
    ):
        super().__init__()
        self.clip_visual = clip_visual_encoder
        self.eps = eps
        self.num_steps = num_steps
        self.step_size = step_size
        self.tau_threshold = tau_threshold

        self.lower_limit = 0.0
        self.upper_limit = 1.0

    def _clamp(self, X):
        return torch.max(
            torch.min(X, self.upper_limit), self.lower_limit
        )

    def _compute_tau(self, orig_feat, noisy_feat):
        diff = noisy_feat - orig_feat
        diff_ratio = diff.norm(dim=-1) / (orig_feat.norm(dim=-1) + 1e-10)
        return diff_ratio

    def _add_light_noise(self, images):
        """添加轻量随机扰动用于检测"""
        delta = torch.zeros_like(images)
        delta.uniform_(-self.eps, self.eps)
        return self._clamp(images + delta)

    def forward(self, images, prompt_token=None):
        """
        检测图像是否为对抗性图像

        Args:
            images: 输入图像 [batch_size, 3, H, W]
            prompt_token: 可选的提示token

        Returns:
            is_adversarial: 布尔张量，指示是否为对抗性图像
            confidence: 置信度 [0, 1]
            tau: 实际计算的特征差异比率
        """
        with torch.no_grad():
            orig_features = self.clip_visual(images, prompt_token)

            noisy_images = self._add_light_noise(images)
            noisy_features = self.clip_visual(noisy_images, prompt_token)

            tau = self._compute_tau(orig_features, noisy_features)

            is_adversarial = tau > self.tau_threshold

            confidence = torch.sigmoid((tau - self.tau_threshold) / 0.1)

        return is_adversarial, confidence, tau


class RobustPromptLearner(nn.Module):
    """
    鲁棒提示学习器

    在标准提示学习器基础上增加对抗性防御机制
    """

    def __init__(
        self,
        base_prompt_learner,
        clip_visual_encoder,
        defense_config=None
    ):
        super().__init__()
        self.base_prompt_learner = base_prompt_learner
        self.device = next(clip_visual_encoder.parameters()).device

        if defense_config is None:
            defense_config = {
                "eps": 4.0 / 255.0,
                "num_steps": 2,
                "step_size": 1.0 / 255.0,
                "tau_threshold": 0.2,
                "beta": 2.0
            }

        self.defense = TestTimeCounterattack(
            clip_visual_encoder, **defense_config
        )

        self.use_defense = True
        self.defense_threshold = 0.3

    def forward(self, image_features, label=None, predictions=None):
        """
        前向传播 - 先应用防御再进行提示学习
        """
        if self.use_defense and hasattr(self, '_last_images'):
            defended_images, tau = self.defense(
                self._last_images, prompt_token=None
            )
            self._last_tau = tau
        else:
            self._last_tau = None

        return self.base_prompt_learner(image_features, label, predictions)

    def set_defense_enabled(self, enabled):
        """启用/禁用防御机制"""
        self.use_defense = enabled


def create_defense_system(clip_model, defense_config=None):
    """
    创建完整的对抗性防御系统

    Args:
        clip_model: CLIP模型
        defense_config: 防御配置字典

    Returns:
        defense_system: 包含检测和防御功能的模块
    """
    if defense_config is None:
        defense_config = {}

    defense = TestTimeCounterattack(
        clip_model.visual,
        eps=defense_config.get("eps", 4.0 / 255.0),
        num_steps=defense_config.get("num_steps", 2),
        step_size=defense_config.get("step_size", 1.0 / 255.0),
        tau_threshold=defense_config.get("tau_threshold", 0.2),
        beta=defense_config.get("beta", 2.0)
    )

    detector = AdversarialDetector(
        clip_model.visual,
        eps=defense_config.get("eps", 4.0 / 255.0),
        num_steps=defense_config.get("num_steps", 2),
        step_size=defense_config.get("step_size", 1.0 / 255.0),
        tau_threshold=defense_config.get("tau_threshold", 0.2)
    )

    class DefenseSystem(nn.Module):
        def __init__(self, defense, detector):
            super().__init__()
            self.defense = defense
            self.detector = detector

        def detect(self, images):
            return self.detector(images)

        def defend(self, images):
            defended, tau = self.defense(images)
            return defended

        def forward(self, images, mode="auto"):
            if mode == "auto":
                is_adv, confidence, tau = self.detect(images)
                if is_adv.any():
                    defended = self.defend(images)
                    return defended, {
                        "is_adversarial": is_adv,
                        "confidence": confidence,
                        "tau": tau
                    }
                return images, None
            elif mode == "detect":
                return self.detect(images)
            elif mode == "defend":
                return self.defend(images)
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return DefenseSystem(defense, detector)
