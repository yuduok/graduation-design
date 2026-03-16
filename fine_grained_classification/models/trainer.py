"""
动态提示词训练器
Trainer for Fine-Grained Classification with Dynamic Prompts
"""
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .custom_clip import load_clip_to_cpu, build_custom_clip


@TRAINER_REGISTRY.register()
class DynamicPromptTrainer(TrainerX):
    """
    动态提示词训练器
    整合CoOp/CoCoOp + 动态优化 + 语义增强
    """
    
    def check_cfg(self, cfg):
        """检查配置"""
        # 确保有DYNAMIC配置节
        assert hasattr(cfg.TRAINER, 'DYNAMIC'), "Missing TRAINER.DYNAMIC config"
        
        # 检查精度设置
        prec = getattr(cfg.TRAINER.DYNAMIC, 'PREC', 'fp32')
        assert prec in ["fp16", "fp32", "amp"], f"Invalid precision: {prec}"
    
    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        prec = getattr(cfg.TRAINER.DYNAMIC, 'PREC', 'fp32')
        if prec == "fp32" or prec == "amp":
            clip_model.float()
        
        print("Building custom CLIP with dynamic prompts")
        self.model = build_custom_clip(
            cfg, classnames, clip_model, 
            model_type=getattr(cfg.TRAINER.DYNAMIC, 'MODEL_TYPE', 'dynamic')
        )
        
        # 冻结非prompt_learner的参数
        print("Turning off gradients in image and text encoders")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "semantic_enhancer" not in name:
                param.requires_grad_(False)
        
        # 加载初始化权重
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        # 移动到设备
        self.model.to(self.device)
        
        # 优化器 - 只优化prompt_learner和semantic_enhancer
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
                print(f"Optimizing: {name}")
        
        self.optim = build_optimizer(
            nn.ParameterList(params_to_optimize), 
            cfg.OPTIM
        )
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # 注册模型
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        if hasattr(self.model, 'semantic_enhancer') and self.model.semantic_enhancer is not None:
            self.register_model("semantic_enhancer", self.model.semantic_enhancer, self.optim, self.sched)
        
        # 混合精度训练
        self.scaler = GradScaler() if prec == "amp" else None
        
        # 多GPU
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        
        # 训练监控
        self.last_monitor_time = time.time()
        self.monitor_interval = getattr(cfg.TRAINER.DYNAMIC, 'MONITOR_INTERVAL', 60)  # 默认60秒
    
    def forward_backward(self, batch):
        """前向和反向传播"""
        image, label = self.parse_batch_train(batch)
        
        model = self.model
        prec = getattr(self.cfg.TRAINER.DYNAMIC, 'PREC', 'fp32')
        
        if prec == "amp":
            with autocast():
                output = model(image, label)
                # 检查输出是否是loss（CoCoOp风格）
                if output.dim() == 0:  # scalar loss
                    loss = output
                else:  # logits
                    # 获取难度权重（如果有）
                    weights = model.get_difficulty_weights()
                    if weights is not None:
                        loss = F.cross_entropy(output, label, weight=None, reduction='none')
                        # 加权损失
                        weighted_loss = (loss * weights).mean()
                        loss = weighted_loss
                    else:
                        loss = F.cross_entropy(output, label)
            
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            
        else:
            output = model(image, label)
            
            # 检查输出是否是loss（CoCoOp风格）
            if output.dim() == 0:  # scalar loss
                loss = output
            else:  # logits
                # 获取难度权重（如果有）
                weights = model.get_difficulty_weights()
                if weights is not None:
                    loss = F.cross_entropy(output, label, reduction='none')
                    # 加权损失
                    weighted_loss = (loss * weights).mean()
                    loss = weighted_loss
                else:
                    loss = F.cross_entropy(output, label)
            
            self.model_backward_and_update(loss)
        
        # 计算准确率（如果输出是logits）
        if output.dim() > 0:  # logits
            acc = compute_accuracy(output, label)[0].item()
        else:  # loss，需要重新计算logits
            with torch.no_grad():
                logits = model(image)
                acc = compute_accuracy(logits, label)[0].item()
        
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        
        # 更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        # 训练监控
        self._monitor_training(loss_summary)
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        """解析训练batch"""
        input_img = batch["img"]
        label = batch["label"]
        input_img = input_img.to(self.device)
        label = label.to(self.device)
        return input_img, label
    
    def _monitor_training(self, loss_summary):
        """
        训练监控 - 定期反馈训练进度
        用于长时间训练，每1-2分钟反馈一次
        """
        current_time = time.time()
        elapsed = current_time - self.last_monitor_time
        
        if elapsed >= self.monitor_interval:
            # 打印训练信息
            epoch_info = f"Epoch {self.epoch}"
            batch_info = f"Batch {self.batch_idx + 1}/{self.num_batches}"
            loss_info = f"Loss: {loss_summary['loss']:.4f}"
            acc_info = f"Acc: {loss_summary['acc']:.2f}%"
            
            print(f"\n[Monitor] {epoch_info} | {batch_info} | {loss_info} | {acc_info}")
            
            # 额外信息
            if hasattr(self.model, 'prompt_learner'):
                ctx = self.model.prompt_learner.ctx
                ctx_norm = ctx.norm().item()
                print(f"  Context norm: {ctx_norm:.4f}")
            
            # 重置监控计时器
            self.last_monitor_time = current_time
    
    def after_train(self):
        """训练完成后的处理"""
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best accuracy: {self.best_result:.2%}")
        print(f"Final epoch: {self.epoch}")
        print("="*50 + "\n")
        
        # 调用父类的after_train
        super().after_train()
    
    def load_model(self, directory, epoch=None):
        """加载模型"""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 忽略固定的token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # 设置strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
