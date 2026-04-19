"""
Bridge evaluator that reuses the original TTC counterattack implementation
from CLIP-Test-time-Counterattacks on the fine-grained OxfordPets model.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
COOP_PATH = PROJECT_ROOT / "CoOp"
DASSL_PATH = COOP_PATH / "dassl"
TTC_ROOT = PROJECT_ROOT / "CLIP-Test-time-Counterattacks"
TTC_CODE_PATH = TTC_ROOT / "code"

for path in [str(CURRENT_DIR), str(COOP_PATH), str(DASSL_PATH), str(TTC_CODE_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import datasets.oxford_pets  # noqa: F401
from dassl.config import get_cfg_default
from dassl.data import DataManager
from yacs.config import CfgNode as CN

from models.custom_clip import TextEncoder, load_clip_to_cpu
from models.dynamic_prompt import AdaptivePromptLearner

from func import clip_img_preprocessing


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

lower_limit, upper_limit = 0, 1


def compute_tau(clip_visual, images, noise):
    orig_feat = clip_visual(clip_img_preprocessing(images), None)
    noisy_feat = clip_visual(clip_img_preprocessing(images + noise), None)
    diff_ratio = (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)
    return diff_ratio


def tau_thres_weighted_counterattacks(
    model,
    images,
    prompter,
    add_prompter,
    alpha,
    attack_iters,
    norm="l_inf",
    epsilon=0,
    visual_model_orig=None,
    tau_thres=None,
    beta=None,
    clip_visual=None,
):
    del visual_model_orig

    delta = torch.zeros_like(images)
    if epsilon <= 0:
        return delta

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    delta = torch.max(torch.min(delta, upper_limit - images), lower_limit - images)
    delta.requires_grad = True

    if attack_iters == 0:
        return delta.data

    diff_ratio = compute_tau(clip_visual, images, delta.data) if clip_visual is not None else None

    tunable_param_names = []
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            tunable_param_names.append(name)
            param.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        original_reps = model.module.encode_image(prompter(clip_img_preprocessing(images)), prompt_token)
        original_norm = torch.norm(original_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]

    for step_id in range(attack_iters):
        prompted_images = prompter(clip_img_preprocessing(images + delta))
        attacked_reps = model.module.encode_image(prompted_images, prompt_token)
        if step_id == 0 and diff_ratio is None:
            feature_diff = attacked_reps - original_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / original_norm

        scheme_sign = (tau_thres - diff_ratio).sign()
        l2_loss = (((attacked_reps - original_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]

        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = images[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = torch.max(torch.min(d, upper_limit - x), lower_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    stacked = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1, device=images.device).unsqueeze(0).expand(images.size(0), -1)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)

    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.0
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(images.size(0), attack_iters + 1, 1, 1, 1)

    final_delta = (weights * stacked).sum(dim=1)

    for name, param in model.module.named_parameters():
        if name in tunable_param_names:
            param.requires_grad = True

    return final_delta


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DynamicPromptTrainer with original TTC defense")
    parser.add_argument("--dataset", type=str, default="oxford_pets", choices=["oxford_pets"])
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/dynamic_rn50.yaml")
    parser.add_argument("--epsilon", type=float, default=1.0 / 255.0, help="PGD attack epsilon")
    parser.add_argument("--alpha", type=float, default=1.0 / 255.0, help="PGD step size")
    parser.add_argument("--num-steps", type=int, default=10, help="PGD steps")
    parser.add_argument("--ttc-eps", type=float, default=4.0 / 255.0, help="TTC epsilon")
    parser.add_argument("--ttc-stepsize", type=float, default=1.0 / 255.0, help="TTC step size")
    parser.add_argument("--ttc-numsteps", type=int, default=2, help="TTC steps")
    parser.add_argument("--tau-thres", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=str, default="security_results/ttc_dynamic_prompt_oxford_pets.json")
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg_default()
    cfg.defrost()
    cfg.TRAINER.DYNAMIC = CN()
    cfg.TRAINER.DYNAMIC.CTX_INIT = "a photo of a"
    cfg.TRAINER.DYNAMIC.N_CTX = 16
    cfg.TRAINER.DYNAMIC.PREC = "fp16"
    cfg.TRAINER.DYNAMIC.MODEL_TYPE = "dynamic"
    cfg.TRAINER.DYNAMIC.USE_DYNAMIC = True
    cfg.TRAINER.DYNAMIC.USE_ADAPTIVE = True
    cfg.TRAINER.DYNAMIC.USE_DIFFICULTY_WEIGHT = True
    cfg.TRAINER.DYNAMIC.ALPHA = 0.05
    cfg.TRAINER.DYNAMIC.BETA = 0.005
    cfg.TRAINER.DYNAMIC.ADAPTIVE_HIDDEN_DIM = None
    cfg.TRAINER.DYNAMIC.USE_SEMANTIC_ENHANCEMENT = False
    cfg.TRAINER.DYNAMIC.USE_ATTRIBUTE_DATABASE = True
    cfg.TRAINER.DYNAMIC.MONITOR_INTERVAL = 60
    cfg.TRAINER.DYNAMIC.LR = 0.002
    cfg.merge_from_file(str(CURRENT_DIR / args.config))
    cfg.DATASET.ROOT = str(PROJECT_ROOT / "data")
    cfg.DATASET.NAME = "OxfordPets"
    cfg.DATASET.NUM_SHOTS = args.shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.SPLIT = 1
    cfg.SEED = args.seed
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.TRAINER.DYNAMIC.PREC = "fp32"
    cfg.freeze()
    return cfg


def create_prompt_cfg():
    cfg = CN()
    cfg.TRAINER = CN()
    cfg.TRAINER.DYNAMIC = CN()
    cfg.TRAINER.DYNAMIC.CTX_INIT = "a photo of a"
    cfg.TRAINER.DYNAMIC.N_CTX = 16
    cfg.TRAINER.DYNAMIC.PREC = "fp32"
    cfg.TRAINER.DYNAMIC.MODEL_TYPE = "dynamic"
    cfg.TRAINER.DYNAMIC.USE_DYNAMIC = True
    cfg.TRAINER.DYNAMIC.USE_ADAPTIVE = True
    cfg.TRAINER.DYNAMIC.USE_DIFFICULTY_WEIGHT = True
    cfg.TRAINER.DYNAMIC.ALPHA = 0.05
    cfg.TRAINER.DYNAMIC.BETA = 0.005
    cfg.TRAINER.DYNAMIC.USE_SEMANTIC_ENHANCEMENT = False
    cfg.TRAINER.DYNAMIC.USE_ATTRIBUTE_DATABASE = True
    return cfg


def reverse_clip_normalize(images):
    mean = CLIP_MEAN.to(images.device, dtype=images.dtype)
    std = CLIP_STD.to(images.device, dtype=images.dtype)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def clamp(images):
    return torch.clamp(images, 0.0, 1.0)


def resolve_model_path(args):
    if args.model_path:
        return Path(args.model_path)

    model_dir = (
        CURRENT_DIR
        / "output_fgd"
        / args.dataset
        / "DynamicPromptTrainer"
        / f"shots_{args.shots}"
        / f"seed_{args.seed}"
        / "prompt_learner"
    )

    checkpoint_hint = model_dir / "checkpoint"
    if checkpoint_hint.exists():
        model_name = checkpoint_hint.read_text().strip()
        candidate = model_dir / model_name
        if candidate.exists():
            return candidate

    best_candidate = model_dir / "model-best.pth.tar"
    if best_candidate.exists():
        return best_candidate

    numbered = sorted(model_dir.glob("model.pth.tar-*"))
    if numbered:
        return numbered[-1]

    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def load_checkpoint_compat(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class DynamicPromptVictimCore(nn.Module):
    def __init__(self, clip_model, classnames):
        super().__init__()
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.prompt_learner = AdaptivePromptLearner(create_prompt_cfg(), classnames, clip_model)

    def load_prompt_weights(self, state_dict):
        state_dict = dict(state_dict)
        state_dict.pop("token_prefix", None)
        state_dict.pop("token_suffix", None)
        self.prompt_learner.load_state_dict(state_dict, strict=False)

    def encode_image(self, images, prompt_token=None):
        del prompt_token
        image_features = self.image_encoder(images.type(self.dtype))
        return image_features

    def forward(self, raw_images):
        images = clip_img_preprocessing(raw_images)
        image_features = self.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        batch_size, n_cls, n_tkn, _ = prompts.shape
        prompts = prompts.view(-1, n_tkn, prompts.size(-1))

        tokenized = self.prompt_learner.tokenized_prompts.to(raw_images.device)
        tokenized = tokenized.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * n_cls, -1)
        text_features = self.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(batch_size, n_cls, -1)

        logits = self.logit_scale.exp() * (
            image_features.unsqueeze(1) @ text_features.transpose(1, 2)
        ).squeeze(1)
        return logits


class ModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, raw_images):
        return self.module(raw_images)


def generate_pgd_attack(model, images, labels, epsilon, alpha, num_steps):
    delta = torch.zeros_like(images)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta)
    delta = torch.clamp(delta - images, -epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(num_steps):
        logits = model(clamp(images + delta))
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, delta)[0]
        delta.data = torch.clamp(delta.data + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.clamp(images + delta.data, 0.0, 1.0) - images

    return delta.detach()


def evaluate(model, dataloader, device, args):
    identity = lambda x: x
    null_prompt = lambda: None

    clean_correct = 0
    clean_ttc_correct = 0
    adv_correct = 0
    adv_ttc_correct = 0
    total = 0
    clean_tau = []
    adv_tau = []

    model.eval()
    model.module.prompt_learner.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="TTC eval")):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        labels = batch["label"].to(device)
        raw_images = reverse_clip_normalize(batch["img"].to(device))

        with torch.no_grad():
            clean_logits = model(raw_images)
            clean_correct += (clean_logits.argmax(dim=1) == labels).sum().item()

        ttc_delta_clean = tau_thres_weighted_counterattacks(
            model,
            raw_images,
            identity,
            null_prompt,
            alpha=args.ttc_stepsize,
            attack_iters=args.ttc_numsteps,
            norm="l_inf",
            epsilon=args.ttc_eps,
            visual_model_orig=None,
            tau_thres=args.tau_thres,
            beta=args.beta,
            clip_visual=model.module.encode_image,
        )
        ttc_delta_clean = ttc_delta_clean.detach()
        clean_tau_batch = compute_tau(model.module.encode_image, raw_images, ttc_delta_clean).detach()
        clean_tau.extend(clean_tau_batch.cpu().tolist())

        with torch.no_grad():
            clean_ttc_logits = model(clamp(raw_images + ttc_delta_clean))
            clean_ttc_correct += (clean_ttc_logits.argmax(dim=1) == labels).sum().item()

        adv_delta = generate_pgd_attack(
            model=model,
            images=raw_images,
            labels=labels,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.num_steps,
        )
        adv_images = clamp(raw_images + adv_delta)

        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_correct += (adv_logits.argmax(dim=1) == labels).sum().item()

        ttc_delta_adv = tau_thres_weighted_counterattacks(
            model,
            adv_images,
            identity,
            null_prompt,
            alpha=args.ttc_stepsize,
            attack_iters=args.ttc_numsteps,
            norm="l_inf",
            epsilon=args.ttc_eps,
            visual_model_orig=None,
            tau_thres=args.tau_thres,
            beta=args.beta,
            clip_visual=model.module.encode_image,
        )
        ttc_delta_adv = ttc_delta_adv.detach()
        adv_tau_batch = compute_tau(model.module.encode_image, adv_images, ttc_delta_adv).detach()
        adv_tau.extend(adv_tau_batch.cpu().tolist())

        with torch.no_grad():
            adv_ttc_logits = model(clamp(adv_images + ttc_delta_adv))
            adv_ttc_correct += (adv_ttc_logits.argmax(dim=1) == labels).sum().item()

        total += labels.size(0)

    return {
        "clean_acc": 100.0 * clean_correct / total,
        "clean_ttc_acc": 100.0 * clean_ttc_correct / total,
        "adv_acc": 100.0 * adv_correct / total,
        "adv_ttc_acc": 100.0 * adv_ttc_correct / total,
        "adv_ttc_gain": 100.0 * (adv_ttc_correct - adv_correct) / total,
        "clean_ttc_delta": 100.0 * (clean_ttc_correct - clean_correct) / total,
        "mean_clean_tau": float(np.mean(clean_tau)) if clean_tau else None,
        "mean_adv_tau": float(np.mean(adv_tau)) if adv_tau else None,
        "num_samples": total,
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    cfg = build_cfg(args)
    dm = DataManager(cfg)
    classnames = dm.dataset.classnames
    dataloader = dm.test_loader

    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    # Build the prompt learner on CPU first. AdaptivePromptLearner creates
    # token tensors on CPU during initialization, so moving CLIP to CUDA
    # beforehand triggers a device mismatch in token_embedding().
    victim = DynamicPromptVictimCore(clip_model, classnames)
    checkpoint_path = resolve_model_path(args)
    checkpoint = load_checkpoint_compat(str(checkpoint_path))
    victim.load_prompt_weights(checkpoint["state_dict"])
    victim.to(device)
    victim.eval()

    model = ModuleWrapper(victim).to(device)
    model.eval()

    results = evaluate(model, dataloader, device, args)
    results.update(
        {
            "dataset": args.dataset,
            "shots": args.shots,
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "attack": {
                "type": "pgd",
                "epsilon": args.epsilon,
                "alpha": args.alpha,
                "num_steps": args.num_steps,
            },
            "ttc": {
                "epsilon": args.ttc_eps,
                "step_size": args.ttc_stepsize,
                "num_steps": args.ttc_numsteps,
                "tau_threshold": args.tau_thres,
                "beta": args.beta,
            },
        }
    )

    output_path = CURRENT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("TTC SECURITY EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {results['num_samples']}")
    print(f"Clean acc:     {results['clean_acc']:.2f}%")
    print(f"Clean + TTC:   {results['clean_ttc_acc']:.2f}%")
    print(f"Adv acc:       {results['adv_acc']:.2f}%")
    print(f"Adv + TTC:     {results['adv_ttc_acc']:.2f}%")
    print(f"Adv TTC gain:  {results['adv_ttc_gain']:.2f}%")
    print(f"Output:        {output_path}")


if __name__ == "__main__":
    main()
