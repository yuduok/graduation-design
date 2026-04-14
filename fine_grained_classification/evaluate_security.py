"""
安全评估脚本 - 测试对抗性防御效果
Security Evaluation Script - Testing Adversarial Defense Effectiveness

该脚本用于评估模型在对抗性攻击下的表现，以及防御机制的有效性。
"""
import argparse
import gc
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
DASSL_PATH = os.path.join(COOP_PATH, "dassl")

if os.path.exists(COOP_PATH) and COOP_PATH not in sys.path:
    sys.path.insert(0, COOP_PATH)
if os.path.exists(DASSL_PATH) and DASSL_PATH not in sys.path:
    sys.path.insert(0, DASSL_PATH)

sys.path.insert(0, CURRENT_DIR)

from clip import clip

from models.custom_clip import load_clip_to_cpu
from models.robust_custom_clip import build_robust_clip
from models.adversarial_defense import TestTimeCounterattack, AdversarialDetector


def clamp(X, lower_limit=0.0, upper_limit=1.0):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def generate_pgd_attack(model, images, labels, epsilon=4.0/255.0, alpha=1.0/255.0,
                        num_steps=10, norm="l_inf"):
    """
    生成PGD对抗性攻击

    Args:
        model: 目标模型
        images: 原始图像
        labels: 真实标签
        epsilon: 扰动上界
        alpha: 步长
        num_steps: 迭代次数
        norm: 范数类型 ("l_inf" 或 "l_2")
    """
    delta = torch.zeros_like(images).to(images.device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon

    delta = clamp(delta, -epsilon, epsilon)
    delta.requires_grad = True

    for _ in range(num_steps):
        output, _ = model(images + delta)
        loss = F.cross_entropy(output, labels)

        grad = torch.autograd.grad(loss, delta)[0]

        if norm == "l_inf":
            delta.data = clamp(delta.data + alpha * torch.sign(grad), -epsilon, epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = grad / (g_norm + 1e-10)
            delta.data = clamp(
                (delta.data + scaled_g * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(delta.data),
                -epsilon, epsilon
            )

    return clamp(delta, -epsilon, epsilon)


def generate_fgsm_attack(model, images, labels, epsilon=4.0/255.0):
    """
    生成FGSM对抗性攻击

    Fast Gradient Sign Method
    """
    delta = torch.zeros_like(images).to(images.device)
    delta.requires_grad = True

    output, _ = model(images + delta)
    loss = F.cross_entropy(output, labels)
    loss.backward()

    delta.data = clamp(delta.data + epsilon * torch.sign(delta.grad), -epsilon, epsilon)
    delta.grad.zero_()

    return clamp(delta, -epsilon, epsilon)


def evaluate_clean_accuracy(model, dataloader, device):
    """评估干净样本准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating clean accuracy"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            output, _ = model(images)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def evaluate_adversarial_accuracy(model, dataloader, device, attack_type="pgd",
                                   epsilon=4.0/255.0, alpha=1.0/255.0,
                                   num_steps=10):
    """评估对抗样本准确率"""
    model.eval()
    correct = 0
    correct_defended = 0
    total = 0

    print(f"\nGenerating adversarial examples with {attack_type} attack (epsilon={epsilon:.4f})")

    for batch in tqdm(dataloader, desc=f"Attacking and evaluating"):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        if attack_type == "pgd":
            delta = generate_pgd_attack(model, images, labels, epsilon, alpha, num_steps)
        elif attack_type == "fgsm":
            delta = generate_fgsm_attack(model, images, labels, epsilon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        with torch.no_grad():
            output_clean, _ = model(images)
            _, predicted_clean = output_clean.max(1)

            output_adv, defense_info = model(images + delta)
            _, predicted_adv = output_adv.max(1)

            total += labels.size(0)
            correct += predicted_clean.eq(labels).sum().item()
            correct_defended += predicted_adv.eq(labels).sum().item()

    clean_acc = 100.0 * correct / total
    adv_acc = 100.0 * correct_defended / total

    return clean_acc, adv_acc


def evaluate_defense_effectiveness(model, dataloader, device, epsilon=4.0/255.0):
    """
    评估防御机制的有效性

    比较:
    1. 无防御的干净准确率
    2. 无防御的对抗准确率
    3. 有防御的干净准确率
    4. 有防御的对抗准确率
    """
    print("\n" + "="*60)
    print("Phase 1: Clean accuracy (no defense)")
    print("="*60)
    model.set_defense_enabled(False)
    model.set_defense_mode("normal")
    clean_acc_no_defense = evaluate_clean_accuracy(model, dataloader, device)

    print("\n" + "="*60)
    print("Phase 2: Adversarial accuracy (no defense)")
    print("="*60)
    adv_acc_no_defense_clean_model, adv_acc_no_defense = evaluate_adversarial_accuracy(
        model, dataloader, device, attack_type="pgd", epsilon=epsilon
    )

    print("\n" + "="*60)
    print("Phase 3: Clean accuracy (with defense)")
    print("="*60)
    model.set_defense_enabled(True)
    model.set_defense_mode("defend")
    clean_acc_with_defense = evaluate_clean_accuracy(model, dataloader, device)

    print("\n" + "="*60)
    print("Phase 4: Adversarial accuracy (with defense)")
    print("="*60)
    model.set_defense_mode("defend")
    adv_acc_with_defense_clean_model, adv_acc_with_defense = evaluate_adversarial_accuracy(
        model, dataloader, device, attack_type="pgd", epsilon=epsilon
    )

    print("\n" + "="*60)
    print("DEFENSE EVALUATION RESULTS")
    print("="*60)
    print(f"Clean accuracy (no defense):        {clean_acc_no_defense:.2f}%")
    print(f"Clean accuracy (with defense):       {clean_acc_with_defense:.2f}%")
    print(f"Adversarial accuracy (no defense):  {adv_acc_no_defense:.2f}%")
    print(f"Adversarial accuracy (with defense): {adv_acc_with_defense:.2f}%")
    print("-"*60)
    print(f"Defense improvement on adversarial:  +{adv_acc_with_defense - adv_acc_no_defense:.2f}%")
    print(f"Clean accuracy drop due to defense:  {clean_acc_with_defense - clean_acc_no_defense:.2f}%")
    print("="*60)

    return {
        "clean_acc_no_defense": clean_acc_no_defense,
        "clean_acc_with_defense": clean_acc_with_defense,
        "adv_acc_no_defense": adv_acc_no_defense,
        "adv_acc_with_defense": adv_acc_with_defense,
        "defense_improvement": adv_acc_with_defense - adv_acc_no_defense,
        "clean_acc_drop": clean_acc_with_defense - clean_acc_no_defense
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Security evaluation for CLIP models")
    parser.add_argument("--root", type=str, default=CURRENT_DIR)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="oxford_pets",
                        choices=["oxford_pets", "oxford_flowers", "stanford_cars",
                                "food101", "caltech101", "sun397"])
    parser.add_argument("--trainer", type=str, default="DynamicPromptTrainer",
                        choices=["CoOp", "CoCoOp", "DynamicPromptTrainer"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--epsilon", type=float, default=4.0/255.0,
                        help="Attack epsilon (perturbation budget)")
    parser.add_argument("--alpha", type=float, default=1.0/255.0,
                        help="PGD step size")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of PGD steps")
    parser.add_argument("--attack-type", type=str, default="pgd",
                        choices=["pgd", "fgsm"],
                        help="Type of adversarial attack")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--defense-eps", type=float, default=4.0/255.0,
                        help="Defense epsilon")
    parser.add_argument("--defense-steps", type=int, default=2,
                        help="Defense number of steps")
    parser.add_argument("--output-dir", type=str, default="security_results")

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    dataset_name_map = {
        "oxford_pets": "OxfordPets",
        "oxford_flowers": "OxfordFlowers",
        "stanford_cars": "StanfordCars",
        "food101": "Food101",
        "caltech101": "Caltech101",
        "sun397": "SUN397"
    }
    dataset_internal_name = dataset_name_map.get(args.dataset, args.dataset)

    print("="*60)
    print("SECURITY EVALUATION")
    print("="*60)
    print(f"Dataset: {dataset_internal_name}")
    print(f"Device: {device}")
    print(f"Attack epsilon: {args.epsilon:.4f} ({args.epsilon * 255:.2f}/255)")
    print(f"Defense epsilon: {args.defense_eps:.4f} ({args.defense_eps * 255:.2f}/255)")
    print("="*60)

    print("\nLoading CLIP model...")
    clip_model, _ = clip.load("RN50", device, jit=False)
    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False

    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.TRAINER = CN()
    cfg.TRAINER.DYNAMIC = CN()
    cfg.TRAINER.DYNAMIC.N_CTX = 16
    cfg.TRAINER.DYNAMIC.CTX_INIT = "a photo of a"
    cfg.TRAINER.DYNAMIC.USE_SEMANTIC_ENHANCEMENT = False
    cfg.TRAINER.DYNAMIC.USE_DYNAMIC = True
    cfg.TRAINER.DYNAMIC.USE_ADAPTIVE = True
    cfg.TRAINER.DYNAMIC.USE_DIFFICULTY_WEIGHT = False
    cfg.TRAINER.DYNAMIC.ALPHA = 0.1
    cfg.TRAINER.DYNAMIC.BETA = 0.01

    defense_config = {
        "eps": args.defense_eps,
        "num_steps": args.defense_steps,
        "step_size": args.defense_eps / args.defense_steps,
        "tau_threshold": 0.2,
        "beta": 2.0
    }

    if args.model_path and os.path.exists(args.model_path):
        print(f"\nLoading model from {args.model_path}")
        from dassl.utils import load_checkpoint
        checkpoint = load_checkpoint(args.model_path)

        sys.path.insert(0, os.path.join(COOP_PATH, "datasets"))
        if args.dataset == "oxford_pets":
            import datasets.oxford_pets as oxford_pets_module
            classnames = oxford_pets_module.OxfordPets().classnames
        elif args.dataset == "oxford_flowers":
            import datasets.oxford_flowers as oxford_flowers_module
            classnames = oxford_flowers_module.OxfordFlowers().classnames
        else:
            classnames = [f"class_{i}" for i in range(37)]

        model = build_robust_clip(cfg, classnames, clip_model, "dynamic", defense_config)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.to(device)
    else:
        print("\nNo trained model found. Using zero-shot CLIP with defense for demonstration.")
        sys.path.insert(0, os.path.join(COOP_PATH, "datasets"))
        if args.dataset == "oxford_pets":
            import datasets.oxford_pets as oxford_pets_module
            classnames = oxford_pets_module.OxfordPets().classnames
        else:
            classnames = [f"class_{i}" for i in range(37)]

        model = build_robust_clip(cfg, classnames, clip_model, "dynamic", defense_config)
        model = model.to(device)

    from dassl.data import build_dataset
    from torch.utils.data import DataLoader

    DATA_PATH = os.path.join(PROJECT_ROOT, "data")

    dataset = build_dataset(dataset_internal_name, DATASET_ROOT=DATA_PATH)
    dataloader = DataLoader(
        dataset.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    print(f"\nTest set size: {len(dataset.test)}")

    results = evaluate_defense_effectiveness(model, dataloader, device, epsilon=args.epsilon)

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"security_evaluation_{args.dataset}.txt")

    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("SECURITY EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Attack type: {args.attack_type}\n")
        f.write(f"Attack epsilon: {args.epsilon:.4f}\n")
        f.write(f"Defense epsilon: {args.defense_eps:.4f}\n")
        f.write(f"Defense steps: {args.defense_steps}\n\n")
        f.write(f"Clean accuracy (no defense):        {results['clean_acc_no_defense']:.2f}%\n")
        f.write(f"Clean accuracy (with defense):       {results['clean_acc_with_defense']:.2f}%\n")
        f.write(f"Adversarial accuracy (no defense):  {results['adv_acc_no_defense']:.2f}%\n")
        f.write(f"Adversarial accuracy (with defense): {results['adv_acc_with_defense']:.2f}%\n\n")
        f.write(f"Defense improvement on adversarial:  +{results['defense_improvement']:.2f}%\n")
        f.write(f"Clean accuracy drop due to defense:  {results['clean_acc_drop']:.2f}%\n")

    print(f"\nResults saved to {results_file}")

    return results


if __name__ == "__main__":
    main()
