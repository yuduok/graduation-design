"""
Modal 安全评估脚本 - 精简版
只测试对抗样本，支持减少测试集大小
"""
import modal
import os

app = modal.App("adversarial-defense-eval-full")

volume = modal.Volume.from_name("graduation-design-full", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install([
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "ftfy",
        "regex",
        "pyyaml",
        "Pillow",
        "yacs",
        "tabulate",
        "scipy",
        "scikit-learn",
        "gdown",
        "future",
        "wilds",
        "tensorboard",
    ])
    .pip_install("git+https://github.com/openai/CLIP.git")
)

REMOTE_ROOT = "/mnt/data/root/graduation-design"


@app.function(
    image=image,
    volumes={"/mnt/data": volume},
    gpu="A100",
    timeout=7200,
)
def evaluate_security(
    dataset: str = "oxford_pets",
    shots: int = 1,
    seed: int = 1,
    epsilon: float = 4.0 / 255.0,
    max_samples: int = 100,
    test_defense: bool = False,
):
    import sys
    import gc
    import random
    import tarfile
    import shutil
    import subprocess
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm

    subprocess.run([sys.executable, "-m", "pip", "install", "wilds", "tensorboard", "omegaconf", "pandas"], check=True)

    PROJECT_ROOT = REMOTE_ROOT
    COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
    CURRENT_DIR = os.path.join(PROJECT_ROOT, "fine_grained_classification")
    INNER_DASSL_PATH = os.path.join(COOP_PATH, "dassl", "dassl")
    SITE_PACKAGES_DASSL = "/usr/local/lib/python3.13/site-packages/dassl"

    if not os.path.exists(SITE_PACKAGES_DASSL):
        shutil.copytree(INNER_DASSL_PATH, SITE_PACKAGES_DASSL)
        print(f"Copied dassl to {SITE_PACKAGES_DASSL}")

    for key in list(sys.modules.keys()):
        if key.startswith('dassl'):
            del sys.modules[key]

    sys.path.insert(0, "/usr/local/lib/python3.13/site-packages")
    sys.path.insert(1, COOP_PATH)
    sys.path.insert(2, CURRENT_DIR)

    print("=" * 60)
    print("ADVERSARIAL EVALUATION (Minimal)")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Shots: {shots}")
    print(f"Seed: {seed}")
    print(f"Attack epsilon: {epsilon:.4f}")
    print(f"Max test samples: {max_samples}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    print("\nExtracting dataset if needed...")
    dataset_dir = os.path.join(PROJECT_ROOT, "data", dataset)
    images_tar = os.path.join(dataset_dir, "images.tar.gz")
    annotations_tar = os.path.join(dataset_dir, "annotations.tar.gz")

    if not os.path.exists(os.path.join(dataset_dir, "images")) and os.path.exists(images_tar):
        print(f"Extracting {images_tar}...")
        with tarfile.open(images_tar, "r:gz") as tar:
            tar.extractall(dataset_dir)

    if not os.path.exists(os.path.join(dataset_dir, "annotations")) and os.path.exists(annotations_tar):
        print(f"Extracting {annotations_tar}...")
        with tarfile.open(annotations_tar, "r:gz") as tar:
            tar.extractall(dataset_dir)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda"
    print(f"\nUsing device: {device}")

    dataset_name_map = {
        "oxford_pets": "OxfordPets",
        "oxford_flowers": "OxfordFlowers",
        "stanford_cars": "StanfordCars",
        "food101": "Food101",
        "caltech101": "Caltech101",
        "sun397": "SUN397",
    }
    dataset_internal_name = dataset_name_map.get(dataset, dataset)

    print("\nLoading CLIP model...")
    import sys

    for key in list(sys.modules.keys()):
        if key == 'clip' or key.startswith('clip.') or key.startswith('dassl'):
            del sys.modules[key]

    sys.path.insert(0, COOP_PATH)
    sys.path.insert(1, "/usr/local/lib/python3.13/site-packages")

    from dassl.config import get_cfg_default
    import clip
    clip_model, _ = clip.load("RN50", device, jit=False)

    clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False
    from yacs.config import CfgNode as CN

    cfg = get_cfg_default()
    cfg.defrost()
    cfg.VERBOSE = False
    cfg.SEED = seed
    cfg.TRAINER.DYNAMIC = CN()
    cfg.TRAINER.DYNAMIC.N_CTX = 16
    cfg.TRAINER.DYNAMIC.CTX_INIT = "a photo of a"
    cfg.TRAINER.DYNAMIC.USE_SEMANTIC_ENHANCEMENT = False
    cfg.TRAINER.DYNAMIC.USE_DYNAMIC = True
    cfg.TRAINER.DYNAMIC.USE_ADAPTIVE = True
    cfg.TRAINER.DYNAMIC.USE_DIFFICULTY_WEIGHT = False
    cfg.TRAINER.DYNAMIC.ALPHA = 0.1
    cfg.TRAINER.DYNAMIC.BETA = 0.01
    cfg.DATASET.ROOT = os.path.join(PROJECT_ROOT, "data")
    cfg.DATASET.NAME = dataset_internal_name
    cfg.DATASET.NUM_SHOTS = shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.SPLIT = 0
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.freeze()

    defense_config = {
        "eps": 4.0/255.0,
        "num_steps": 2,
        "step_size": 2.0/255.0,
        "tau_threshold": 0.2,
        "beta": 2.0,
    }

    model_path = os.path.join(
        PROJECT_ROOT, "fine_grained_classification", "output_fgd", dataset,
        "DynamicPromptTrainer", f"shots_{shots}", f"seed_{seed}",
        "prompt_learner", "model.pth.tar-100"
    )

    if dataset == "oxford_pets":
        classnames = [
            'Abyssinian', 'american_bulldog', 'basset_hound', 'beagle', 'Bengal',
            'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua',
            'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'German_Shorthaired_Pointer',
            'Great_Dane', 'Havanese', 'japanese_chin', 'Keeshond', 'Leonberger',
            'Maine_Coon', 'Miniature_Pinscher', 'newfoundland', 'Persian', 'Pomeranian',
            'pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard', 'Samoyed',
            'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
            'wheaten_terrier', 'Yorkshire_Terrier'
        ]
    else:
        classnames = [f"class_{i}" for i in range(37)]

    print(f"\nLooking for model at: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print("ERROR: Model not found!")
        return {"error": "Model not found"}

    from models.robust_custom_clip import build_robust_clip

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_robust_clip(cfg, classnames, clip_model, "dynamic", defense_config)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)

    import datasets.oxford_pets

    from dassl.data.data_manager import DataManager
    datamanager = DataManager(cfg)
    dataloader = datamanager.test_loader

    print(f"\nTest set size: {len(datamanager.dataset.test)}")
    print(f"Using max samples: {max_samples}")

    def clamp(X, lower=0.0, upper=1.0):
        lower_t = torch.tensor(lower, device=X.device, dtype=X.dtype)
        upper_t = torch.tensor(upper, device=X.device, dtype=X.dtype)
        return torch.max(torch.min(X, upper_t), lower_t)

    def generate_fgsm_attack(model, images, labels, epsilon):
        delta = torch.zeros_like(images).to(device)
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, -epsilon, epsilon)
        delta.requires_grad = True

        output, _ = model(images + delta)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        delta.data = clamp(delta.data + epsilon * torch.sign(delta.grad), -epsilon, epsilon)
        delta.grad.zero_()
        return clamp(delta, -epsilon, epsilon)

    def generate_perturbations(model, dataloader, device, epsilon, max_samples=None):
        model.eval()
        perturbations = []
        all_labels = []

        sample_count = 0
        for batch in tqdm(dataloader, desc="Generating perturbations"):
            if max_samples and sample_count >= max_samples:
                break

            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            model.set_defense_enabled(False)
            model.set_defense_mode("normal")
            delta = generate_fgsm_attack(model, images, labels, epsilon)

            perturbations.append(delta.cpu())
            all_labels.append(labels.cpu())

            sample_count += images.size(0)
            del images, labels, delta
            torch.cuda.empty_cache()

        return perturbations, all_labels

    def evaluate_with_perturbations(model, dataloader, device, perturbations, labels_list, use_defense, desc="Evaluating"):
        model.eval()
        correct = 0
        total = 0

        sample_idx = 0
        for batch, delta_batch, labels_batch in tqdm(zip(dataloader, perturbations, labels_list), desc=desc):
            batch_size = labels_batch.size(0)
            indices = list(range(sample_idx, sample_idx + batch_size))

            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            delta = delta_batch.to(device)

            if use_defense:
                model.set_defense_enabled(True)
                model.set_defense_mode("defend")
                with torch.no_grad():
                    try:
                        defended_images, defense_info = model(images + delta)
                        if defense_info is not None:
                            is_adv = defense_info.get("is_adversarial", torch.tensor([False]).to(device))
                            tau = defense_info.get("tau", torch.tensor([0.0]).to(device))
                            print(f"\nDebug: {is_adv.sum().item()}/{is_adv.size(0)} adversarial, tau mean={tau.mean().item():.4f}")
                        output, _ = model(defended_images)
                    except Exception as e:
                        print(f"\nDefense error: {e}")
                        raise
            else:
                model.set_defense_enabled(False)
                model.set_defense_mode("normal")

            with torch.no_grad():
                if not use_defense:
                    output, _ = model(images + delta)
                _, predicted = output.max(1)
                correct += predicted.eq(labels).sum().item()

            total += labels.size(0)
            sample_idx += batch_size
            del images, labels, delta, output
            torch.cuda.empty_cache()

        return 100.0 * correct / total

    print("\n" + "=" * 60)
    print("Phase 1: Adversarial accuracy (NO defense)")
    print("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()

    perturbations, labels_list = generate_perturbations(model, dataloader, device, epsilon, max_samples)
    adv_acc_no_def = evaluate_with_perturbations(model, dataloader, device, perturbations, labels_list, use_defense=False, desc="No defense")

    print("\n" + "=" * 60)
    print("Phase 2: Adversarial accuracy (WITH defense)")
    print("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()

    adv_acc_with_def = evaluate_with_perturbations(model, dataloader, device, perturbations, labels_list, use_defense=True, desc="With defense")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Adversarial accuracy (no defense):   {adv_acc_no_def:.2f}%")
    print(f"Adversarial accuracy (with defense): {adv_acc_with_def:.2f}%")
    print("-" * 60)
    print(f"Defense improvement:                 +{adv_acc_with_def - adv_acc_no_def:.2f}%")
    print("=" * 60)

    return {
        "adv_acc_no_defense": adv_acc_no_def,
        "adv_acc_with_defense": adv_acc_with_def,
        "defense_improvement": adv_acc_with_def - adv_acc_no_def,
    }


@app.local_entrypoint()
def main(
    dataset: str = "oxford_pets",
    shots: int = 1,
    seed: int = 1,
    max_samples: int = 100,
):
    print(f"Launching adversarial evaluation for {dataset}...")
    print(f"Using max {max_samples} test samples...")
    results = evaluate_security.remote(
        dataset=dataset,
        shots=shots,
        seed=seed,
        max_samples=max_samples,
    )
    print("\n[Complete]")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
