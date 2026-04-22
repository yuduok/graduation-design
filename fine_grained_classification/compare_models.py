"""
细粒度分类对比评估脚本
对比 CLIP、CoOp、CoCoOp 和 DynamicPrompt 在细粒度分类任务上的表现
"""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, COOP_PATH)

import clip
from clip import clip


OXFORD_PETS_CLASSNAMES = [
    'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle',
    'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua',
    'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
    'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian',
    'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]


class OxfordPetsDataset(Dataset):
    def __init__(self, data_root, split="test"):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "oxford_pets", "images")
        self.split_file = os.path.join(data_root, "oxford_pets", "split_zhou_OxfordPets.json")

        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                splits = json.load(f)
            raw_data = splits.get(split, [])
            self.data = []
            for item in raw_data:
                if isinstance(item, list) and len(item) >= 2:
                    self.data.append({'image': item[0], 'label': int(item[1])})
                elif isinstance(item, dict):
                    self.data.append(item)
        else:
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'img': image, 'label': item['label'], 'image_name': item['image']}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare fine-grained classification models")
    parser.add_argument("--dataset", type=str, default="oxford_pets",
                       choices=["oxford_pets"],
                       help="dataset name")
    parser.add_argument("--data-root", type=str,
                       default="/home/zengyule/graduation-design/data",
                       help="path to dataset root")
    parser.add_argument("--backbone", type=str, default="RN50",
                       choices=["RN50", "RN101", "ViT-B/16", "ViT-B/32"],
                       help="CLIP backbone")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="number of data workers")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="device to use")
    parser.add_argument("--models", nargs="+",
                       default=["clip", "coop", "cocoop", "dynamic"],
                       help="models to compare")
    parser.add_argument("--model-dirs", type=str, default=None,
                       help="JSON string mapping model names to checkpoint directories")
    return parser.parse_args()


class ZeroShotCLIP:
    def __init__(self, backbone, classnames, device):
        self.device = device
        self.classnames = classnames
        print(f"Loading Zero-shot CLIP ({backbone})...")
        self.clip_model, _ = clip.load(backbone, device=device)
        self.clip_model.eval()

        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
        self.tokens = clip.tokenize(prompts).to(device)

    @torch.no_grad()
    def evaluate(self, dataloader):
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="Zero-shot CLIP"):
            images = batch['img'].to(self.device)
            labels = batch['label']

            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.clip_model.encode_text(self.tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = self.clip_model.logit_scale.exp() * image_features @ text_features.T
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels.to(self.device)).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs
        }


def load_clip_model(backbone, device):
    print(f"Loading CLIP ({backbone})...")
    clip_model, _ = clip.load(backbone, device=device)
    clip_model.eval()
    clip_model.float()
    return clip_model


class CoOpClassifier(nn.Module):
    def __init__(self, clip_model, classnames, ctx, tokenized_prompts, token_prefix, token_suffix):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.ctx = ctx
        self.n_ctx = ctx.shape[0]
        self.tokenized_prompts = tokenized_prompts
        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        self.dtype = clip_model.dtype

    def forward(self, images):
        image_features = self.clip_model.encode_image(images.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = []
        for i in range(len(self.classnames)):
            prefix = self.token_prefix[i:i+1]
            ctx_i = self.ctx.unsqueeze(0)
            suffix = self.token_suffix[i:i+1]
            prompt = torch.cat([prefix, ctx_i, suffix], dim=1)
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        x = prompts + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection

        text_features = x / x.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CoOpEvaluator:
    def __init__(self, backbone, checkpoint_dir, classnames, device):
        self.device = device
        self.clip_model = load_clip_model(backbone, device)

        print(f"Loading CoOp model from {checkpoint_dir}...")
        checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model-best.pth.tar")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar-20")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: CoOp checkpoint not found at {checkpoint_path}")
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Loaded CoOp checkpoint from epoch {epoch}")

        n_ctx = state_dict['ctx'].shape[0]
        ctx = state_dict['ctx'].to(device)

        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx:, :]

        self.model = CoOpClassifier(self.clip_model, classnames, ctx, tokenized_prompts, token_prefix, token_suffix)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader):
        if self.model is None:
            return {"accuracy": 0, "predictions": [], "labels": [], "probabilities": torch.zeros(1)}

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="CoOp"):
            images = batch['img'].to(self.device)
            labels = batch['label']

            logits = self.model(images)
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels.to(self.device)).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs
        }


class SoftPromptAdapter(nn.Module):
    def __init__(self, vis_dim=512, ctx_dim=512):
        super().__init__()
        hidden_dim = vis_dim // 16
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, hidden_dim)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_dim, ctx_dim))
        ]))

    def forward(self, image_features, base_ctx, n_cls):
        batch_size = image_features.shape[0]
        bias = self.meta_net(image_features).unsqueeze(1)
        ctx = base_ctx.unsqueeze(0).expand(batch_size, -1, -1)
        ctx_shifted = ctx + bias
        ctx_shifted = ctx_shifted.unsqueeze(1).expand(-1, n_cls, -1, -1)
        return ctx_shifted


class DynamicPromptClassifier(nn.Module):
    def __init__(self, clip_model, classnames, ctx, class_adaptive_factors, meta_net_state_dict, tokenized_prompts, token_prefix, token_suffix):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.ctx = nn.Parameter(ctx)
        self.class_adaptive_factors = nn.Parameter(class_adaptive_factors.squeeze(0))
        self.n_ctx = ctx.shape[0]
        self.tokenized_prompts = tokenized_prompts
        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        self.dtype = clip_model.dtype

        vis_dim = clip_model.visual.output_dim
        ctx_dim = ctx.shape[-1]
        
        adapter = SoftPromptAdapter(vis_dim=vis_dim, ctx_dim=ctx_dim)
        if meta_net_state_dict is not None:
            adapter.meta_net.load_state_dict(meta_net_state_dict)
        self.soft_prompt_adapter = adapter

    def forward(self, images):
        image_features = self.clip_model.encode_image(images.type(self.dtype))
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        base_ctx = self.ctx * self.class_adaptive_factors
        ctx_shifted = self.soft_prompt_adapter(image_features_norm, base_ctx, len(self.classnames))

        batch_size = images.shape[0]
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)

        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=2)

        logits_list = []
        logit_scale = self.clip_model.logit_scale.exp()
        for i in range(batch_size):
            prompt_i = prompts[i]
            x = prompt_i + self.clip_model.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
            text_features = x / x.norm(dim=-1, keepdim=True)
            logit = logit_scale * image_features_norm[i] @ text_features.t()
            logits_list.append(logit)

        logits = torch.stack(logits_list)
        return logits


class DynamicPromptEvaluator:
    def __init__(self, backbone, checkpoint_dir, classnames, device):
        self.device = device
        self.clip_model = load_clip_model(backbone, device)

        print(f"Loading DynamicPrompt model from {checkpoint_dir}...")
        checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model-best.pth.tar")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar-20")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: DynamicPrompt checkpoint not found at {checkpoint_path}")
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Loaded DynamicPrompt checkpoint from epoch {epoch}")
        print(f"Available state_dict keys: {list(state_dict.keys())[:10]}...")

        n_ctx = state_dict['ctx'].shape[0]
        ctx = state_dict['ctx'].to(device)

        meta_net_state_dict = None
        soft_prompt_keys = [k for k in state_dict.keys() if 'soft_prompt_adapter' in k or 'meta_net' in k]
        if soft_prompt_keys:
            print(f"Soft prompt related keys: {soft_prompt_keys}")
            if 'soft_prompt_adapter.meta_net.linear1.weight' in state_dict:
                meta_net_state_dict = {
                    'linear1.weight': state_dict['soft_prompt_adapter.meta_net.linear1.weight'],
                    'linear1.bias': state_dict['soft_prompt_adapter.meta_net.linear1.bias'],
                    'linear2.weight': state_dict['soft_prompt_adapter.meta_net.linear2.weight'],
                    'linear2.bias': state_dict['soft_prompt_adapter.meta_net.linear2.bias'],
                }
            elif 'meta_net.linear1.weight' in state_dict:
                meta_net_state_dict = {
                    'linear1.weight': state_dict['meta_net.linear1.weight'],
                    'linear1.bias': state_dict['meta_net.linear1.bias'],
                    'linear2.weight': state_dict['meta_net.linear2.weight'],
                    'linear2.bias': state_dict['meta_net.linear2.bias'],
                }

        class_adaptive_factors = None
        if 'class_adaptive_factors' in state_dict:
            class_adaptive_factors = state_dict['class_adaptive_factors'].to(device)
        else:
            class_adaptive_factors = torch.ones(1, n_ctx, ctx.shape[-1], device=device)

        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx:, :]

        self.model = DynamicPromptClassifier(
            self.clip_model, classnames, ctx, class_adaptive_factors,
            meta_net_state_dict, tokenized_prompts, token_prefix, token_suffix
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader):
        if self.model is None:
            return {"accuracy": 0, "predictions": [], "labels": [], "probabilities": torch.zeros(1)}

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="DynamicPrompt"):
            images = batch['img'].to(self.device)
            labels = batch['label']

            logits = self.model(images)
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels.to(self.device)).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs
        }


class CoCoOpEvaluator(DynamicPromptEvaluator):
    def __init__(self, backbone, checkpoint_dir, classnames, device):
        self.device = device
        self.clip_model = load_clip_model(backbone, device)

        print(f"Loading CoCoOp model from {checkpoint_dir}...")
        checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model-best.pth.tar")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar-20")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: CoCoOp checkpoint not found at {checkpoint_path}")
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Loaded CoCoOp checkpoint from epoch {epoch}")

        n_ctx = state_dict['ctx'].shape[0]
        ctx = state_dict['ctx'].to(device)

        meta_net_state_dict = None
        if 'meta_net.linear1.weight' in state_dict:
            meta_net_state_dict = {
                'meta_net.linear1.weight': state_dict['meta_net.linear1.weight'],
                'meta_net.linear1.bias': state_dict['meta_net.linear1.bias'],
                'meta_net.linear2.weight': state_dict['meta_net.linear2.weight'],
                'meta_net.linear2.bias': state_dict['meta_net.linear2.bias'],
            }

        class_adaptive_factors = torch.ones(1, n_ctx, ctx.shape[-1], device=device)

        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx:, :]

        self.model = DynamicPromptClassifier(
            self.clip_model, classnames, ctx, class_adaptive_factors,
            meta_net_state_dict, tokenized_prompts, token_prefix, token_suffix
        )
        self.model.to(device)
        self.model.eval()


def compute_top_k_accuracy(probs, labels, k=5):
    topk_preds = torch.topk(probs, k=k, dim=1).indices
    correct = 0
    for i, label in enumerate(labels):
        if label in topk_preds[i]:
            correct += 1
    return correct / len(labels)


def plot_accuracy_comparison(results, output_dir):
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]

    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = plt.bar(models, accuracies, color=colors[:len(models)])

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Fine-Grained Classification Accuracy Comparison", fontsize=14)
    plt.ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved accuracy comparison to {os.path.join(output_dir, 'accuracy_comparison.png')}")


def plot_confidence_distribution(results, output_dir):
    plt.figure(figsize=(12, 6))

    for i, (model_name, result) in enumerate(results.items()):
        probs = result["probabilities"]
        if probs.shape[0] == 1:
            continue
        max_probs = probs.max(dim=1).values.numpy()

        plt.subplot(1, len(results), i + 1)
        plt.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Count")
        plt.title(f"{model_name}\nMean: {max_probs.mean():.3f}")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved confidence distribution to {os.path.join(output_dir, 'confidence_distribution.png')}")


def plot_top_k_accuracies(results, output_dir, k_values=[1, 3, 5]):
    valid_results = {k: v for k, v in results.items() if v["probabilities"].shape[0] > 1}
    if not valid_results:
        print("No valid results for Top-K comparison")
        return

    labels = list(valid_results.values())[0]["labels"]
    labels_tensor = torch.tensor(labels)

    topk_results = {}
    for model_name, result in valid_results.items():
        probs = result["probabilities"]
        topk_accs = {}
        for k in k_values:
            topk_accs[k] = compute_top_k_accuracy(probs, labels_tensor, k) * 100
        topk_results[model_name] = topk_accs

    x = np.arange(len(k_values))
    width = 0.2
    num_models = len(topk_results)

    plt.figure(figsize=(12, 6))
    for i, (model_name, topk_accs) in enumerate(topk_results.items()):
        accuracies = [topk_accs[k] for k in k_values]
        offset = (i - num_models/2 + 0.5) * width
        bars = plt.bar(x + offset, accuracies, width, label=model_name)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.xlabel("Top-K")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-K Accuracy Comparison")
    plt.xticks(x, [f"Top-{k}" for k in k_values])
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_k_accuracies.png"), dpi=150)
    plt.close()
    print(f"Saved Top-K accuracies to {os.path.join(output_dir, 'top_k_accuracies.png')}")


def print_detailed_results(results):
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")

        probs = result["probabilities"]
        if probs.shape[0] == 1:
            continue

        labels = result["labels"]
        preds = result["predictions"]

        max_probs = probs.max(dim=1).values
        print(f"  Mean Confidence: {max_probs.mean()*100:.2f}%")
        print(f"  Median Confidence: {max_probs.median()*100:.2f}%")

        correct_mask = torch.tensor(preds) == torch.tensor(labels)
        incorrect_mask = ~correct_mask

        if correct_mask.any():
            correct_confs = max_probs[correct_mask]
            print(f"  Correct Prediction Confidence: {correct_confs.mean()*100:.2f}%")
        if incorrect_mask.any():
            incorrect_confs = max_probs[incorrect_mask]
            print(f"  Incorrect Prediction Confidence: {incorrect_confs.mean()*100:.2f}%")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = "cpu"

    print("="*60)
    print("FINE-GRAINED CLASSIFICATION MODEL COMPARISON")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Models to compare: {args.models}")
    print("="*60)

    classnames = OXFORD_PETS_CLASSNAMES

    print("\nLoading test dataset...")
    dataset = OxfordPetsDataset(args.data_root, split="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test samples: {len(dataset)}")

    results = {}

    model_dirs = {}
    if args.model_dirs:
        model_dirs = json.loads(args.model_dirs)

    if "clip" in args.models:
        print("\n" + "-"*40)
        print("Evaluating Zero-shot CLIP")
        print("-"*40)
        evaluator = ZeroShotCLIP(args.backbone, classnames, device)
        results["Zero-shot CLIP"] = evaluator.evaluate(dataloader)

    if "coop" in args.models:
        print("\n" + "-"*40)
        print("Evaluating CoOp")
        print("-"*40)
        coop_dir = model_dirs.get("coop", f"{PROJECT_ROOT}/fine_grained_classification/output_fgd/{args.dataset}/CoOp/shots_16/seed_1")
        evaluator = CoOpEvaluator(args.backbone, coop_dir, classnames, device)
        results["CoOp"] = evaluator.evaluate(dataloader)

    if "cocoop" in args.models:
        print("\n" + "-"*40)
        print("Evaluating CoCoOp")
        print("-"*40)
        cocoop_dir = model_dirs.get("cocoop", f"{PROJECT_ROOT}/fine_grained_classification/output_fgd/{args.dataset}/CoCoOp/shots_16/seed_1")
        evaluator = CoCoOpEvaluator(args.backbone, cocoop_dir, classnames, device)
        results["CoCoOp"] = evaluator.evaluate(dataloader)

    if "dynamic" in args.models:
        print("\n" + "-"*40)
        print("Evaluating DynamicPrompt")
        print("-"*40)
        dynamic_dir = model_dirs.get("dynamic", f"{PROJECT_ROOT}/fine_grained_classification/output_fgd/{args.dataset}/DynamicPromptTrainer/shots_16/seed_1")
        evaluator = DynamicPromptEvaluator(args.backbone, dynamic_dir, classnames, device)
        results["DynamicPrompt"] = evaluator.evaluate(dataloader)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<15}")
    print("-"*35)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['accuracy']*100:.2f}%")
    print("="*60)

    print_detailed_results(results)

    print("\nGenerating visualization plots...")
    plot_accuracy_comparison(results, args.output_dir)
    plot_confidence_distribution(results, args.output_dir)
    plot_top_k_accuracies(results, args.output_dir)

    summary = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "models": {},
        "best_model": max(results.keys(), key=lambda k: results[k]["accuracy"])
    }

    for model_name, result in results.items():
        summary["models"][model_name] = {
            "accuracy": result["accuracy"],
            "mean_confidence": result["probabilities"].max(dim=1).values.mean().item() if result["probabilities"].shape[0] > 1 else 0
        }

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
