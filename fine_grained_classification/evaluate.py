"""
评估脚本 - 测试训练好的模型
Evaluation Script for Fine-Grained Classification
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_clip import load_clip_to_cpu, build_custom_clip
from utils.helpers import (
    visualize_predictions, visualize_attention_map,
    compute_confusion_matrix, plot_confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-grained classifier")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="path to trained model directory")
    parser.add_argument("--data-dir", type=str, 
                        default="/Users/yudu/Documents/毕业设计/data",
                        help="path to dataset")
    parser.add_argument("--output", type=str, default="eval_results",
                        help="output directory for results")
    parser.add_argument("--backbone", type=str, default="RN50",
                        choices=["RN50", "RN101", "ViT-B/32"],
                        help="CLIP backbone")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--top-k", type=int, default=5,
                        help="show top-k predictions")
    return parser.parse_args()


def evaluate(args):
    """评估模型"""
    device = "cuda" if torch.cuda.is_available() else "mps"
    if device == "cpu":
        device = "cpu"
    
    print(f"Device: {device}")
    print(f"Model dir: {args.model_dir}")
    
    # 加载类别名称
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
    
    # 加载CLIP
    print("Loading CLIP...")
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model.eval()
    clip_model.float()
    
    # 构建自定义模型
    print("Building model...")
    
    # 简单实现：直接使用CLIP + 提示词
    model = SimpleClassifier(clip_model, classnames, device)
    
    # 加载检查点
    checkpoint_path = os.path.join(args.model_dir, "model-best.pth.tar")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best accuracy: {checkpoint.get('best_result', 'N/A')}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 评估
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    # 加载验证集
    val_dataset = OxfordPetsDataset(args.data_dir, split="test")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(val_loader):
        images = batch['img'].to(device)
        labels = batch['label']
        
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
            probs = logits.softmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())
        
        correct += (preds == labels.to(device)).sum().item()
        total += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(val_loader)}")
    
    # 汇总结果
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    accuracy = correct / total
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Top-K 准确率
    for k in [1, 3, 5]:
        correct_k = 0
        for probs, labels_batch in zip(all_probs, all_labels):
            topk_preds = probs.topk(k)[1]
            correct_k += sum(1 for pred in topk_preds if pred == labels_batch.item())
        print(f"Top-{k} Accuracy: {correct_k/total:.2%}")
    
    # 混淆矩阵
    print("\nComputing confusion matrix...")
    cm = compute_confusion_matrix(all_preds, all_labels, classnames)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.output, "confusion_matrix.png")
    fig = plot_confusion_matrix(cm, classnames, save_path=cm_path, normalize=True)
    print(f"Saved confusion matrix to {cm_path}")
    
    # 保存结果
    results = {
        "accuracy": accuracy,
        "total_samples": total,
        "classnames": classnames,
    }
    
    import json
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    return accuracy


class SimpleClassifier(nn.Module):
    """简单分类器：CLIP + 提示词"""
    
    def __init__(self, clip_model, classnames, device):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.device = device
        self.dtype = clip_model.dtype
        
        # 生成提示词
        self.prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
        self.tokens = clip.tokenize(self.prompts).to(device)
        
    def forward(self, images):
        with torch.no_grad():
            # 编码图像
            image_features = self.clip_model.encode_image(images.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 编码文本
            text_features = self.clip_model.encode_text(self.tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T
            
        return logits


class OxfordPetsDataset:
    """Oxford-IIIT Pets 数据集简化版"""
    
    def __init__(self, data_dir, split="test"):
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        
        # 加载划分
        split_file = os.path.join(data_dir, "split_zhou_OxfordPets.json")
        if os.path.exists(split_file):
            import json
            with open(split_file, 'r') as f:
                splits = json.load(f)
            self.data = splits.get(split, [])
        else:
            # 使用全部图片
            self.data = []
            for img_file in os.listdir(self.image_dir):
                if img_file.endswith('.jpg'):
                    self.data.append({'image': img_file, 'label': 0})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(img_path).convert('RGB')
        
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        return {
            'img': transform(image),
            'label': item['label']
        }


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
