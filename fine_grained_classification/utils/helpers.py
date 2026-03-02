"""
工具函数模块
Utility Functions for Fine-Grained Classification
"""
import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset_splits(data_dir, split_file=None):
    """
    加载数据集划分
    Args:
        data_dir: 数据集根目录
        split_file: 划分文件路径
    Returns:
        splits: 划分字典
    """
    if split_file is None:
        split_file = os.path.join(data_dir, "split_zhou_OxfordPets.json")
    
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        return splits
    else:
        print(f"Warning: Split file not found at {split_file}")
        return None


def get_class_names(data_dir):
    """
    获取类别名称
    Args:
        data_dir: 数据集根目录
    Returns:
        class_names: 类别名称列表
    """
    # 从annotations文件夹读取类别名称
    annotations_dir = os.path.join(data_dir, "annotations")
    
    # 尝试读取annotations文件
    # Oxford-IIIT Pets使用annotations.txt
    annotations_file = os.path.join(annotations_dir, "annotations.txt")
    
    if os.path.exists(annotations_file):
        class_names = []
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # 格式: image_name class_id species breed_id
                parts = line.split()
                if len(parts) >= 3:
                    breed_name = parts[1]  # breed名称
                    if breed_name not in class_names:
                        class_names.append(breed_name)
        return class_names
    else:
        # 备用方案：从images文件夹读取
        print("Warning: annotations.txt not found, reading from images folder")
        images_dir = os.path.join(data_dir, "images")
        if os.path.exists(images_dir):
            class_names = sorted([d for d in os.listdir(images_dir) 
                                  if os.path.isdir(os.path.join(images_dir, d))])
            return class_names
        else:
            raise FileNotFoundError(f"Cannot find class names in {data_dir}")


def load_image(image_path, target_size=(224, 224)):
    """
    加载图像
    Args:
        image_path: 图像路径
        target_size: 目标尺寸
    Returns:
        image: PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image


def visualize_predictions(image, predictions, class_names, top_k=5, save_path=None):
    """
    可视化预测结果
    Args:
        image: PIL Image
        predictions: 预测概率 [n_classes]
        class_names: 类别名称列表
        top_k: 显示top-k预测
        save_path: 保存路径（可选）
    Returns:
        fig: matplotlib figure
    """
    # 获取top-k预测
    top_probs, top_indices = torch.topk(predictions, top_k)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    top_classes = [class_names[i].replace("_", " ") for i in top_indices]
    
    # 创建图像
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左侧：显示图像
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    
    # 右侧：显示预测结果
    y_pos = np.arange(top_k)
    bars = axes[1].barh(y_pos, top_probs, color='steelblue')
    
    # 设置y轴标签
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_classes, fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title(f'Top-{top_k} Predictions', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    
    # 添加概率标签
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        width = bar.get_width()
        axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.2%}',
                    ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_attention_map(image, attention_map, alpha=0.5, save_path=None):
    """
    可视化注意力图
    Args:
        image: PIL Image
        attention_map: 注意力图 [H, W]
        alpha: 透明度
        save_path: 保存路径（可选）
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    
    # 注意力图
    im = axes[1].imshow(attention_map, cmap='hot')
    axes[1].axis('off')
    axes[1].set_title('Attention Map', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加图
    axes[2].imshow(image)
    axes[2].imshow(attention_map, cmap='hot', alpha=alpha)
    axes[2].axis('off')
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention map to {save_path}")
    
    return fig


def compute_confusion_matrix(predictions, labels, class_names):
    """
    计算混淆矩阵
    Args:
        predictions: 预测标签
        labels: 真实标签
        class_names: 类别名称列表
    Returns:
        cm: 混淆矩阵
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
    return cm


def plot_confusion_matrix(cm, class_names, save_path=None, 
                        normalize=False, title="Confusion Matrix"):
    """
    绘制混淆矩阵
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径（可选）
        normalize: 是否归一化
        title: 标题
    Returns:
        fig: matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def print_training_summary(trainer):
    """
    打印训练摘要
    Args:
        trainer: 训练器对象
    """
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total epochs: {trainer.current_epoch}")
    print(f"Best accuracy: {trainer.best_result:.2%}")
    print(f"Output directory: {trainer.cfg.OUTPUT_DIR}")
    print("="*60 + "\n")


def save_results(predictions, labels, class_names, output_path):
    """
    保存结果
    Args:
        predictions: 预测结果
        labels: 真实标签
        class_names: 类别名称
        output_path: 输出路径
    """
    results = {
        "predictions": predictions.cpu().numpy().tolist(),
        "labels": labels.cpu().numpy().tolist(),
        "class_names": class_names
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_path}")


def setup_seed(seed):
    """
    设置随机种子
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def count_parameters(model):
    """
    计算模型参数数量
    Args:
        model: 模型
    Returns:
        total_params: 总参数数
        trainable_params: 可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return total_params, trainable_params


def get_device():
    """
    获取设备
    Returns:
        device: torch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device
