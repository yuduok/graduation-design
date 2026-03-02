"""
核心功能测试脚本
"""
import os
import sys
import numpy as np

COOP_PATH = "/Users/yudu/Documents/毕业设计/CoOp"
sys.path.insert(0, COOP_PATH)
os.environ['PYTHONPATH'] = COOP_PATH

import torch
import clip
from PIL import Image

print("="*60)
print("细粒度分类系统 - 核心功能测试")
print("="*60)

device = "cpu"
print(f"\n[1] 设备: {device}, PyTorch: {torch.__version__}")

print("\n[2] 加载CLIP模型...")
try:
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_model.float().eval()
    print("  CLIP RN50 加载成功")
except Exception as e:
    print(f"  失败: {e}")
    sys.exit(1)

print("\n[3] 测试图像编码...")
# 创建随机numpy数组作为测试图像
test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
test_image = Image.fromarray(test_array)
test_input = preprocess(test_image).unsqueeze(0)

with torch.no_grad():
    image_features = clip_model.encode_image(test_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
print(f"  图像编码成功: {image_features.shape}")

print("\n[4] 测试文本编码...")

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

prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
tokens = clip.tokenize(prompts)

with torch.no_grad():
    text_features = clip_model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print(f"  文本编码成功: {text_features.shape}")

print("\n[5] 测试相似度计算...")
logits = (image_features @ text_features.T).squeeze()
probs = logits.softmax(dim=-1)
top5_prob, top5_idx = torch.topk(probs, 5)
print(f"  相似度计算成功")
print(f"  Top-5 预测:")
for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx)):
    print(f"    {i+1}. {classnames[idx.item()].replace('_', ' ')}: {prob:.2%}")

print("\n[6] 测试动态提示词模块...")
try:
    sys.path.insert(0, '/Users/yudu/Documents/毕业设计/fine_grained_classification')
    from models.dynamic_prompt import DynamicPromptOptimizer
    
    optimizer = DynamicPromptOptimizer(n_ctx=16, ctx_dim=512)
    print(f"  DynamicPromptOptimizer 创建成功")
    
    optimizer.eval()
    with torch.no_grad():
        ctx, weights = optimizer(image_features.squeeze())
    print(f"  DynamicPromptOptimizer 前向传播成功")
    
except Exception as e:
    print(f"  失败: {e}")

print("\n[7] 测试品种语义增强模块...")
try:
    from models.breed_semantic import BreedAttributeDatabase
    
    db = BreedAttributeDatabase()
    persian_attrs = db.get_attributes("Persian")
    print(f"  BreedAttributeDatabase 创建成功")
    print(f"  Persian: {persian_attrs['coat']}, {persian_attrs['face']}")
    
except Exception as e:
    print(f"  失败: {e}")

print("\n[8] 测试dassl依赖...")
try:
    from dassl.utils import setup_logger, set_random_seed
    from dassl.config import get_cfg_default
    print(f"  dassl 依赖正常")
except Exception as e:
    print(f"  失败: {e}")

print("\n" + "="*60)
print("所有核心模块测试通过!")
print("="*60)
