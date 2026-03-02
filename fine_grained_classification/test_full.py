"""
完整端到端测试
"""
import os
import sys
sys.path.insert(0, '/Users/yudu/Documents/毕业设计/CoOp')

import torch
import clip
from PIL import Image
import numpy as np

print("="*60)
print("细粒度分类系统 - 完整端到端测试")
print("="*60)

device = "cpu"
print(f"\n[1] 设备: {device}")

# 加载模型
print("\n[2] 加载CLIP RN50...")
model, preprocess = clip.load("RN50", device=device)
model.float().eval()
print("  ✓ CLIP加载成功")

# 数据集路径
data_dir = "/Users/yudu/Documents/毕业设计/data/oxford_pets"
images_dir = os.path.join(data_dir, "images")
print(f"\n[3] 数据集: {images_dir}")
print(f"  存在: {os.path.exists(images_dir)}")

if os.path.exists(images_dir):
    # 列出一些图片
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"  图片数量: {len(all_images)}")
    
    if len(all_images) > 0:
        # 测试分类一张图片
        test_img = all_images[0]
        img_path = os.path.join(images_dir, test_img)
        print(f"\n[4] 测试图片: {test_img}")
        
        image = Image.open(img_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # 类别名称
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
        
        # 提取品种名（去掉数字后缀）
        test_breed = '_'.join(test_img.split('_')[:-1])
        print(f"  真实品种: {test_breed}")
        
        # 生成提示词
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
        text_tokens = clip.tokenize(prompts).to(device)
        
        # 编码
        print("\n[5] 执行分类...")
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits = (image_features @ text_features.T).squeeze()
            probs = logits.softmax(dim=-1)
        
        # Top-5
        top5_prob, top5_idx = torch.topk(probs, 5)
        print(f"\n[6] Top-5预测结果:")
        for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx)):
            pred_breed = classnames[idx.item()]
            match = "✓" if pred_breed == test_breed else ""
            print(f"  {i+1}. {pred_breed}: {prob:.2%} {match}")

print("\n" + "="*60)
print("端到端测试完成!")
print("="*60)
