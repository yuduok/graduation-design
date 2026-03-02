"""
Demo独立测试脚本
"""
import os
import sys
import torch
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保使用CoOp的clip
sys.path.insert(0, '/Users/yudu/Documents/毕业设计/CoOp')
import clip

# 添加项目路径
PROJECT_ROOT = "/Users/yudu/Documents/毕业设计/fine_grained_classification"
sys.path.insert(0, PROJECT_ROOT)

from demo.pet_classifier_demo import PetClassifierDemo

print("="*60)
print("Demo功能测试")
print("="*60)

# 创建分类器
print("\n[1] 初始化分类器...")
classifier = PetClassifierDemo()

# 创建测试图像
print("\n[2] 创建测试图像...")
test_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
test_image = Image.fromarray(test_array).convert('RGB')
print(f"  图像尺寸: {test_image.size}")

# 执行分类
print("\n[3] 执行分类...")
probs, prompts = classifier.classify(test_image, use_prompts=True)
print(f"  概率维度: {probs.shape}")

# 获取Top-5
print("\n[4] Top-5预测结果:")
results = classifier.get_top_predictions(probs, prompts, top_k=5)
for i, result in enumerate(results):
    print(f"  {i+1}. {result['breed'].replace('_', ' ')}: {result['probability']:.2%}")

# 可视化
print("\n[5] 生成可视化图...")
fig = classifier.visualize_results(test_image, results)
save_path = "/Users/yudu/Documents/毕业设计/fine_grained_classification/demo_test_output.png"
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  保存到: {save_path}")

print("\n" + "="*60)
print("Demo测试通过!")
print("="*60)
print("\n启动Streamlit界面:")
print("  streamlit run /Users/yudu/Documents/毕业设计/fine_grained_classification/demo/pet_classifier_demo.py")
