"""
简化核心功能测试脚本
"""
import os
import sys

COOP_PATH = "/Users/yudu/Documents/毕业设计/CoOp"
sys.path.insert(0, COOP_PATH)

print("="*60)
print("细粒度分类系统 - 简化功能测试")
print("="*60)

# 1. 测试环境
print("\n[1] 环境检查")
import torch
print(f"  PyTorch: {torch.__version__}")

# 2. 测试CLIP加载（CPU模式）
print("\n[2] 加载CLIP模型 (CPU模式)")
try:
    import clip
    clip_model, _ = clip.load("RN50", device="cpu")
    clip_model.eval()
    print("  ✓ CLIP RN50 加载成功")
except Exception as e:
    print(f"  ✗ CLIP加载失败: {e}")
    sys.exit(1)

# 3. 测试动态提示词模块导入
print("\n[3] 测试动态提示词模块")
try:
    sys.path.insert(0, '/Users/yudu/Documents/毕业设计/fine_grained_classification')
    from models.dynamic_prompt import DynamicPromptOptimizer, DifficultyWeightCalculator
    
    optimizer = DynamicPromptOptimizer(n_ctx=16, ctx_dim=512)
    calc = DifficultyWeightCalculator()
    print(f"  ✓ DynamicPromptOptimizer 创建成功")
    print(f"  ✓ DifficultyWeightCalculator 创建成功")
except Exception as e:
    print(f"  ✗ 动态提示词模块导入失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试品种语义增强模块
print("\n[4] 测试品种语义增强模块")
try:
    from models.breed_semantic import BreedAttributeDatabase
    
    db = BreedAttributeDatabase()
    persian_attrs = db.get_attributes("Persian")
    prompts = db.get_prompts("Persian")
    print(f"  ✓ BreedAttributeDatabase 创建成功")
    print(f"  Persian提示词: {prompts[:2]}")
except Exception as e:
    print(f"  ✗ 语义增强模块导入失败: {e}")

# 5. 测试dassl依赖
print("\n[5] 测试dassl依赖")
try:
    from dassl.utils import setup_logger, set_random_seed
    from dassl.config import get_cfg_default
    print(f"  ✓ dassl 依赖正常")
except Exception as e:
    print(f"  ✗ dassl 导入失败: {e}")

# 6. 测试训练脚本导入
print("\n[6] 测试训练脚本")
try:
    os.chdir('/Users/yudu/Documents/毕业设计/fine_grained_classification')
    sys.path.insert(0, 'CoOp')
    from train import parse_args, setup_cfg
    print(f"  ✓ train.py 导入成功")
except Exception as e:
    print(f"  ✗ train.py 导入失败: {e}")

# 7. 测试评估脚本导入
print("\n[7] 测试评估脚本")
try:
    from evaluate import SimpleClassifier
    print(f"  ✓ evaluate.py 导入成功")
except Exception as e:
    print(f"  ⚠ evaluate.py 部分导入失败: {e}")

# 8. 测试训练器导入
print("\n[8] 测试训练器")
try:
    from models.trainer import DynamicPromptTrainer
    print(f"  ✓ DynamicPromptTrainer 导入成功")
except Exception as e:
    print(f"  ⚠ 训练器导入失败: {e}")

# 总结
print("\n" + "="*60)
print("核心模块导入测试完成!")
print("="*60)
