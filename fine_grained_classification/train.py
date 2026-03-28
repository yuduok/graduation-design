"""
训练脚本 - 动态提示词细粒度分类
Training Script for Fine-Grained Classification with Dynamic Prompts
"""
import argparse
import gc
import os
import sys

# CUDA 内存优化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# CoOp目录（与项目目录平行）
# 自动查找 CoOp 目录（相对于当前文件的上层目录）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
if os.path.exists(COOP_PATH) and COOP_PATH not in sys.path:
    sys.path.insert(0, COOP_PATH)

# 添加 dassl 包路径（CoOp/dassl/dassl -> CoOp/dassl）
DASSL_PATH = os.path.join(COOP_PATH, "dassl")
if os.path.exists(DASSL_PATH) and DASSL_PATH not in sys.path:
    sys.path.insert(0, DASSL_PATH)

# CoOp数据目录
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# 项目根目录
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 导入CoOp数据集模块（注册数据集）
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

# 导入CoOp trainer 模块
import trainers.coop
import trainers.cocoop
import trainers.zsclip

# 导入自定义模块
import models.trainer  # 注册 DynamicPromptTrainer

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train fine-grained classifier with dynamic prompts")
    parser.add_argument("-root", "--root", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                       help="path to project root")
    parser.add_argument("-c", "--config", type=str, default="configs/dynamic_rn50.yaml",
                       help="path to config file")
    parser.add_argument("-s", "--save-dir", type=str, default=None,
                       help="directory to save training outputs (auto-generated if not specified)")
    parser.add_argument("-d", "--dataset", type=str, default="oxford_pets",
                       choices=["oxford_pets", "oxford_flowers", "stanford_cars", "food101", "caltech101", "sun397", "ucf101", "fgvc_aircraft", "dtd", "eurosat", "imagenet"],
                       help="dataset name")
    parser.add_argument("-t", "--trainer", type=str, default="DynamicPromptTrainer",
                       choices=["CoOp", "CoCoOp", "DynamicPromptTrainer"],
                       help="trainer name")
    parser.add_argument("-b", "--batch-size", type=int, default=16,
                       help="batch size (default 16)")
    parser.add_argument("-e", "--epochs", type=int, default=None,
                       help="number of epochs")
    parser.add_argument("--seed", type=int, default=1,
                       help="random seed")
    parser.add_argument("-j", "--num-workers", type=int, default=8,
                       help="number of data workers")
    parser.add_argument("--eval-only", action="store_true",
                       help="only perform evaluation")
    parser.add_argument("--model-dir", type=str, default="",
                       help="path to model checkpoint for evaluation")
    parser.add_argument("--rate", type=int, default=1,
                       help="rate of few-shot")
    parser.add_argument("--shots", type=int, default=None,
                       help="number of few-shot data")
    parser.add_argument("--split", type=str, default="1",
                       choices=["0", "1", "2", "3"],
                       help="dataset split for few-shot")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cuda", "cpu", "mps"],
                       help="device to use")
    
    return parser.parse_args()


def extend_cfg(cfg):
    """扩展默认配置，添加自定义变量"""
    from yacs.config import CfgNode as CN
    
    # CoOp trainer 配置
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    
    # CoCoOp trainer 配置
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"
    
    # 数据集配置
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.SPLIT = 0
    cfg.DATASET.NUM_SHOTS = 0
    
    # 评估配置
    cfg.EVAL_ONLY = False
    cfg.MODEL_DIR = ""
    
    # 动态提示词配置
    cfg.TRAINER.DYNAMIC = CN()
    cfg.TRAINER.DYNAMIC.CTX_INIT = "a photo of a"
    cfg.TRAINER.DYNAMIC.N_CTX = 16
    cfg.TRAINER.DYNAMIC.PREC = "fp32"
    cfg.TRAINER.DYNAMIC.MODEL_TYPE = "dynamic"
    cfg.TRAINER.DYNAMIC.USE_DYNAMIC = True
    cfg.TRAINER.DYNAMIC.USE_ADAPTIVE = True
    cfg.TRAINER.DYNAMIC.USE_DIFFICULTY_WEIGHT = True
    cfg.TRAINER.DYNAMIC.ALPHA = 0.1
    cfg.TRAINER.DYNAMIC.BETA = 0.01
    cfg.TRAINER.DYNAMIC.ADAPTIVE_HIDDEN_DIM = 64
    cfg.TRAINER.DYNAMIC.USE_SEMANTIC_ENHANCEMENT = False
    cfg.TRAINER.DYNAMIC.USE_ATTRIBUTE_DATABASE = True
    cfg.TRAINER.DYNAMIC.MONITOR_INTERVAL = 60
    cfg.TRAINER.DYNAMIC.LR = 0.001  # 降低学习率（原0.002对动态提示词过高）


def setup_cfg(args):
    """设置配置"""
    cfg = get_cfg_default()
    
    # 先扩展配置，添加自定义变量
    extend_cfg(cfg)
    
    # 使用绝对路径
    config_path = os.path.join(args.root, args.config)
    cfg.merge_from_file(config_path)
    
    # 合并命令行参数
    if args.root:
        cfg.ROOT = args.root
    
    # 设置数据集根目录（使用CoOp的data目录）
    cfg.DATASET.ROOT = DATA_PATH
    
    if args.dataset:
        # 数据集名称映射（用户友好名称 -> CoOp内部名称）
        dataset_name_map = {
            "oxford_pets": "OxfordPets",
            "oxford_flowers": "OxfordFlowers",
            "stanford_cars": "StanfordCars",
            "food101": "Food101",
            "caltech101": "Caltech101",
            "sun397": "SUN397",
            "ucf101": "UCF101",
            "fgvc_aircraft": "FGVCAircraft",
            "dtd": "DescribableTextures",
            "eurosat": "EuroSAT",
            "imagenet": "ImageNet"
        }
        cfg.DATASET.NAME = dataset_name_map.get(args.dataset, args.dataset)
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.batch_size:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
        cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size * 2
    if args.epochs:
        cfg.OPTIM.MAX_EPOCH = args.epochs
    if args.num_workers:
        cfg.DATALOADER.NUM_WORKERS = args.num_workers
    if args.rate:
        cfg.DATASET.NUM_SHOTS = args.rate
    if args.shots:
        cfg.DATASET.NUM_SHOTS = args.shots
    if args.split:
        cfg.DATASET.SPLIT = args.split
    if args.eval_only:
        cfg.EVAL_ONLY = True
    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
    
    # 设置保存目录：自动按 trainer/shots/seed 分组
    if args.save_dir:
        cfg.OUTPUT_DIR = os.path.join(args.root, args.save_dir)
    else:
        shots = args.shots if args.shots else args.rate
        save_dir = f"output_fgd/{args.dataset}/{args.trainer}/shots_{shots}/seed_{args.seed}"
        cfg.OUTPUT_DIR = os.path.join(args.root, save_dir)
    
    # 设置随机种子
    cfg.SEED = args.seed
    
    # 确保输出目录存在
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


def main():
    args = parse_args()
    
    # 设置配置
    cfg = setup_cfg(args)
    
    # 打印配置
    print("="*60)
    print("Configuration:")
    print(f"  Dataset: {cfg.DATASET.NAME}")
    print(f"  Trainer: {cfg.TRAINER.NAME}")
    print(f"  Backbone: {cfg.MODEL.BACKBONE.NAME}")
    print(f"  Batch size: {cfg.DATALOADER.TRAIN_X.BATCH_SIZE}")
    print(f"  Epochs: {cfg.OPTIM.MAX_EPOCH}")
    print(f"  Shots: {cfg.DATASET.NUM_SHOTS}")
    print(f"  Device: {args.device}")
    print(f"  Config: {os.path.join(args.root, args.config)}")
    print("="*60)
    
    # 设置日志
    setup_logger(cfg.OUTPUT_DIR)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    
    # 打印环境信息
    print("\nEnvironment info:")
    print(collect_env_info())
    print()
    
    # CUDA 内存优化：训练前清理缓存
    if args.device == "cuda":
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print("[Memory] GPU cache cleared before training\n")
    
    # 构建训练器
    trainer = build_trainer(cfg)
    
    # 如果只做评估
    if cfg.EVAL_ONLY:
        print("\n" + "="*60)
        print("Evaluation only mode")
        print("="*60 + "\n")
        trainer.load_model(cfg.MODEL_DIR)
        trainer.test()
        return
    
    # 训练（训练结束后会自动测试）
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()


if __name__ == "__main__":
    main()
