"""
训练脚本 - 动态提示词细粒度分类
Training Script for Fine-Grained Classification with Dynamic Prompts
"""
import argparse
import os
import sys

# 添加CoOp路径到sys.path
coop_path = os.path.join(os.path.dirname(__file__), "CoOp")
if coop_path not in sys.path:
    sys.path.insert(0, coop_path)

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train fine-grained classifier with dynamic prompts")
    parser.add_argument("-root", "--root", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                       help="path to project root")
    parser.add_argument("-c", "--config", type=str, default="configs/dynamic_rn50.yaml",
                       help="path to config file")
    parser.add_argument("-s", "--save-dir", type=str, default="output_fgd/oxford_pets",
                       help="directory to save training outputs")
    parser.add_argument("-d", "--dataset", type=str, default="oxford_pets",
                       choices=["oxford_pets", "caltech101", "food101"],
                       help="dataset name")
    parser.add_argument("-t", "--trainer", type=str, default="DynamicPromptTrainer",
                       choices=["CoOp", "CoCoOp", "DynamicPromptTrainer"],
                       help="trainer name")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                       help="batch size")
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
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="device to use")
    
    return parser.parse_args()


def setup_cfg(args):
    """设置配置"""
    cfg = get_cfg_default()
    
    # 使用绝对路径
    config_path = os.path.join(args.root, args.config)
    cfg.merge_from_file(config_path)
    
    # 合并命令行参数
    if args.root:
        cfg.ROOT = args.root
    if args.dataset:
        cfg.DATASET.NAME = args.dataset
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
    
    # 设置保存目录
    if args.save_dir:
        cfg.OUTPUT_DIR = os.path.join(args.root, args.save_dir)
    
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
    
    # 训练
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # 测试
    print("\n" + "="*60)
    print("Testing...")
    print("="*60 + "\n")
    trainer.test()


if __name__ == "__main__":
    main()
