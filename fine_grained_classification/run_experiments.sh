#!/bin/bash
# 自动化实验脚本 - 运行全部对比实验
# 3 个方法 (DynamicPromptTrainer, CoOp, CoCoOp) × 5 个 shot 设置 (1, 2, 4, 8, 16)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 用法: bash run_experiments.sh [device] [batch_size] [epochs_list] [backbone_list]
# 示例: bash run_experiments.sh cuda 16 "100,80,60,40,20" "RN50"
#   - device: cuda 或 cpu (默认: cuda)
#   - batch_size: 批大小 (默认: 16)
#   - epochs_list: 1,2,4,8,16-shot 对应的 epoch 数，用逗号分隔 (默认: 100,80,60,40,20)
#   - backbone_list: 逗号分隔的 backbone 名称 (默认: RN50)

# 设备选择
DEVICE="${1:-cuda}"
BATCH_SIZE="${2:-16}"
EPOCHS_LIST="${3:-100,80,60,40,20}"
BACKBONE_LIST="${4:-RN50}"
DATASET="oxford_pets"
SEED=1

# 解析 epochs 列表
IFS=',' read -ra EPOCHS_ARR <<< "$EPOCHS_LIST"
# 解析 backbone 列表
IFS=',' read -ra BACKBONES_ARR <<< "$BACKBONE_LIST"
SHOTS_LIST=(1 2 4 8 16)

echo "=============================================="
echo "  Fine-Grained Classification Experiments"
echo "=============================================="
echo "  Device:       $DEVICE"
echo "  Batch size:   $BATCH_SIZE"
echo "  Epochs list:  $EPOCHS_LIST (1,2,4,8,16-shot)"
echo "  Backbones:    $BACKBONE_LIST"
echo "  Dataset:      $DATASET"
echo "  Seed:         $SEED"
echo "=============================================="
echo ""

TRAINERS=("DynamicPromptTrainer" "CoOp" "CoCoOp")

TOTAL=$((${#TRAINERS[@]} * ${#SHOTS_LIST[@]} * ${#BACKBONES_ARR[@]}))
CURRENT=0
FAILED=0
RESULTS_LOG="$SCRIPT_DIR/experiment_results.log"
echo "Experiment Results - $(date)" > "$RESULTS_LOG"
echo "==========================================" >> "$RESULTS_LOG"

for TRAINER in "${TRAINERS[@]}"; do
    for BACKBONE in "${BACKBONES_ARR[@]}"; do
        for i in "${!SHOTS_LIST[@]}"; do
            SHOTS=${SHOTS_LIST[$i]}
            CURRENT=$((CURRENT + 1))

            # 从外部传入的 epochs 列表获取对应的 epoch
            EPOCHS=${EPOCHS_ARR[$i]}

            echo ""
            echo "[$CURRENT/$TOTAL] Running: $TRAINER with $SHOTS-shot (epochs=$EPOCHS, backbone=$BACKBONE)"
            echo "----------------------------------------------"

            START_TIME=$(date +%s)

            python train.py \
                -d "$DATASET" \
                -e "$EPOCHS" \
                -b "$BATCH_SIZE" \
                --shots "$SHOTS" \
                --trainer "$TRAINER" \
                --backbone "$BACKBONE" \
                --device "$DEVICE" \
                --seed "$SEED" \
                2>&1 | tee -a "$RESULTS_LOG"

            EXIT_CODE=${PIPESTATUS[0]}
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            if [ $EXIT_CODE -eq 0 ]; then
                echo "[$CURRENT/$TOTAL] DONE: $TRAINER $SHOTS-shot $BACKBONE (${DURATION}s)" | tee -a "$RESULTS_LOG"
            else
                echo "[$CURRENT/$TOTAL] FAILED: $TRAINER $SHOTS-shot $BACKBONE (exit code $EXIT_CODE)" | tee -a "$RESULTS_LOG"
                FAILED=$((FAILED + 1))
            fi
            echo ""
        done
    done
done

echo ""
echo "=============================================="
echo "  All experiments finished!"
echo "  Total: $TOTAL | Failed: $FAILED"
echo "  Results log: $RESULTS_LOG"
echo "=============================================="
echo ""
echo "Run 'python collect_results.py' to generate comparison tables."
