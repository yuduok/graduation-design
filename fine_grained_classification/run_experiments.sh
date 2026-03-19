#!/bin/bash
# 自动化实验脚本 - 运行全部对比实验
# 3 个方法 (DynamicPromptTrainer, CoOp, CoCoOp) × 3 个 shot 设置 (1, 4, 16)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 设备选择（可通过命令行参数覆盖）
DEVICE="${1:-cuda}"
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-16}"
DATASET="oxford_pets"
SEED=1

echo "=============================================="
echo "  Fine-Grained Classification Experiments"
echo "=============================================="
echo "  Device:     $DEVICE"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Dataset:    $DATASET"
echo "  Seed:       $SEED"
echo "=============================================="
echo ""

TRAINERS=("DynamicPromptTrainer" "CoOp" "CoCoOp")
SHOTS_LIST=(1 4 16)

TOTAL=$((${#TRAINERS[@]} * ${#SHOTS_LIST[@]}))
CURRENT=0
FAILED=0
RESULTS_LOG="$SCRIPT_DIR/experiment_results.log"
echo "Experiment Results - $(date)" > "$RESULTS_LOG"
echo "==========================================" >> "$RESULTS_LOG"

for TRAINER in "${TRAINERS[@]}"; do
    for SHOTS in "${SHOTS_LIST[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[$CURRENT/$TOTAL] Running: $TRAINER with $SHOTS-shot"
        echo "----------------------------------------------"

        START_TIME=$(date +%s)

        python train.py \
            -d "$DATASET" \
            -e "$EPOCHS" \
            -b "$BATCH_SIZE" \
            --shots "$SHOTS" \
            --trainer "$TRAINER" \
            --device "$DEVICE" \
            --seed "$SEED" \
            2>&1 | tee -a "$RESULTS_LOG"

        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$CURRENT/$TOTAL] DONE: $TRAINER $SHOTS-shot (${DURATION}s)" | tee -a "$RESULTS_LOG"
        else
            echo "[$CURRENT/$TOTAL] FAILED: $TRAINER $SHOTS-shot (exit code $EXIT_CODE)" | tee -a "$RESULTS_LOG"
            FAILED=$((FAILED + 1))
        fi
        echo ""
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
