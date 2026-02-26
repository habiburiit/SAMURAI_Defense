#!/bin/bash
# =============================================================================
# SAMURAI Framework - Complete Research Pipeline Script
# University of Florida, ECE Department
# =============================================================================
# Usage:
#   bash run_pipeline.sh                          # default settings
#   bash run_pipeline.sh --gpu 1 --epochs 50      # custom settings
#   bash run_pipeline.sh --gpu 0 --dataset CIFAR100 --arch resnet50
# =============================================================================

# ─── Default Configuration ────────────────────────────────────────────────────
GPU=1
DATASET="CIFAR10"
ARCH="resnet18"
EPOCHS=50
SAMPLES=100
ATTACKS="fgsm pgd deepfool cw bim"

# ─── Parse Arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU="$2";     shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --arch)    ARCH="$2";    shift 2 ;;
        --epochs)  EPOCHS="$2";  shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --attacks) ATTACKS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Banner ───────────────────────────────────────────────────────────────────
echo -e "${BLUE}"
echo "============================================================"
echo "  SAMURAI Full Research Pipeline"
echo "============================================================"
echo "  GPU        : $GPU"
echo "  Dataset    : $DATASET"
echo "  Architecture: $ARCH"
echo "  Epochs     : $EPOCHS"
echo "  Samples    : $SAMPLES"
echo "  Attacks    : $ATTACKS"
echo "============================================================"
echo -e "${NC}"

# ─── Helper Function ──────────────────────────────────────────────────────────
run_step() {
    local step_num=$1
    local step_name=$2
    local cmd=$3
    local logfile=$4

    echo -e "${BLUE}[Step $step_num] $step_name${NC}"
    echo "  Command: $cmd"
    echo "  Log: $logfile"

    if eval "CUDA_VISIBLE_DEVICES=$GPU $cmd | tee $logfile"; then
        echo -e "${GREEN}  ✓ Step $step_num complete${NC}"
    else
        echo -e "${RED}  ✗ Step $step_num FAILED! Check $logfile${NC}"
        exit 1
    fi
    echo ""
}

# ─── Step 1: Train Model ──────────────────────────────────────────────────────
run_step 1 "Training $ARCH on $DATASET for $EPOCHS epochs" \
    "python samurai.py --train --dataset $DATASET --architecture $ARCH --epochs $EPOCHS" \
    "logs/1_train.log"

# ─── Step 2: Check Model Accuracy ─────────────────────────────────────────────
echo -e "${BLUE}[Step 2] Checking model accuracy...${NC}"
CUDA_VISIBLE_DEVICES=$GPU python check_accuracy.py \
    --dataset $DATASET --architecture $ARCH --gpu 0 | tee logs/2_accuracy.log
echo ""

# ─── Step 3: Run All Attacks ──────────────────────────────────────────────────
echo -e "${BLUE}[Step 3] Running adversarial attacks...${NC}"
for attack in $ATTACKS; do
    echo -e "  Running $attack attack..."
    CUDA_VISIBLE_DEVICES=$GPU python samurai.py \
        --attack $attack \
        --dataset $DATASET \
        --architecture $ARCH \
        --num_samples $SAMPLES | tee logs/3_attack_${attack}.log
    echo -e "${GREEN}  ✓ $attack complete${NC}"
done
echo ""

# ─── Step 4: Extract APC Metrics ──────────────────────────────────────────────
echo -e "${BLUE}[Step 4] Extracting APC metrics...${NC}"
for attack in $ATTACKS; do
    echo -e "  Extracting APC for $attack..."
    CUDA_VISIBLE_DEVICES=$GPU python samurai.py \
        --apc \
        --attack $attack \
        --dataset $DATASET \
        --architecture $ARCH | tee logs/4_apc_${attack}.log
    echo -e "${GREEN}  ✓ APC extraction for $attack complete${NC}"
done
echo ""

# ─── Step 5: Train Detector ───────────────────────────────────────────────────
run_step 5 "Training adversarial detector models" \
    "python samurai.py --train_detector" \
    "logs/5_detector.log"

# ─── Step 6: Evaluate Detector ────────────────────────────────────────────────
run_step 6 "Evaluating detector models" \
    "python samurai.py --evaluate_detector" \
    "logs/6_eval.log"

# ─── Step 7: Generate Verification Metrics ────────────────────────────────────
run_step 7 "Generating verification metrics table" \
    "python samurai.py --verification_metrics --dataset $DATASET --architecture $ARCH" \
    "logs/7_verification.log"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo -e "${GREEN}"
echo "============================================================"
echo "  SAMURAI Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Key output files:"
echo "  ├── ${DATASET,,}_${ARCH}.pth              - Trained model"
echo "  ├── all_adversarial_non_adversarial.csv   - APC features"
echo "  ├── verification_metrics_table.csv        - Results table"
echo "  ├── verification_metrics_latex.txt        - LaTeX table"
echo "  ├── feature_importance.png                - Feature plot"
echo "  └── logs/                                 - All log files"
echo "============================================================"
echo -e "${NC}"
