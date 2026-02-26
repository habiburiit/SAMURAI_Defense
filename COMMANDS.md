# SAMURAI - Complete Command Reference

## 🔧 Setup
```bash
bash setup.sh                  # Full automated setup
conda activate samurai_env     # Activate environment
python verify.py               # Verify installation
```

---

## 🏋️ Training
```bash
# CIFAR10
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset CIFAR10 --architecture resnet18 --epochs 50
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset CIFAR10 --architecture resnet50 --epochs 50
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset CIFAR10 --architecture vgg16    --epochs 50

# CIFAR100
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset CIFAR100 --architecture resnet18 --epochs 50
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset CIFAR100 --architecture resnet50 --epochs 50

# MNIST
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset MNIST --architecture resnet18 --epochs 20

# STL10
CUDA_VISIBLE_DEVICES=0 python samurai.py --train --dataset STL10 --architecture resnet18 --epochs 50
```

---

## ⚔️ Attacks
```bash
# FGSM
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack fgsm     --dataset CIFAR10 --architecture resnet18 --num_samples 100
# PGD
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack pgd      --dataset CIFAR10 --architecture resnet18 --num_samples 100
# DeepFool
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack deepfool --dataset CIFAR10 --architecture resnet18 --num_samples 100
# C&W
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack cw       --dataset CIFAR10 --architecture resnet18 --num_samples 100
# BIM
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack bim      --dataset CIFAR10 --architecture resnet18 --num_samples 100
# PGD L2
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack pgd_l2   --dataset CIFAR10 --architecture resnet18 --num_samples 100
# DeepFool Linf
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack deepfool_linf --dataset CIFAR10 --architecture resnet18 --num_samples 100
# Boundary
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack boundary --dataset CIFAR10 --architecture resnet18 --num_samples 100

# All attacks at once
for attack in fgsm pgd deepfool cw bim; do
    CUDA_VISIBLE_DEVICES=0 python samurai.py --attack $attack --dataset CIFAR10 --architecture resnet18 --num_samples 100 | tee attack_${attack}.log
done
```

---

## 📊 APC Extraction
```bash
CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack fgsm     --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack pgd      --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack deepfool --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack cw       --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack bim      --dataset CIFAR10 --architecture resnet18

# All APC extractions at once
for attack in fgsm pgd deepfool cw bim; do
    CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack $attack --dataset CIFAR10 --architecture resnet18 | tee apc_${attack}.log
done
```

---

## 🤖 Detector
```bash
CUDA_VISIBLE_DEVICES=0 python samurai.py --train_detector                    # Train all detectors
CUDA_VISIBLE_DEVICES=0 python samurai.py --train_detector --explain_shap     # Train + SHAP plots
CUDA_VISIBLE_DEVICES=0 python samurai.py --evaluate_detector                 # Evaluate detectors
CUDA_VISIBLE_DEVICES=0 python samurai.py --analyze_features                  # Feature impact analysis
```

---

## 📈 Analysis
```bash
# Verification metrics table (for paper)
CUDA_VISIBLE_DEVICES=0 python samurai.py --verification_metrics --dataset CIFAR10 --architecture resnet18

# APC divergence analysis
CUDA_VISIBLE_DEVICES=0 python samurai.py --analyze_divergence --dataset CIFAR10 --architecture resnet18 --attack pgd

# Cross-dataset divergence
CUDA_VISIBLE_DEVICES=0 python samurai.py --extract_divergence --dataset1 CIFAR10 --dataset2 CIFAR100 --architecture resnet18

# Class-wise divergence
CUDA_VISIBLE_DEVICES=0 python samurai.py --extract_class_divergence --dataset CIFAR10 --architecture resnet18
```

---

## 🔍 Debugging
```bash
python samurai.py --attack_guide                                              # Full usage guide
CUDA_VISIBLE_DEVICES=0 python samurai.py --list_attacks     --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python samurai.py --test_attacks     --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python samurai.py --benchmark_attacks --dataset CIFAR10 --architecture resnet18 --num_samples 20
CUDA_VISIBLE_DEVICES=0 python samurai.py --debug_attack pgd --dataset CIFAR10 --architecture resnet18
CUDA_VISIBLE_DEVICES=0 python check_accuracy.py --dataset CIFAR10 --architecture resnet18
```

---

## 🚀 Full Pipeline (one shot)
```bash
bash run_pipeline.sh                                    # default: GPU=1, CIFAR10, resnet18, 50 epochs
bash run_pipeline.sh --gpu 0 --epochs 100               # custom epochs
bash run_pipeline.sh --gpu 2 --dataset CIFAR100 --arch resnet50
```
