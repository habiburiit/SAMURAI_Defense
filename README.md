# SAMURAI 🗡️
## SAMURAI: Runtime Attack Detection in AI Accelerators Using AI Performance Counters (APC)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg"/>
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-orange.svg"/>
  <img src="https://img.shields.io/badge/CUDA-12.x-green.svg"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
  <img src="https://img.shields.io/badge/University-Florida-blue.svg"/>
</p>

```
  SSSSS    AAAAA   M     M  U     U  RRRRRR   AAAAA   IIIII
 S     S  A     A  MM   MM  U     U  R     R  A     A    I
 S        AAAAAAA  M M M M  U     U  RRRRRR   AAAAAAA   I
  SSSSS   A     A  M  M  M  U     U  R  R     A     A   I
       S  A     A  M     M  U     U  R   R    A     A   I
 S     S  A     A  M     M  U     U  R    R   A     A   I
  SSSSS    AAAAA   M     M   UUUUU   R     R   AAAAA   IIIII
```

> **Developers:** Habibur Rahaman, Atri Chatterjee, Professor Swarup Bhunia  
> **Institution:** University of Florida, ECE Department  
> **Advisor:** Professor Swarup Bhunia



<p align="center">
  <img src="SAMURAI_for_Defense.png" alt="SAMURAI Framework Diagram" width="950"/>
</p>

---

## 📌 Overview

SAMURAI is a comprehensive adversarial attack detection framework leveraging **AI Performance Counters (APC)** to detect adversarial examples in deep neural networks. It monitors hardware-level and software-level behavioral signals across neural network layers to distinguish clean inputs from adversarially perturbed ones.

SAMURAI achieves **~98% detection accuracy** by capturing anomalous patterns across four categories of performance counters that adversarial examples inevitably trigger.

---

## 🔬 What Are AI Performance Counters (APCs)?

SAMURAI monitors **4 categories** of performance counters during neural network inference:

### 1️⃣ Activation-Based Counters
These measure how neurons respond to inputs at each layer.

| Counter | Description |
|---|---|
| `node_activity` | Percentage of neurons with positive activations |
| `activation_mean` | Average activation value per layer |
| `activation_std` | Standard deviation of activations |
| `activation_variance` | Variance of activation distribution |
| `positive_ratio` | Fraction of positive activations |
| `negative_ratio` | Fraction of negative activations |
| `zero_ratio` | Fraction of zero activations (dead neurons) |

### 2️⃣ Switching Activity-Based Counters
These capture how activation patterns change and distribute across layers.

| Counter | Description |
|---|---|
| `sparsity` | Percentage of zero-valued activations |
| `layer_entropy` | Information entropy of activation distribution |
| `kurtosis` | Tailedness of activation distribution |
| `skewness` | Asymmetry of activation distribution |

### 3️⃣ Computational / Norm-Based Counters
These track mathematical properties of layer computations.

| Counter | Description |
|---|---|
| `l1_norm` | L1 norm of layer activations |
| `l2_norm` | L2 norm of layer activations |
| `frobenius_norm` | Frobenius norm of activation tensor |
| `spectral_norm` | Largest singular value of activation matrix |
| `rank` | Effective rank of activation tensor |
| `tensor_memory` | Memory footprint of activations (MB) |
| `flops` | Floating point operations count |
| `inference_time` | Per-layer execution time (seconds) |

### 4️⃣ System-Level Hardware Counters
These monitor GPU/CPU behavior during inference.

| Counter | Description |
|---|---|
| `GPU_Util_Before/After` | GPU utilization percentage |
| `GPU_Memory_Before/After` | GPU memory usage (GB) |
| `GPU_Temp_Before/After` | GPU temperature (°C) |
| `CPU_Usage_Before/After` | CPU utilization percentage |
| `Memory_Usage_Before/After` | System RAM utilization |
| `Inference_Time` | Total inference latency |
| `Throughput` | Images processed per second |

### 5️⃣ Inference Output Counters
These capture model output behavior.

| Counter | Description |
|---|---|
| `Confidence` | Maximum softmax probability |
| `Loss` | Cross-entropy loss value |
| `Output_Entropy` | Entropy of output probability distribution |
| `Predicted_Label` | Model's predicted class |
| `SSIM` | Structural similarity to clean image |
| `MSE` | Mean squared perturbation error |
| `PSNR` | Peak signal-to-noise ratio |
| `L2_Distance` | L2 distance from clean image |
| `Linf_Distance` | L-infinity distance from clean image |

---

## 🏗️ Architecture

```
SAMURAI/
├── samurai.py                  # Main framework script
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated environment setup
├── verify.py                   # Environment verification script
├── check_accuracy.py           # Model accuracy checker
├── README.md                   # This file
├── data/                       # Downloaded datasets (auto-created)
├── *.pth                       # Trained model weights (auto-created)
├── *_clean_images/             # Clean image directories (auto-created)
├── *_*_images/                 # Adversarial image directories (auto-created)
├── all_adversarial_non_adversarial.csv   # APC features dataset
├── verification_metrics_table.csv        # Final results table
├── verification_metrics_latex.txt        # LaTeX table for papers
├── feature_importance.png               # Feature importance plot
└── *.pkl / *.h5                         # Trained detector models
```

---

## ⚙️ Supported Configurations

### Datasets
| Dataset | Classes | Description |
|---|---|---|
| `CIFAR10` | 10 | 32x32 color images |
| `CIFAR100` | 100 | 32x32 color images, fine-grained |
| `MNIST` | 10 | 28x28 grayscale digits |
| `SVHN` | 10 | Street view house numbers |
| `STL10` | 10 | 96x96 color images |

### Model Architectures
| Architecture | Description |
|---|---|
| `resnet18/34/50/101/152` | Residual Networks |
| `vgg16/19` | VGG Networks |
| `alexnet` | AlexNet |
| `densenet121/169` | DenseNet |
| `inception_v3` | Inception v3 |
| `mobilenet_v2` | MobileNet v2 |
| `VITA` | ViT Large (google/vit-large-patch16-224-in21k) |
| `VITB` | ViT Base (google/vit-base-patch16-224) |

### Adversarial Attacks
| Attack | Norm | Description |
|---|---|---|
| `fgsm` | L∞ | Fast Gradient Sign Method |
| `pgd` | L∞ | Projected Gradient Descent |
| `pgd_l2` | L2 | L2 PGD |
| `pgd_linf` | L∞ | L-inf PGD |
| `deepfool` | L2 | DeepFool minimal perturbation |
| `deepfool_linf` | L∞ | L-inf DeepFool |
| `cw` | L2 | Carlini & Wagner |
| `bim` | L∞ | Basic Iterative Method |
| `boundary` | L2 | Boundary Attack |

### Detector Models
| Model | Type |
|---|---|
| XGBoost | Gradient Boosting |
| Random Forest | Ensemble |
| SVM | Support Vector Machine |
| Logistic Regression | Linear |
| DNN | Deep Neural Network |
| LSTM | Recurrent Neural Network |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/SAMURAI.git
cd SAMURAI
```

### 2. Setup Environment
```bash
chmod +x setup.sh
bash setup.sh
conda activate samurai_env
```

### 3. Verify Installation
```bash
python verify.py
```

### 4. Run Full Pipeline
```bash
# On GPU 1 (adjust CUDA_VISIBLE_DEVICES as needed)
CUDA_VISIBLE_DEVICES=1 python samurai.py --train --dataset CIFAR10 --architecture resnet18 --epochs 50

CUDA_VISIBLE_DEVICES=1 python samurai.py --attack pgd --dataset CIFAR10 --architecture resnet18 --num_samples 100

CUDA_VISIBLE_DEVICES=1 python samurai.py --apc --attack pgd --dataset CIFAR10 --architecture resnet18

CUDA_VISIBLE_DEVICES=1 python samurai.py --train_detector

CUDA_VISIBLE_DEVICES=1 python samurai.py --evaluate_detector

CUDA_VISIBLE_DEVICES=1 python samurai.py --verification_metrics --dataset CIFAR10 --architecture resnet18
```

---

## 📦 Installation

### Prerequisites
- Linux (Ubuntu 20.04+)
- NVIDIA GPU with CUDA 11.x or 12.x
- Miniconda or Anaconda
- Python 3.10

### Manual Installation
```bash
# Create environment
conda create -n samurai_env python=3.10 -y
conda activate samurai_env

# Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install TensorFlow
pip install tensorflow==2.15.0

# Fix NumPy compatibility
pip install "numpy==1.26.4"

# Install all dependencies
pip install -r requirements.txt
```

---

## 💻 Complete Command Reference

### Training Commands
```bash
# Train ResNet18 on CIFAR10 for 10 epochs
CUDA_VISIBLE_DEVICES=0 python samurai.py --train \
    --dataset CIFAR10 \
    --architecture resnet18 \
    --epochs 10

# Train ResNet50 on CIFAR100 for 50 epochs
CUDA_VISIBLE_DEVICES=0 python samurai.py --train \
    --dataset CIFAR100 \
    --architecture resnet50 \
    --epochs 50

# Train VGG16 on MNIST
CUDA_VISIBLE_DEVICES=0 python samurai.py --train \
    --dataset MNIST \
    --architecture vgg16 \
    --epochs 20
```

### Attack Commands
```bash
# FGSM Attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack fgsm \
    --dataset CIFAR10 --architecture resnet18 --num_samples 100

# PGD Attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack pgd \
    --dataset CIFAR10 --architecture resnet18 --num_samples 100

# DeepFool Attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack deepfool \
    --dataset CIFAR10 --architecture resnet18 --num_samples 100

# Carlini & Wagner Attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack cw \
    --dataset CIFAR10 --architecture resnet18 --num_samples 100

# BIM Attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack bim \
    --dataset CIFAR10 --architecture resnet18 --num_samples 100

# Boundary Attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --attack boundary \
    --dataset CIFAR10 --architecture resnet18 --num_samples 100
```

### APC Extraction Commands
```bash
# Extract APC metrics for PGD attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack pgd \
    --dataset CIFAR10 --architecture resnet18

# Extract APC metrics for all attacks (run sequentially)
for attack in fgsm pgd deepfool cw bim; do
    CUDA_VISIBLE_DEVICES=0 python samurai.py --apc --attack $attack \
        --dataset CIFAR10 --architecture resnet18
done
```

### Detector Training & Evaluation Commands
```bash
# Train all detector models
CUDA_VISIBLE_DEVICES=0 python samurai.py --train_detector

# Train with SHAP explanations
CUDA_VISIBLE_DEVICES=0 python samurai.py --train_detector --explain_shap

# Evaluate trained detectors
CUDA_VISIBLE_DEVICES=0 python samurai.py --evaluate_detector

# Analyze feature importance
CUDA_VISIBLE_DEVICES=0 python samurai.py --analyze_features
```

### Verification & Analysis Commands
```bash
# Generate verification metrics table (for paper)
CUDA_VISIBLE_DEVICES=0 python samurai.py --verification_metrics \
    --dataset CIFAR10 --architecture resnet18

# Analyze APC divergence between clean and adversarial
CUDA_VISIBLE_DEVICES=0 python samurai.py --analyze_divergence \
    --dataset CIFAR10 --architecture resnet18 --attack pgd

# Extract cross-dataset divergence metrics
CUDA_VISIBLE_DEVICES=0 python samurai.py --extract_divergence \
    --dataset1 CIFAR10 --dataset2 CIFAR100 --architecture resnet18

# Extract class-wise divergence metrics
CUDA_VISIBLE_DEVICES=0 python samurai.py --extract_class_divergence \
    --dataset CIFAR10 --architecture resnet18
```

### Attack Testing & Debugging Commands
```bash
# List all available attacks
CUDA_VISIBLE_DEVICES=0 python samurai.py --list_attacks \
    --dataset CIFAR10 --architecture resnet18

# Test all available attacks
CUDA_VISIBLE_DEVICES=0 python samurai.py --test_attacks \
    --dataset CIFAR10 --architecture resnet18

# Benchmark all attacks
CUDA_VISIBLE_DEVICES=0 python samurai.py --benchmark_attacks \
    --dataset CIFAR10 --architecture resnet18 --num_samples 20

# Debug specific attack
CUDA_VISIBLE_DEVICES=0 python samurai.py --debug_attack pgd \
    --dataset CIFAR10 --architecture resnet18

# Show complete usage guide
python samurai.py --attack_guide
```

---

## 🔄 Full Research Pipeline

### Complete One-Command Pipeline
```bash
#!/bin/bash
DATASET="CIFAR10"
ARCH="resnet18"
GPU="1"
SAMPLES=100
EPOCHS=50

# Step 1: Train
CUDA_VISIBLE_DEVICES=$GPU python samurai.py --train \
    --dataset $DATASET --architecture $ARCH --epochs $EPOCHS | tee 1_train.log

# Step 2: Run all attacks
for attack in fgsm pgd deepfool cw bim; do
    CUDA_VISIBLE_DEVICES=$GPU python samurai.py --attack $attack \
        --dataset $DATASET --architecture $ARCH \
        --num_samples $SAMPLES | tee attack_${attack}.log
done

# Step 3: Extract APC metrics for all attacks
for attack in fgsm pgd deepfool cw bim; do
    CUDA_VISIBLE_DEVICES=$GPU python samurai.py --apc --attack $attack \
        --dataset $DATASET --architecture $ARCH | tee apc_${attack}.log
done

# Step 4: Train detector
CUDA_VISIBLE_DEVICES=$GPU python samurai.py --train_detector | tee 4_detector.log

# Step 5: Evaluate detector
CUDA_VISIBLE_DEVICES=$GPU python samurai.py --evaluate_detector | tee 5_eval.log

# Step 6: Generate verification table
CUDA_VISIBLE_DEVICES=$GPU python samurai.py --verification_metrics \
    --dataset $DATASET --architecture $ARCH | tee 6_verification.log

echo "Pipeline complete! Check verification_metrics_latex.txt for paper results."
```

---

## 📊 Expected Results (CIFAR10 + ResNet18)

| Attack | ASR (%) | Detection Acc (%) | SSIM |
|---|---|---|---|
| Clean | 0.00 | — | 1.000 |
| FGSM | ~75-85 | ~95-98 | ~0.85 |
| PGD | ~78-85 | ~96-98 | ~0.82 |
| DeepFool | ~80-90 | ~96-99 | ~0.88 |
| C&W | ~78-88 | ~95-98 | ~0.80 |
| BIM | ~78-85 | ~95-98 | ~0.83 |

---

## 📁 Output Files Reference

| File | Created By | Description |
|---|---|---|
| `cifar10_resnet18.pth` | `--train` | Trained model weights |
| `cifar10_resnet18_clean_images/` | `--attack` | Clean test images |
| `cifar10_resnet18_pgd_images/` | `--attack pgd` | PGD adversarial images |
| `all_adversarial_non_adversarial.csv` | `--apc` | APC feature dataset |
| `xgboost_model.pkl` | `--train_detector` | XGBoost detector |
| `random_forest_model.pkl` | `--train_detector` | RF detector |
| `svm_model.pkl` | `--train_detector` | SVM detector |
| `dnn_model.h5` | `--train_detector` | DNN detector |
| `lstm_model.h5` | `--train_detector` | LSTM detector |
| `scaler.pkl` | `--train_detector` | Feature scaler |
| `feature_selector.pkl` | `--train_detector` | Feature selector |
| `feature_importance.csv` | `--train_detector` | Feature rankings |
| `feature_importance_plot.png` | `--train_detector` | Feature importance plot |
| `shap_summary_xgboost.png` | `--explain_shap` | SHAP summary plot |
| `verification_metrics_table.csv` | `--verification_metrics` | Results table |
| `verification_metrics_latex.txt` | `--verification_metrics` | LaTeX table |
| `apc_divergence_heatmap.png` | `--analyze_divergence` | Divergence heatmap |

---

## 🖥️ Multi-GPU Usage

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python samurai.py --train ...

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python samurai.py --train ...

# Use GPUs 1,2,3 (avoid GPU 0 if busy)
CUDA_VISIBLE_DEVICES=1,2,3 python samurai.py --train ...

# Check GPU status before running
nvidia-smi
```

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: skimage` | `pip install scikit-image==0.21.0 --no-deps` |
| `numpy incompatible` | `pip install "numpy==1.26.4"` |
| `AutoFeatureExtractor error` | `pip install "transformers==4.40.0"` |
| `opencv numpy>=2 conflict` | `pip install "opencv-python==4.8.0.76"` |
| `TF-TRT Warning` | Harmless — TensorRT not needed |
| `cuDNN factory warning` | Harmless — TF internal deduplication |
| Low ASR | Check normalization in `AdversarialAttacker.__init__` |
| `No correctly classified samples` | Add normalization to model prediction check |

---

## 📝 Citation

If you use SAMURAI in your research, please cite:

```bibtex
@inproceedings{rahaman2024samurai,
  title={Samurai: A framework for safeguarding against malicious usage and resilience of ai},
  author={Rahaman, Habibur and Chatterjee, Atri and Bhunia, Swarup},
  booktitle={2024 IEEE 33rd Asian Test Symposium (ATS)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
```
Paper Also got accepted in will be online within few days: 
IEEE Transactions on Circuits and Systems for Artificial Intelligence (Journal) 2026.
Authors: Habibur Rahaman, Atri Chatterjee and Professor. Swarup Bhunia
```
---

## 📄 License

Copyright © University of Florida. All Rights Reserved.

---

## 👨‍💻 Developers

| Name | Role |
|---|---|
| **Habibur Rahaman** | Lead Developer, PhD Student, Email Id: rahaman.habibur@ufl.edu | 
| **Atri Chatterjee** | Co-Developer, PhD Student, Email Id: a.chatterjee@ufl.edu |  
| **Prof. Swarup Bhunia** | Advisor, Email Id: swarup@ece.ulf.edu|

**Department:** Electrical and Computer Engineering  
**University:** University of Florida
