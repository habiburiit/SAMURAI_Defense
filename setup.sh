#!/bin/bash
# =============================================================================
# SAMURAI Framework - Automated Environment Setup Script
# University of Florida, ECE Department
# Developers: Habibur Rahaman, Atri Chatterjee, Prof. Swarup Bhunia
# =============================================================================

set -e  # Exit on any error

# ─── Colors for output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Banner ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}"
echo "  SSSSS    AAAAA   M     M  U     U  RRRRRR   AAAAA   IIIII"
echo " S     S  A     A  MM   MM  U     U  R     R  A     A    I  "
echo " S        AAAAAAA  M M M M  U     U  RRRRRR   AAAAAAA   I  "
echo "  SSSSS   A     A  M  M  M  U     U  R  R     A     A   I  "
echo "       S  A     A  M     M  U     U  R   R    A     A   I  "
echo " S     S  A     A  M     M  U     U  R    R   A     A   I  "
echo "  SSSSS    AAAAA   M     M   UUUUU   R     R   AAAAA   IIIII"
echo ""
echo "         Safeguarding against Malicious Usage and Resilience of AI"
echo "                     University of Florida Setup Script"
echo -e "${NC}"

# ─── Step 1: Check Prerequisites ──────────────────────────────────────────────
echo -e "${BLUE}[1/7] Checking prerequisites...${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda not found. Please install Miniconda first.${NC}"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}  ✓ conda found${NC}"

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}  WARNING: nvidia-smi not found. CPU-only mode will be used.${NC}"
    CUDA_VERSION="cpu"
else
    echo -e "${GREEN}  ✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}  ✓ CUDA Version: $CUDA_VERSION${NC}"
fi

# ─── Step 2: Create Conda Environment ─────────────────────────────────────────
echo -e "${BLUE}[2/7] Creating conda environment 'samurai_env'...${NC}"

# Remove existing environment if it exists
if conda env list | grep -q "samurai_env"; then
    echo -e "${YELLOW}  Environment 'samurai_env' already exists. Removing...${NC}"
    conda env remove -n samurai_env -y
fi

conda create -n samurai_env python=3.10 -y
echo -e "${GREEN}  ✓ Environment created${NC}"

# ─── Step 3: Activate Environment ─────────────────────────────────────────────
echo -e "${BLUE}[3/7] Activating environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate samurai_env
echo -e "${GREEN}  ✓ Environment activated${NC}"

# ─── Step 4: Install PyTorch ───────────────────────────────────────────────────
echo -e "${BLUE}[4/7] Installing PyTorch with CUDA support...${NC}"

# Determine CUDA version for PyTorch install
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

if [ "$CUDA_VERSION" == "cpu" ]; then
    pip install torch torchvision torchaudio
elif [ "$CUDA_MAJOR" -ge "12" ]; then
    echo -e "  Installing PyTorch for CUDA 12.x..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_MAJOR" -eq "11" ]; then
    echo -e "  Installing PyTorch for CUDA 11.x..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}  Unknown CUDA version, installing default PyTorch...${NC}"
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
fi

echo -e "${GREEN}  ✓ PyTorch installed${NC}"

# ─── Step 5: Install TensorFlow ────────────────────────────────────────────────
echo -e "${BLUE}[5/7] Installing TensorFlow...${NC}"
pip install tensorflow==2.15.0
echo -e "${GREEN}  ✓ TensorFlow installed${NC}"

# ─── Step 6: Pin NumPy and Install All Dependencies ───────────────────────────
echo -e "${BLUE}[6/7] Installing all dependencies with pinned NumPy...${NC}"

# Pin NumPy FIRST to avoid conflicts
pip install "numpy==1.26.4"

# Install packages with compatibility constraints
pip install \
    "transformers==4.40.0" \
    "scikit-image==0.21.0" \
    "opencv-python==4.8.0.76" \
    "scikit-learn==1.3.2" \
    "ml_dtypes==0.2.0" \
    --no-deps 2>/dev/null || true

# Install remaining packages
pip install \
    foolbox==3.3.4 \
    shap \
    seaborn \
    xgboost \
    fvcore \
    psutil \
    nvidia-ml-py \
    scipy \
    Pillow \
    pandas \
    matplotlib \
    joblib \
    requests \
    tqdm \
    huggingface-hub \
    safetensors \
    tokenizers

# Final NumPy pin (some packages may try to upgrade it)
pip install "numpy==1.26.4" --force-reinstall

echo -e "${GREEN}  ✓ All dependencies installed${NC}"

# ─── Step 7: Verify Installation ──────────────────────────────────────────────
echo -e "${BLUE}[7/7] Verifying installation...${NC}"

python - <<'PYEOF'
import sys

errors = []
warnings = []

# Check PyTorch
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        warnings.append("CUDA not available - CPU mode only")
except ImportError as e:
    errors.append(f"PyTorch: {e}")

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow: {tf.__version__}")
except ImportError as e:
    errors.append(f"TensorFlow: {e}")

# Check NumPy
try:
    import numpy as np
    print(f"  ✓ NumPy: {np.__version__}")
    if np.__version__.startswith('2'):
        errors.append("NumPy 2.x detected - TensorFlow incompatible. Run: pip install 'numpy==1.26.4'")
except ImportError as e:
    errors.append(f"NumPy: {e}")

# Check Foolbox
try:
    import foolbox
    print(f"  ✓ Foolbox: {foolbox.__version__}")
except ImportError as e:
    errors.append(f"Foolbox: {e}")

# Check Transformers
try:
    import transformers
    print(f"  ✓ Transformers: {transformers.__version__}")
except ImportError as e:
    errors.append(f"Transformers: {e}")

# Check SHAP
try:
    import shap
    print(f"  ✓ SHAP: {shap.__version__}")
except ImportError as e:
    errors.append(f"SHAP: {e}")

# Check scikit-image
try:
    import skimage
    print(f"  ✓ scikit-image: {skimage.__version__}")
except ImportError as e:
    errors.append(f"scikit-image: {e}")

# Check OpenCV
try:
    import cv2
    print(f"  ✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    errors.append(f"OpenCV: {e}")

# Check pynvml
try:
    import pynvml
    print(f"  ✓ nvidia-ml-py: installed")
except ImportError as e:
    warnings.append(f"nvidia-ml-py: {e}")

print()
if warnings:
    for w in warnings:
        print(f"  ⚠ WARNING: {w}")

if errors:
    print()
    for e in errors:
        print(f"  ✗ ERROR: {e}")
    sys.exit(1)
else:
    print("  ✅ All checks passed! SAMURAI is ready.")
PYEOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}"
    echo "============================================================"
    echo "  SAMURAI Environment Setup Complete!"
    echo "============================================================"
    echo ""
    echo "  Activate with:  conda activate samurai_env"
    echo ""
    echo "  Quick start:"
    echo "  CUDA_VISIBLE_DEVICES=0 python samurai.py --train \\"
    echo "      --dataset CIFAR10 --architecture resnet18 --epochs 50"
    echo "============================================================"
    echo -e "${NC}"
else
    echo -e "${RED}"
    echo "============================================================"
    echo "  Setup completed with errors. Check messages above."
    echo "============================================================"
    echo -e "${NC}"
    exit 1
fi
