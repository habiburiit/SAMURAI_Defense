#############################################################################
############## Code Developed by Habibur Rahaman ###########################
######################## University of Florida #############################
######################### ECE Department ################################### 
################### Advisor Professor Swarup Bhunia ########################
####################### All Rights Reserved ################################ 
##############################################################################

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import shap
import seaborn as sns
import foolbox as fb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy, wasserstein_distance
import cv2

import xgboost as xgb
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from fvcore.nn import FlopCountAnalysis
import psutil
import pynvml
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Any
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import requests
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntroDisplay:
    """Handle the introduction display"""
    
    @staticmethod
    def show_intro():
        intro_lines = [
            "  SSSSS    AAAAA   M     M  U     U  RRRRRR   AAAAA   IIIII  ",
            " S     S  A     A  MM   MM  U     U  R     R  A     A    I    ",
            " S        AAAAAAA  M M M M  U     U  RRRRRR   AAAAAAA   I    ",
            "  SSSSS   A     A  M  M  M  U     U  R  R     A     A   I    ",
            "       S  A     A  M     M  U     U  R   R    A     A   I    ",
            " S     S  A     A  M     M  U     U  R    R   A     A   I    ",
            "  SSSSS    AAAAA   M     M   UUUUU   R     R   AAAAA   IIIII  ",
            "",
            "    Safeguarding against Malicious Usage and Resilience of AI  ",
            "",
            "          Enhanced Framework with Advanced APC Features          ",
            "",
            "       Developers: Habibur Rahaman, Atri Chatterjee, Professor. Swarup Bhunia       ",
            "",
            "             Copyrighted to University of Florida              "
        ]
        
        for line in intro_lines:
            print(line.center(80))
            time.sleep(0.3)

class SystemMonitor:
    """Handle system monitoring functionality"""
    
    @staticmethod
    def initialize_nvml():
        try:
            pynvml.nvmlInit()
            return True
        except pynvml.NVMLError as error:
            logger.error(f"Failed to initialize NVML: {error}")
            return False

    @staticmethod
    def get_gpu_utilization_and_temp():
        if torch.cuda.is_available():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    'gpu_util': utilization.gpu,
                    'memory_util': utilization.memory,
                    'temperature': temperature,
                    'memory_used': memory_info.used / (1024**3),  # GB
                    'memory_total': memory_info.total / (1024**3)  # GB
                }
            except pynvml.NVMLError as error:
                logger.error(f"Failed to get GPU metrics: {error}")
        return {
            'gpu_util': "N/A", 'memory_util': "N/A", 'temperature': "N/A",
            'memory_used': "N/A", 'memory_total': "N/A"
        }

class DatasetManager:
    """Handle dataset loading and management"""
    
    @staticmethod
    def get_datasets(name: str):
        dataset_map = {
            'CIFAR10': (torchvision.datasets.CIFAR10, 10),
            'CIFAR100': (torchvision.datasets.CIFAR100, 100),
            'MNIST': (torchvision.datasets.MNIST, 10),
            'SVHN': (torchvision.datasets.SVHN, 10),
            'ImageNet': (torchvision.datasets.ImageNet, 1000),
            'STL10': (torchvision.datasets.STL10, 10)
        }

        if name not in dataset_map:
            raise ValueError(f"Unsupported dataset: {name}")

        dataset_class, num_classes = dataset_map[name]

        if name == 'MNIST':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif name == 'STL10':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        
        if name in ('SVHN','STL10'):
            trainset = dataset_class(root='./data', split='train', download=True, transform=transform)
            testset = dataset_class(root='./data', split='test', download=True, transform=transform)
        else:
            trainset = dataset_class(root='./data', train=True, download=True, transform=transform)
            testset = dataset_class(root='./data', train=False, download=True, transform=transform)
        return trainset, testset, num_classes

class ModelManager:
    """Handle model creation and management"""
    
    @staticmethod
    def get_model(arch: str, num_classes: int):
        model_map = {
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'alexnet': models.alexnet,
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'inception_v3': models.inception_v3,
            'mobilenet_v2': models.mobilenet_v2,
            'VITA': {
                "model":AutoModelForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k"),
                "feature_extractor": AutoFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")
            },
            'VITB': {
                "model":AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224"),
                "feature_extractor": AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            }
        }

        if arch not in model_map:
            raise ValueError(f"Unsupported architecture: {arch}")
        if('VIT' not in arch):
            model = model_map[arch](pretrained=False)
        else:
            model = model_map[arch]['model']

        
        # Modify the final layer based on architecture
        if 'resnet' in arch:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'vgg' in arch or 'alexnet' in arch:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif 'densenet' in arch:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif 'inception' in arch:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'mobilenet' in arch:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif 'VIT' in arch:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        return model
    @staticmethod
    def get_feature_extractor(arch: str):
        feature_map = {
            'VITA': AutoFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k"),
            'VITB': AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        }
        return feature_map[arch]
    @staticmethod
    def initialize_model(device, dataset_name: str, architecture: str):
        _, _, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.get_model(architecture, num_classes).to(device)
        
        model_path = f'./{dataset_name.lower()}_{architecture}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file {model_path} not found. Using randomly initialized model.")
        
        model.eval()
        model.layer_outputs = []
        return model

class APCMetricsCalculator:
    """Enhanced APC metrics calculator with additional features"""
    
    @staticmethod
    def register_hooks(model):
        hooks = []
        model.layer_outputs = []
        layer_times = {}

        def pre_hook(name):
            def hook(module, input):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                layer_times[name] = time.time()
            return hook
        
        def post_hook(name):
            def hook(module, input, output):
                if(torch.cuda.is_available()):
                    torch.cuda.synchronize()
                start_time = layer_times.get(name, None)
                if start_time is not None:
                    elapsed = time.time() - start_time
                else:
                    elapsed = None
                model.layer_outputs.append({
                    'name': name,
                    'output': output,
                    'input': input[0] if isinstance(input, tuple) else input,
                    'time_taken': elapsed
                })
            return hook
        is_vit = "vit" in model.__class__.__name__.lower()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                if is_vit:
                    if name.startswith("encoder") or name.startswith("vit.encoder"):
                        continue

                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        hooks.append(module.register_forward_pre_hook(pre_hook(name)))
                        hook = module.register_forward_hook(post_hook(name))
                        hooks.append(hook)
                else:
                    hooks.append(module.register_forward_pre_hook(pre_hook(name)))
                    hook = module.register_forward_hook(post_hook(name))
                    hooks.append(hook)
        return hooks

    @staticmethod
    def calculate_sparsity(tensor):
        """Calculate sparsity percentage"""
        return 100.0 * torch.count_nonzero(tensor == 0).item() / tensor.numel()

    @staticmethod
    def calculate_tensor_memory(tensor):
        """Calculate tensor memory in MB"""
        return tensor.element_size() * tensor.nelement() / (1024 ** 2)

    @staticmethod
    def calculate_flops(model, input_tensor):
        """Calculate FLOPs"""
        try:
            flops = FlopCountAnalysis(model, input_tensor)
            return flops.total()
        except:
            return 0

    @staticmethod
    def calculate_node_activity(tensor):
        """Calculate node activity percentage"""
        active_nodes = torch.count_nonzero(tensor > 0)
        total_nodes = tensor.numel()
        return 100.0 * active_nodes.item() / total_nodes

    @staticmethod
    def calculate_layer_entropy(tensor):
        """Calculate entropy of layer activations"""
        flat_tensor = tensor.flatten().cpu().numpy()
        flat_tensor = np.abs(flat_tensor)
        if np.sum(flat_tensor) == 0:
            return 0
        normalized = flat_tensor / np.sum(flat_tensor)
        return entropy(normalized + 1e-10)

    @staticmethod
    def calculate_activation_variance(tensor):
        """Calculate variance of activations"""
        return torch.var(tensor).item()

    @staticmethod
    def calculate_activation_mean(tensor):
        """Calculate mean of activations"""
        return torch.mean(tensor).item()

    @staticmethod
    def calculate_activation_std(tensor):
        """Calculate standard deviation of activations"""
        return torch.std(tensor).item()

    @staticmethod
    def calculate_l1_norm(tensor):
        """Calculate L1 norm"""
        return torch.norm(tensor, p=1).item()

    @staticmethod
    def calculate_l2_norm(tensor):
        """Calculate L2 norm"""
        return torch.norm(tensor, p=2).item()

    @staticmethod
    def calculate_frobenius_norm(tensor):
        """Calculate Frobenius norm"""
        return torch.norm(tensor, p='fro').item()

    @staticmethod
    def calculate_spectral_norm(tensor):
        """Calculate spectral norm (largest singular value)"""
        if tensor.dim() >= 2:
            flat_tensor = tensor.view(tensor.size(0), -1)
            try:
                _, s, _ = torch.svd(flat_tensor)
                return s[0].item() if len(s) > 0 else 0
            except:
                return 0
        return 0

    @staticmethod
    def calculate_rank(tensor):
        """Calculate effective rank of tensor"""
        if tensor.dim() >= 2:
            flat_tensor = tensor.view(tensor.size(0), -1)
            try:
                _, s, _ = torch.svd(flat_tensor)
                normalized_s = s / s[0] if s[0] > 0 else s
                return torch.sum(normalized_s > 1e-6).item()
            except:
                return 0
        return 0

    @staticmethod
    def calculate_kurtosis(tensor):
        """Calculate kurtosis of activations"""
        flat_tensor = tensor.flatten().cpu().numpy()
        mean = np.mean(flat_tensor)
        std = np.std(flat_tensor)
        if std == 0:
            return 0
        normalized = (flat_tensor - mean) / std
        return np.mean(normalized ** 4) - 3

    @staticmethod
    def calculate_skewness(tensor):
        """Calculate skewness of activations"""
        flat_tensor = tensor.flatten().cpu().numpy()
        mean = np.mean(flat_tensor)
        std = np.std(flat_tensor)
        if std == 0:
            return 0
        normalized = (flat_tensor - mean) / std
        return np.mean(normalized ** 3)

    @staticmethod
    def calculate_activation_patterns(tensor):
        """Calculate activation pattern metrics"""
        positive_ratio = torch.sum(tensor > 0).item() / tensor.numel()
        negative_ratio = torch.sum(tensor < 0).item() / tensor.numel()
        zero_ratio = torch.sum(tensor == 0).item() / tensor.numel()
        
        return {
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'zero_ratio': zero_ratio
        }

class ImageAnalyzer:
    """Handle image analysis including SSIM calculations"""
    
    @staticmethod
    def calculate_ssim(img1_path: str, img2_path: str):
        """Calculate SSIM between two images"""
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
            
            if img1 is None or img2 is None:
                return 0
            
            # Resize images to same size if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Calculate SSIM
            ssim_value = ssim(img1, img2, multichannel=True, win_size=3, data_range=255)
            return ssim_value
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return 0

    @staticmethod
    def calculate_perturbation_metrics(clean_img_path: str, adv_img_path: str):
        """Calculate various perturbation metrics"""
        try:
            clean_img = cv2.imread(clean_img_path, cv2.IMREAD_COLOR)
            adv_img = cv2.imread(adv_img_path, cv2.IMREAD_COLOR)
            
            if clean_img is None or adv_img is None:
                return {}
            
            if clean_img.shape != adv_img.shape:
                adv_img = cv2.resize(adv_img, (clean_img.shape[1], clean_img.shape[0]))
            
            # Convert to float for calculations
            clean_img = clean_img.astype(np.float32) / 255.0
            adv_img = adv_img.astype(np.float32) / 255.0
            
            # Calculate metrics
            mse = np.mean((clean_img - adv_img) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # L2 distance
            l2_distance = np.linalg.norm(clean_img - adv_img)
            
            # Linf distance
            linf_distance = np.max(np.abs(clean_img - adv_img))
            
            return {
                'mse': mse,
                'psnr': psnr,
                'l2_distance': l2_distance,
                'linf_distance': linf_distance
            }
        except Exception as e:
            logger.error(f"Error calculating perturbation metrics: {e}")
            return {}
'''
class AdversarialAttacker:
    """Handle various adversarial attacks"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    def list_available_attacks(self):
        """List all available attacks in current foolbox version"""
        available_attacks = []
        try:
            # Test each attack to see if it's available
            test_attacks = [
                ("fgsm", "fb.attacks.FGSM()"),
                ("pgd", "fb.attacks.PGD()"),
                ("deepfool", "fb.attacks.LinfDeepFoolAttack()"),
                ("cw", "fb.attacks.L2CarliniWagnerAttack()"),
                ("bim", "fb.attacks.L2BasicIterativeAttack()"),
                ("jsma", "fb.attacks.SaliencyMapAttack()"),
                ("jsma_alt", "fb.attacks.JSMA()"),
                ("lbfgs", "fb.attacks.L2LBFGSAttack()"),
                ("lbfgs_alt", "fb.attacks.LBFGSAttack()")
            ]
            
            for name, attack_str in test_attacks:
                try:
                    eval(attack_str)
                    available_attacks.append(name)
                except:
                    pass
                    
            logger.info(f"Available attacks: {available_attacks}")
            return available_attacks
            
        except Exception as e:
            logger.error(f"Error checking available attacks: {e}")
            return ["fgsm", "pgd"]  # Basic fallback

    def get_attack(self, attack_type: str):
        """Get attack instance based on attack type"""
        try:
            attack_map = {
                "fgsm": fb.attacks.FGSM(),
                "pgd": fb.attacks.PGD(steps=40, abs_stepsize=0.01),
                "deepfool": fb.attacks.LinfDeepFoolAttack(steps=50),
                "cw": fb.attacks.L2CarliniWagnerAttack(steps=100),
                "bim": fb.attacks.L2BasicIterativeAttack(steps=40),
            }
            
            # Try different names for JSMA based on foolbox version
            try:
                attack_map["jsma"] = fb.attacks.SaliencyMapAttack(steps=100)
            except AttributeError:
                try:
                    attack_map["jsma"] = fb.attacks.JSMA(steps=100)
                except AttributeError:
                    logger.warning("JSMA attack not available in this foolbox version")
                    attack_map["jsma"] = fb.attacks.FGSM()  # Fallback to FGSM
            
            # Try different names for L-BFGS
            try:
                attack_map["lbfgs"] = fb.attacks.L2LBFGSAttack(steps=100)
            except AttributeError:
                try:
                    attack_map["lbfgs"] = fb.attacks.LBFGSAttack(steps=100)
                except AttributeError:
                    logger.warning("L-BFGS attack not available in this foolbox version")
                    attack_map["lbfgs"] = fb.attacks.PGD(steps=40)  # Fallback to PGD
                    
        except Exception as e:
            logger.error(f"Error setting up attacks: {e}")
            # Minimal fallback attack map
            attack_map = {
                "fgsm": fb.attacks.FGSM(),
                "pgd": fb.attacks.PGD(steps=40),
                "deepfool": fb.attacks.LinfDeepFoolAttack(steps=50),
                "cw": fb.attacks.L2CarliniWagnerAttack(steps=100),
                "bim": fb.attacks.L2BasicIterativeAttack(steps=40),
                "jsma": fb.attacks.FGSM(),  # Fallback
                "lbfgs": fb.attacks.PGD(steps=40)  # Fallback
            }
        
        if attack_type not in attack_map:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        return attack_map[attack_type]

    def perform_attack(self, dataset_name: str, architecture: str, attack_type: str, 
                      num_samples: int = 100):
        """Perform adversarial attack"""
        _, testset, num_classes = DatasetManager.get_datasets(dataset_name)
        
        # Simple transform for attack
        transform = transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor()
        ])
        testset.transform = transform
        testloader = DataLoader(testset, batch_size=4, shuffle=False)

        epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
        save_dir = f'./{dataset_name.lower()}_{architecture}_{attack_type}_images'
        clean_dir = f'./{dataset_name.lower()}_{architecture}_clean_images'
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

        attack = self.get_attack(attack_type)
        logger.info(f"Performing {attack_type} attack...")

        successful_attacks = 0
        total_samples = 0
        
        for batch_idx, (cln_data, true_label) in enumerate(testloader):
            if total_samples >= num_samples:
                break
                
            cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
            
            try:
                # Save clean images
                for i in range(cln_data.size(0)):
                    if total_samples + i >= num_samples:
                        break
                    clean_image = transforms.ToPILImage()(cln_data[i].cpu())
                    clean_path = os.path.join(clean_dir, 
                        f'clean_image_{total_samples + i}_label_{true_label[i].item()}.png')
                    clean_image.save(clean_path)
                
                # Perform attack
                _, adv_examples, success = attack(self.fmodel, cln_data, true_label, 
                                                epsilons=epsilons)
                
                # Save adversarial images
                for eps_idx, epsilon in enumerate(epsilons):
                    if adv_examples[eps_idx] is not None:
                        for i in range(adv_examples[eps_idx].size(0)):
                            if total_samples + i >= num_samples:
                                break
                            adv_image = transforms.ToPILImage()(adv_examples[eps_idx][i].cpu())
                            adv_path = os.path.join(save_dir, 
                                f'{attack_type}_image_{total_samples + i}_eps_{epsilon}_label_{true_label[i].item()}.png')
                            adv_image.save(adv_path)
                            
                            if success[eps_idx][i]:
                                successful_attacks += 1
                
                total_samples += cln_data.size(0)
                
            except Exception as e:
                logger.error(f"Attack failed for batch {batch_idx}: {str(e)}")
                continue

        # Calculate Attack Success Rate (ASR)
        asr = (successful_attacks / (total_samples * len(epsilons))) * 100
        logger.info(f"Attack Success Rate (ASR): {asr:.2f}%")
        
        logger.info(f"Clean images saved to {clean_dir}")
        logger.info(f"Adversarial images saved to {save_dir}")
        
        return asr
'''
class AdversarialAttacker:
    """Enhanced adversarial attacker with robust attack implementations"""
    '''
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        
        # Test available attacks on initialization
        self.available_attacks = self._test_available_attacks()
        logger.info(f"Available attacks: {list(self.available_attacks.keys())}")
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Add normalization preprocessing so foolbox sees
        # the same normalized inputs the model was trained on
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        preprocessing = dict(mean=mean, std=std, axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

        # Test available attacks on initialization
        self.available_attacks = self._test_available_attacks()
        logger.info(f"Available attacks: {list(self.available_attacks.keys())}")
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Foolbox requires 1D tensors for mean/std preprocessing
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        preprocessing = dict(mean=mean, std=std, axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

        # Test available attacks on initialization
        self.available_attacks = self._test_available_attacks()
        logger.info(f"Available attacks: {list(self.available_attacks.keys())}")
    '''
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Foolbox requires 1D tensors on same device as model
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std  = torch.tensor([0.229, 0.224, 0.225]).to(device)
        preprocessing = dict(mean=mean, std=std, axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

        # Test available attacks on initialization
        self.available_attacks = self._test_available_attacks()
        logger.info(f"Available attacks: {list(self.available_attacks.keys())}")
    
    def _test_available_attacks(self):
        """Test which attacks are available in the current Foolbox version"""
        available_attacks = {}
        
        # Test each attack individually
        attack_tests = [
            # Basic attacks that should always work
            ("fgsm", lambda: fb.attacks.FGSM()),
            ("pgd", lambda: fb.attacks.PGD()),
            
            # L2 attacks
            ("pgd_l2", lambda: fb.attacks.L2PGD()),
            ("cw", lambda: fb.attacks.L2CarliniWagnerAttack()),
            ("deepfool", lambda: fb.attacks.L2DeepFoolAttack()),
            
            # Linf attacks
            ("pgd_linf", lambda: fb.attacks.LinfPGD()),
            ("deepfool_linf", lambda: fb.attacks.LinfDeepFoolAttack()),
            ("bim", lambda: fb.attacks.LinfBasicIterativeAttack()),
            
            # JSMA variants (try different names)
            ("jsma", lambda: fb.attacks.SaliencyMapAttack()),
            ("jsma_alt", lambda: fb.attacks.JSMA()),
            
            # L-BFGS variants
            ("l
