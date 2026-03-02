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
            ("lbfgs", lambda: fb.attacks.LBFGSAttack()),
            ("lbfgs_l2", lambda: fb.attacks.L2LBFGSAttack()),
            
            # Additional attacks
            ("boundary", lambda: fb.attacks.BoundaryAttack()),
            ("nes", lambda: fb.attacks.NESAttack()),
            ("spsa", lambda: fb.attacks.SPSAAttack()),
        ]
        
        for name, attack_func in attack_tests:
            try:
                attack = attack_func()
                available_attacks[name] = attack_func
                logger.debug(f"✓ {name} attack is available")
            except (AttributeError, ImportError, TypeError) as e:
                logger.debug(f"✗ {name} attack not available: {e}")
                
        return available_attacks

    def list_available_attacks(self):
        """List all available attacks"""
        return list(self.available_attacks.keys())

    def get_attack(self, attack_type: str, **kwargs):
        """Get attack instance with enhanced parameters and fallback handling"""
        if attack_type not in self.available_attacks:
            # Try to find similar attack
            similar_attacks = [k for k in self.available_attacks.keys() if attack_type.lower() in k.lower()]
            if similar_attacks:
                attack_type = similar_attacks[0]
                logger.info(f"Using {attack_type} instead")
            else:
                available = ", ".join(self.available_attacks.keys())
                raise ValueError(f"Attack '{attack_type}' not available. Available attacks: {available}")
        
        try:
            return self._get_enhanced_attack(attack_type, **kwargs)
        except Exception as e:
            logger.warning(f"Enhanced configuration failed for {attack_type}: {e}")
            logger.info(f"Falling back to basic configuration for {attack_type}")
            return self._get_basic_attack(attack_type)

    def _get_enhanced_attack(self, attack_type: str, **kwargs):
        """Get attack with enhanced configurations"""
        
        if attack_type == "fgsm":
            return fb.attacks.FGSM()
        
        elif attack_type == "pgd":
            return fb.attacks.PGD(
                steps=kwargs.get('steps', 40),
                abs_stepsize=kwargs.get('stepsize', 0.01),
                random_start=kwargs.get('random_start', True)
            )
        
        elif attack_type == "pgd_l2":
            return fb.attacks.L2PGD(
                steps=kwargs.get('steps', 40),
                abs_stepsize=kwargs.get('stepsize', 0.1),
                random_start=kwargs.get('random_start', True)
            )
        
        elif attack_type == "pgd_linf":
            return fb.attacks.LinfPGD(
                steps=kwargs.get('steps', 40),
                abs_stepsize=kwargs.get('stepsize', 0.01),
                random_start=kwargs.get('random_start', True)
            )
        
        elif attack_type == "deepfool":
            return fb.attacks.L2DeepFoolAttack(
                steps=kwargs.get('steps', 50),
                overshoot=kwargs.get('overshoot', 0.02)
            )
        
        elif attack_type == "deepfool_linf":
            return fb.attacks.LinfDeepFoolAttack(
                steps=kwargs.get('steps', 50)
            )
        
        elif attack_type == "cw":
            try:
                return fb.attacks.L2CarliniWagnerAttack(
                    steps=kwargs.get('steps', 100),
                    stepsize=kwargs.get('stepsize', 0.01),
                    confidence=kwargs.get('confidence', 0)
                )
            except TypeError as e:
                logger.warning(f"Some C&W parameters not supported: {e}")
                # Fallback to minimal parameters
                return fb.attacks.L2CarliniWagnerAttack()
        
        elif attack_type == "bim":
            return fb.attacks.LinfBasicIterativeAttack(
                steps=kwargs.get('steps', 40),
                abs_stepsize=kwargs.get('stepsize', 0.01)
            )
        
        elif attack_type in ["jsma", "jsma_alt"]:
            if attack_type == "jsma":
                try:
                    return fb.attacks.SaliencyMapAttack(
                        steps=kwargs.get('steps', 1000),
                        max_perturbations_per_pixel=kwargs.get('max_pert', 256)
                    )
                except TypeError:
                    return fb.attacks.SaliencyMapAttack()
            else:
                try:
                    return fb.attacks.JSMA(steps=kwargs.get('steps', 1000))
                except TypeError:
                    return fb.attacks.JSMA()
        
        elif attack_type in ["lbfgs", "lbfgs_l2"]:
            if attack_type == "lbfgs":
                try:
                    return fb.attacks.LBFGSAttack(
                        steps=kwargs.get('steps', 100),
                        lr=kwargs.get('lr', 1e-2)
                    )
                except TypeError:
                    return fb.attacks.LBFGSAttack()
            else:
                try:
                    return fb.attacks.L2LBFGSAttack(steps=kwargs.get('steps', 100))
                except TypeError:
                    return fb.attacks.L2LBFGSAttack()
        
        elif attack_type == "boundary":
            try:
                return fb.attacks.BoundaryAttack(
                    steps=kwargs.get('steps', 1000),
                    spherical_step=kwargs.get('spherical_step', 0.01),
                    source_step=kwargs.get('source_step', 0.01)
                )
            except TypeError:
                return fb.attacks.BoundaryAttack()
        
        elif attack_type == "nes":
            try:
                return fb.attacks.NESAttack(steps=kwargs.get('steps', 1000))
            except TypeError:
                return fb.attacks.NESAttack()
        
        elif attack_type == "spsa":
            try:
                return fb.attacks.SPSAAttack(steps=kwargs.get('steps', 1000))
            except TypeError:
                return fb.attacks.SPSAAttack()
        
        else:
            # Fallback to stored function
            return self.available_attacks[attack_type]()

    def _get_basic_attack(self, attack_type: str):
        """Get attack with basic/minimal configuration"""
        
        basic_configs = {
            "fgsm": lambda: fb.attacks.FGSM(),
            "pgd": lambda: fb.attacks.PGD(),
            "pgd_l2": lambda: fb.attacks.L2PGD(),
            "pgd_linf": lambda: fb.attacks.LinfPGD(),
            "deepfool": lambda: fb.attacks.L2DeepFoolAttack(),
            "deepfool_linf": lambda: fb.attacks.LinfDeepFoolAttack(),
            "cw": lambda: fb.attacks.L2CarliniWagnerAttack(),  # Basic C&W without parameters
            "bim": lambda: fb.attacks.LinfBasicIterativeAttack(),
            "jsma": lambda: fb.attacks.SaliencyMapAttack() if hasattr(fb.attacks, 'SaliencyMapAttack') else fb.attacks.JSMA(),
            "jsma_alt": lambda: fb.attacks.JSMA() if hasattr(fb.attacks, 'JSMA') else fb.attacks.SaliencyMapAttack(),
            "lbfgs": lambda: fb.attacks.LBFGSAttack(),
            "lbfgs_l2": lambda: fb.attacks.L2LBFGSAttack(),
            "boundary": lambda: fb.attacks.BoundaryAttack(),
            "nes": lambda: fb.attacks.NESAttack(),
            "spsa": lambda: fb.attacks.SPSAAttack(),
        }
        
        if attack_type in basic_configs:
            try:
                return basic_configs[attack_type]()
            except Exception as e:
                logger.error(f"Even basic configuration failed for {attack_type}: {e}")
                # Ultimate fallback to stored function
                return self.available_attacks[attack_type]()
        else:
            return self.available_attacks[attack_type]()

    def perform_attack(self, dataset_name: str, architecture: str, attack_type: str, 
                      num_samples: int = 100, epsilons: list = None):
        """Enhanced attack performance with better error handling"""
        
        if attack_type not in self.available_attacks:
            logger.error(f"Attack {attack_type} not available!")
            return 0.0
        
        _, testset, num_classes = DatasetManager.get_datasets(dataset_name)
        
        # Simple transform for attack
        transform = transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor()
        ])
        testset.transform = transform
        testloader = DataLoader(testset, batch_size=4, shuffle=False)

        # Default epsilons based on attack type
        if epsilons is None:
            if attack_type in ['cw', 'deepfool', 'lbfgs', 'lbfgs_l2']:
                # L2 attacks - use larger epsilons
                epsilons = [1.0, 2.0, 3.0, 4.0, 5.0]
            elif attack_type in ['jsma', 'jsma_alt']:
                # JSMA uses different perturbation bounds
                epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                # Linf attacks
                epsilons = [0.01, 0.03, 0.05, 0.1, 0.3]
        
        save_dir = f'./{dataset_name.lower()}_{architecture}_{attack_type}_images'
        clean_dir = f'./{dataset_name.lower()}_{architecture}_clean_images'
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

        # Get attack with enhanced parameters based on attack type
        attack_params = self._get_attack_specific_params(attack_type)
        attack = self.get_attack(attack_type, **attack_params)
        
        logger.info(f"Performing {attack_type} attack with epsilons: {epsilons}")

        successful_attacks = 0
        total_samples = 0
        failed_batches = 0
        
        for batch_idx, (cln_data, true_label) in enumerate(testloader):
            if total_samples >= num_samples:
                break
                
            cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
            
            try:
                # Save clean images first
                for i in range(cln_data.size(0)):
                    if total_samples + i >= num_samples:
                        break
                    clean_image = transforms.ToPILImage()(cln_data[i].cpu())
                    clean_path = os.path.join(clean_dir, 
                        f'clean_image_{total_samples + i}_label_{true_label[i].item()}.png')
                    clean_image.save(clean_path)
                
                
                # Verify model predictions on clean data
                # Must normalize before passing to model
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                with torch.no_grad():
                    clean_predictions = self.model(normalize(cln_data))
                    clean_pred_labels = torch.argmax(clean_predictions, dim=1)
                '''
                # Verify model predictions on clean data
                with torch.no_grad():
                    clean_predictions = self.model(cln_data)
                    clean_pred_labels = torch.argmax(clean_predictions, dim=1)
                '''
                # Filter out misclassified clean samples
                correctly_classified = (clean_pred_labels == true_label)
                if not correctly_classified.any():
                    logger.warning(f"No correctly classified samples in batch {batch_idx}")
                    total_samples += cln_data.size(0)
                    continue
                
                # Use only correctly classified samples
                clean_data_filtered = cln_data[correctly_classified]
                true_label_filtered = true_label[correctly_classified]
                
                # Perform attack with enhanced error handling
                success_count = self._perform_single_attack(
                    attack, clean_data_filtered, true_label_filtered, epsilons,
                    save_dir, total_samples, attack_type
                )
                
                successful_attacks += success_count
                total_samples += cln_data.size(0)
                
                if batch_idx % 10 == 0:
                    current_asr = (successful_attacks / (total_samples * len(epsilons))) * 100 if total_samples > 0 else 0
                    logger.info(f"Batch {batch_idx}: Current ASR = {current_asr:.2f}%")
                
            except Exception as e:
                failed_batches += 1
                logger.error(f"Attack failed for batch {batch_idx}: {str(e)}")
                if failed_batches > 5:
                    logger.error("Too many failed batches, trying with simpler parameters...")
                    # Try with simpler attack parameters
                    attack = self._get_basic_attack(attack_type)
                    failed_batches = 0
                
                total_samples += cln_data.size(0)
                continue

        # Calculate final ASR
        if total_samples > 0:
            asr = (successful_attacks / (total_samples * len(epsilons))) * 100
        else:
            asr = 0.0
            
        logger.info(f"Final Attack Success Rate (ASR): {asr:.2f}%")
        logger.info(f"Total successful attacks: {successful_attacks}")
        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Clean images saved to {clean_dir}")
        logger.info(f"Adversarial images saved to {save_dir}")
        
        return asr

    def _get_attack_specific_params(self, attack_type: str):
        """Get attack-specific parameters for better performance"""
        params = {}
        
        if attack_type == 'pgd':
            params = {'steps': 40, 'stepsize': 0.01, 'random_start': True}
        elif attack_type == 'deepfool':
            params = {'steps': 50, 'overshoot': 0.02}
        elif attack_type == 'cw':
            # FIXED: Only use parameters that are actually supported
            params = {'steps': 100, 'stepsize': 0.01, 'confidence': 0}
        elif attack_type in ['jsma', 'jsma_alt']:
            params = {'steps': 1000}  # Removed unsupported max_pert parameter
        elif attack_type in ['lbfgs', 'lbfgs_l2']:
            params = {'steps': 100}
        elif attack_type == 'bim':
            params = {'steps': 40, 'stepsize': 0.01}
        
        return params

    def _perform_single_attack(self, attack, clean_data, true_labels, epsilons, 
                              save_dir, sample_offset, attack_type):
        """Perform attack on a single batch with enhanced error handling"""
        successful_attacks = 0
        
        try:
            # For some attacks, we need to handle them differently
            if attack_type in ['jsma', 'jsma_alt']:
                # JSMA typically works with single samples
                for i, (sample, label) in enumerate(zip(clean_data, true_labels)):
                    sample = sample.unsqueeze(0)
                    label = label.unsqueeze(0)
                    
                    try:
                        _, adv_examples, success = attack(self.fmodel, sample, label, epsilons=epsilons)
                        
                        for eps_idx, epsilon in enumerate(epsilons):
                            if adv_examples[eps_idx] is not None and len(adv_examples[eps_idx]) > 0:
                                adv_image = transforms.ToPILImage()(adv_examples[eps_idx][0].cpu())
                                adv_path = os.path.join(save_dir, 
                                    f'{attack_type}_image_{sample_offset + i}_eps_{epsilon}_label_{label.item()}.png')
                                adv_image.save(adv_path)
                                
                                if success[eps_idx][0]:
                                    successful_attacks += 1
                    except Exception as e:
                        logger.debug(f"JSMA failed for sample {i}: {e}")
                        continue
            
            else:
                # Standard attack procedure for other attacks
                _, adv_examples, success = attack(self.fmodel, clean_data, true_labels, epsilons=epsilons)
                
                # Save adversarial images
                for eps_idx, epsilon in enumerate(epsilons):
                    if adv_examples[eps_idx] is not None:
                        for i in range(adv_examples[eps_idx].size(0)):
                            try:
                                adv_image = transforms.ToPILImage()(adv_examples[eps_idx][i].cpu().clamp(0, 1))
                                adv_path = os.path.join(save_dir, 
                                    f'{attack_type}_image_{sample_offset + i}_eps_{epsilon}_label_{true_labels[i].item()}.png')
                                adv_image.save(adv_path)
                                
                                if success[eps_idx][i]:
                                    successful_attacks += 1
                            except Exception as e:
                                logger.debug(f"Error saving adversarial image {i} for eps {epsilon}: {e}")
                                continue
                                
        except Exception as e:
            logger.error(f"Attack execution failed: {e}")
            # Try fallback attack if main attack fails
            try:
                logger.info("Trying fallback FGSM attack...")
                fallback_attack = fb.attacks.FGSM()
                _, adv_examples, success = fallback_attack(self.fmodel, clean_data, true_labels, epsilons=epsilons)
                
                for eps_idx, epsilon in enumerate(epsilons):
                    if adv_examples[eps_idx] is not None:
                        for i in range(adv_examples[eps_idx].size(0)):
                            try:
                                adv_image = transforms.ToPILImage()(adv_examples[eps_idx][i].cpu().clamp(0, 1))
                                adv_path = os.path.join(save_dir, 
                                    f'fallback_fgsm_image_{sample_offset + i}_eps_{epsilon}_label_{true_labels[i].item()}.png')
                                adv_image.save(adv_path)
                                
                                if success[eps_idx][i]:
                                    successful_attacks += 1
                            except:
                                continue
                                
            except Exception as fallback_error:
                logger.error(f"Even fallback attack failed: {fallback_error}")
        
        return successful_attacks

    def test_attack(self, attack_type: str, dataset_name: str, architecture: str, num_test_samples: int = 5):
        """Test if an attack works with a small number of samples"""
        logger.info(f"Testing {attack_type} attack...")
        
        try:
            # Load a small batch of data
            _, testset, _ = DatasetManager.get_datasets(dataset_name)
            transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            testset.transform = transform
            testloader = DataLoader(testset, batch_size=2, shuffle=False)
            
            # Get first batch
            cln_data, true_label = next(iter(testloader))
            cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
            
            # Test attack with basic configuration
            attack = self.get_attack(attack_type)
            _, adv_examples, success = attack(self.fmodel, cln_data, true_label, epsilons=[0.1])
            
            if adv_examples[0] is not None:
                logger.info(f" {attack_type} attack test successful!")
                return True
            else:
                logger.warning(f" {attack_type} attack test failed - no adversarial examples generated")
                return False
                
        except Exception as e:
            logger.error(f"✗ {attack_type} attack test failed: {e}")
            return False

    def get_attack_info(self, attack_type: str):
        """Get information about a specific attack"""
        attack_info = {
            "fgsm": {
                "name": "Fast Gradient Sign Method",
                "norm": "L∞",
                "description": "Single-step gradient-based attack"
            },
            "pgd": {
                "name": "Projected Gradient Descent",
                "norm": "L∞",
                "description": "Multi-step iterative gradient-based attack"
            },
            "pgd_l2": {
                "name": "L2 Projected Gradient Descent",
                "norm": "L2",
                "description": "Multi-step L2 gradient-based attack"
            },
            "deepfool": {
                "name": "DeepFool",
                "norm": "L2",
                "description": "Minimal perturbation attack using linear approximation"
            },
            "deepfool_linf": {
                "name": "DeepFool L∞",
                "norm": "L∞",
                "description": "L∞ version of DeepFool attack"
            },
            "cw": {
                "name": "Carlini & Wagner",
                "norm": "L2",
                "description": "Optimization-based attack with high transferability"
            },
            "jsma": {
                "name": "Jacobian-based Saliency Map Attack",
                "norm": "L0",
                "description": "Greedy pixel-wise perturbation attack"
            },
            "lbfgs": {
                "name": "L-BFGS Attack",
                "norm": "L2",
                "description": "Optimization-based attack using L-BFGS"
            },
            "bim": {
                "name": "Basic Iterative Method",
                "norm": "L∞",
                "description": "Iterative version of FGSM"
            }
        }
        
        return attack_info.get(attack_type, {"name": attack_type, "description": "Unknown attack"})



class EnhancedAPCProcessor:
    """Enhanced APC processor with comprehensive metrics"""
    
    def __init__(self, model, device, architecture):
        self.model = model
        self.device = device
        self.metrics_calculator = APCMetricsCalculator()
        self.image_analyzer = ImageAnalyzer()
        self.system_monitor = SystemMonitor()
        self.is_vit = "vit" in model.__class__.__name__.lower()
        if self.is_vit:
            self.feature_extractor = ModelManager.get_feature_extractor(architecture)
            logger.info(f"Using ViT feature extractor for architecture: {architecture}")
        else:
            self.feature_extractor = None

    def process_image(self, img_path: str, input_tensor: torch.Tensor, 
                     true_label: int, label: str, clean_img_path: str = None):
        """Extract comprehensive APC metrics for a single image"""
        metrics = {}
        hooks = self.metrics_calculator.register_hooks(self.model)

        try:
            with torch.no_grad():
                self.model.layer_outputs = []

                # System monitoring before inference
                cpu_before = psutil.cpu_percent()
                memory_before = psutil.virtual_memory().percent
                gpu_metrics_before = self.system_monitor.get_gpu_utilization_and_temp()
                
                if self.is_vit:
                    image = Image.open(img_path).convert("RGB")
                    processed = self.feature_extractor(images=image, return_tensors="pt")
                    input_tensor = processed['pixel_values'].to(self.device)
                start_time = time.time()
                output = self.model(input_tensor)
                inference_time = time.time() - start_time
                
                # System monitoring after inference
                cpu_after = psutil.cpu_percent()
                memory_after = psutil.virtual_memory().percent
                gpu_metrics_after = self.system_monitor.get_gpu_utilization_and_temp()
                if(self.is_vit and hasattr(output,"logits")):
                    output = output.logits
                # Basic inference metrics
                predicted_label = torch.argmax(output, 1).item()
                confidence = torch.max(nn.functional.softmax(output, dim=1)).item()
                loss = nn.CrossEntropyLoss()(output, torch.tensor([true_label], device=self.device)).item()
                output_entropy = -torch.sum(nn.functional.softmax(output, dim=1) * 
                                          torch.log(nn.functional.softmax(output, dim=1) + 1e-10)).item()
                throughput = 1 / inference_time

                # Layer-wise metrics calculation
                layer_metrics = self._calculate_layer_metrics()
                
                # Image-specific metrics
                image_metrics = {}
                if clean_img_path and label == "Adversarial":
                    ssim_value = self.image_analyzer.calculate_ssim(clean_img_path, img_path)
                    perturbation_metrics = self.image_analyzer.calculate_perturbation_metrics(
                        clean_img_path, img_path)
                    image_metrics.update({
                        'ssim': ssim_value,
                        **perturbation_metrics
                    })

                # Compile all metrics
                metrics.update({
                    "Image": os.path.basename(img_path),
                    "Adversarial_or_Non_Adversarial": label,
                    "Predicted_Label": predicted_label,
                    "True_Label": true_label,
                    "Confidence": confidence,
                    "Loss": loss,
                    "Output_Entropy": output_entropy,
                    "Throughput": throughput,
                    "Inference_Time": inference_time,
                    "CPU_Usage_Before": cpu_before,
                    "CPU_Usage_After": cpu_after,
                    "Memory_Usage_Before": memory_before,
                    "Memory_Usage_After": memory_after,
                    "GPU_Util_Before": gpu_metrics_before['gpu_util'],
                    "GPU_Util_After": gpu_metrics_after['gpu_util'],
                    "GPU_Memory_Before": gpu_metrics_before['memory_used'],
                    "GPU_Memory_After": gpu_metrics_after['memory_used'],
                    "GPU_Temp_Before": gpu_metrics_before['temperature'],
                    "GPU_Temp_After": gpu_metrics_after['temperature'],
                    **layer_metrics,
                    **image_metrics
                })

        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return metrics

    def _calculate_layer_metrics(self):
        """Calculate comprehensive layer-wise metrics"""
        layer_metrics = {}
        aggregated_metrics = defaultdict(list)
        
        for i, layer_data in enumerate(self.model.layer_outputs):
            if isinstance(layer_data, dict):
                output = layer_data['output']
                layer_name = f"layer_{i}"
                layer_type = layer_data.get('type', None)
                inference_time = layer_data.get('time_taken',None)
            else:
                output = layer_data
                layer_name = f"layer_{i}"
                layer_type = None
                inference_time = None
            
            if output is None:
                continue
            if self.is_vit and layer_type not in [torch.nn.Conv2d, torch.nn.Linear]:
                continue
            
            # Basic metrics
            sparsity = self.metrics_calculator.calculate_sparsity(output)
            activity = self.metrics_calculator.calculate_node_activity(output)
            memory = self.metrics_calculator.calculate_tensor_memory(output)
            
            # Statistical metrics
            variance = self.metrics_calculator.calculate_activation_variance(output)
            mean_act = self.metrics_calculator.calculate_activation_mean(output)
            std_act = self.metrics_calculator.calculate_activation_std(output)
            entropy_act = self.metrics_calculator.calculate_layer_entropy(output)
            
            # Norm metrics
            l1_norm = self.metrics_calculator.calculate_l1_norm(output)
            l2_norm = self.metrics_calculator.calculate_l2_norm(output)
            frobenius_norm = self.metrics_calculator.calculate_frobenius_norm(output)
            spectral_norm = self.metrics_calculator.calculate_spectral_norm(output)
            
            # Shape metrics
            rank = self.metrics_calculator.calculate_rank(output)
            kurtosis = self.metrics_calculator.calculate_kurtosis(output)
            skewness = self.metrics_calculator.calculate_skewness(output)
            
            # Activation patterns
            patterns = self.metrics_calculator.calculate_activation_patterns(output)
            
            # Store individual layer metrics
            metrics_dict = {
                f"{layer_name}_sparsity": sparsity,
                f"{layer_name}_activity": activity,
                f"{layer_name}_memory": memory,
                f"{layer_name}_variance": variance,
                f"{layer_name}_mean": mean_act,
                f"{layer_name}_std": std_act,
                f"{layer_name}_entropy": entropy_act,
                f"{layer_name}_l1_norm": l1_norm,
                f"{layer_name}_l2_norm": l2_norm,
                f"{layer_name}_frobenius_norm": frobenius_norm,
                f"{layer_name}_spectral_norm": spectral_norm,
                f"{layer_name}_rank": rank,
                f"{layer_name}_kurtosis": kurtosis,
                f"{layer_name}_skewness": skewness,
                f"{layer_name}_positive_ratio": patterns['positive_ratio'],
                f"{layer_name}_negative_ratio": patterns['negative_ratio'],
                f"{layer_name}_zero_ratio": patterns['zero_ratio'],
                f"{layer_name}_inference_time": inference_time
            }
            
            layer_metrics.update(metrics_dict)
            
            # Aggregate for summary statistics
            for key, value in metrics_dict.items():
                metric_type = key.split('_', 2)[-1]  # Get the metric type
                aggregated_metrics[metric_type].append(value)
        
        # Add aggregated statistics
        for metric_type, values in aggregated_metrics.items():
            if values:
                layer_metrics.update({
                    f"avg_{metric_type}": np.mean(values),
                    f"std_{metric_type}": np.std(values),
                    f"min_{metric_type}": np.min(values),
                    f"max_{metric_type}": np.max(values),
                    f"median_{metric_type}": np.median(values)
                })
        
        return layer_metrics

    def calculate_apc_divergence(self, clean_metrics: Dict, adv_metrics: Dict):
        """Calculate APC divergence between clean and adversarial images"""
        divergences = {}
        
        # Get common keys (excluding metadata)
        excluded_keys = ['Image', 'Adversarial_or_Non_Adversarial', 'Predicted_Label', 'True_Label']
        common_keys = set(clean_metrics.keys()) & set(adv_metrics.keys())
        common_keys = [k for k in common_keys if k not in excluded_keys]
        for key in common_keys:
            clean_val = clean_metrics.get(key, 0)
            adv_val = adv_metrics.get(key, 0)
            
            if isinstance(clean_val, (int, float)) and isinstance(adv_val, (int, float)):
                # Absolute difference
                abs_diff = abs(adv_val - clean_val)
                divergences[f'{key}_abs_diff'] = abs_diff
                
                # Relative difference (percentage change)
                if clean_val != 0:
                    rel_diff = abs_diff / abs(clean_val) * 100
                else:
                    rel_diff = abs_diff * 100
                divergences[f'{key}_rel_diff'] = rel_diff
                
                # Store original values for reference
                divergences[f'{key}_100CIFAR'] = clean_val
                divergences[f'{key}_10STL'] = adv_val
        
        # Calculate SSIM if available
        if 'ssim' in adv_metrics:
            divergences['SSIM'] = adv_metrics['ssim']
        
        return divergences

class FrameworkManager:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_monitor = SystemMonitor()
        logger.info(f"Using device: {self.device}")
        
    def train_model(self, dataset_name: str, architecture: str, epochs: int = 10):
        """Train a model"""
        logger.info(f"Training {architecture} on {dataset_name}")
        
        trainset, testset, num_classes = DatasetManager.get_datasets(dataset_name)
        print(len(trainset), len(testset))
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

        model = ModelManager.get_model(architecture, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        is_vit = "VIT" in architecture.lower()
        if is_vit:
            feature_extractor = ModelManager.get_feature_extractor(architecture)  # Assuming this returns a HF feature extractor

        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0], data[1]
                if is_vit:
                    # Convert to list of PIL images if not already
                    images = [Image.fromarray(img.numpy().transpose(1,2,0)) for img in inputs]
                    inputs = feature_extractor(images=images, return_tensors="pt")["pixel_values"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if(dataset_name == 'STL10'):
                    if i % 50 == 49:
                        logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                        running_loss = 0.0
                else:
                    if i % 200 == 199:
                        logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                        running_loss = 0.0

        logger.info('Finished Training')
        model_save_path = f'./{dataset_name.lower()}_{architecture}.pth'
        torch.save(model.state_dict(), model_save_path)
        logger.info(f'Model saved to {model_save_path}')

    def perform_attack(self, dataset_name: str, architecture: str, attack_type: str):
        """Perform adversarial attack"""
        _, _, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.get_model(architecture, num_classes).to(self.device)
        
        model_path = f'./{dataset_name.lower()}_{architecture}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning(f"Model file {model_path} not found. Using randomly initialized model.")
        
        model.eval()
        
        attacker = AdversarialAttacker(model, self.device)
        asr = attacker.perform_attack(dataset_name, architecture, attack_type)
        
        return asr

    def extract_class_divergence_metrics(self, dataset_name: str, architecture: str, samples_per_class: int = 50):
        """
        Extract APC Divergence Metrics between different classes within a single dataset.
        """
        logger.info("Extracting Class-wise Divergence Metrics...")
        _, testset, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.initialize_model(self.device, dataset_name, architecture)
        apc_processor = EnhancedAPCProcessor(model, self.device, architecture)
        class_indices = {i: [] for i in range(num_classes)}
    
        logger.info("Mapping test set samples to their respective classes...")
        # Iterate through the testset to map indices to labels
        for idx, (_, label_tensor) in enumerate(testset):
            label = int(label_tensor)
            if label in class_indices:
                class_indices[label].append(idx)
        min_samples = samples_per_class
    
        for label in range(num_classes):
            min_samples = min(min_samples, len(class_indices[label]))
        if min_samples == 0:
            logger.error("No samples found for at least one class. Cannot calculate class-wise divergence.")
            return []
        actual_samples_per_class = min_samples
        logger.info(f"Targeting {samples_per_class} samples per class, but only {actual_samples_per_class} are available equally among all classes. Using {actual_samples_per_class} for each class.")

        base_clean_dir = f"./{dataset_name.lower()}_{architecture}_clean_images_by_class"
        if not os.path.exists(base_clean_dir):
            logger.info(f"Creating exactly {actual_samples_per_class} clean images for each of the {min(10,num_classes)} classes...")
            os.makedirs(base_clean_dir, exist_ok=True)
            for label in range(min(10,num_classes)):
                selected_indices = class_indices[label][:actual_samples_per_class]
                for i, data_index in enumerate(selected_indices):
                    img, _ = testset[data_index]  
                    img_path = os.path.join(base_clean_dir, f"clean_image_idx_{i}_class_{label}.png")
                    transforms.ToPILImage()(img).save(img_path)
        logger.info(f"Processing clean images for {dataset_name}...")
        clean_metrics = self._process_directory(apc_processor, base_clean_dir, "Non Adversarial")
        csv_name = 'class_divergence_metrics_' + dataset_name + '_' + architecture + '.csv'
        self._save_metrics_to_csv(clean_metrics, csv_name)
        logger.info(f"APC metrics saved to {csv_name}")
        return clean_metrics

    def extract_divergence_metrics(self, dataset1_name: str, dataset2_name: str, architecture: str):
        """Extract APC Metrics"""
        logger.info("Extracting Divergence Metrics...")

        _, _, num_classes = DatasetManager.get_datasets(dataset1_name)
        model = ModelManager.initialize_model(self.device, dataset1_name, architecture)

        clean_dir = f"./{dataset1_name.lower()}_{architecture}_clean_images"
        if not os.path.exists(clean_dir):
            logger.info(f"Creating clean images for {dataset1_name}...")
            os.makedirs(clean_dir, exist_ok=True)
            _, testset, _ = DatasetManager.get_datasets(dataset1_name)
            
            for idx, (img, label) in enumerate(testset):
                if idx >= 100:  # Limit to 100 images
                    break
                img_path = os.path.join(clean_dir, f"clean_image_{idx}_label_{label}.png")
                transforms.ToPILImage()(img).save(img_path)
        apc_processor = EnhancedAPCProcessor(model, self.device, architecture)
        
        # Process clean images
        logger.info(f"Processing clean images for {dataset1_name}...")
        clean_metrics_dataset1 = self._process_directory(apc_processor, clean_dir, "Non Adversarial")
        #self._save_metrics_to_csv(clean_metrics_dataset1,'CIFAR_10_clean_metrics.csv')
        _, _, num_classes = DatasetManager.get_datasets(dataset2_name)
        model = ModelManager.initialize_model(self.device, dataset2_name, architecture)

        clean_dir = f"./{dataset2_name.lower()}_{architecture}_clean_images"
        if not os.path.exists(clean_dir):
            logger.info(f"Creating clean images for {dataset2_name}...")
            os.makedirs(clean_dir, exist_ok=True)
            _, testset, _ = DatasetManager.get_datasets(dataset2_name)
            
            for idx, (img, label) in enumerate(testset):
                if idx >= 100:  # Limit to 100 images
                    break
                img_path = os.path.join(clean_dir, f"clean_image_{idx}_label_{label}.png")
                transforms.ToPILImage()(img).save(img_path)
        apc_processor = EnhancedAPCProcessor(model, self.device, architecture)
        
        # Process clean images
        logger.info(f"Processing clean images for {dataset2_name}...")
        clean_metrics_dataset2 = self._process_directory(apc_processor, clean_dir, "Non Adversarial")
        #self._save_metrics_to_csv(clean_metrics_dataset2,'CIFAR_100_clean_metrics.csv')
        logger.info("Calculating APC divergences for both datasets...")
        divergences = self._calculate_divergences(clean_metrics_dataset1, clean_metrics_dataset2, apc_processor)
        csv_name = 'divergence_metrics_' + dataset1_name + '_' + dataset2_name + '_' + architecture + '.csv'
        self._save_metrics_to_csv(divergences, csv_name)
        logger.info(f"APC metrics saved to {csv_name}")
        
        return divergences


    def extract_apc_metrics(self, dataset_name: str, architecture: str, attack_type: str):
        """Extract APC metrics"""
        logger.info("Extracting APC metrics...")
        
        _, _, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.initialize_model(self.device, dataset_name, architecture)
        
        # Directories
        clean_dir = f"./{dataset_name.lower()}_{architecture}_clean_images"
        adv_dir = f"./{dataset_name.lower()}_{architecture}_{attack_type}_images"
        
        # Create clean images if they don't exist
        if not os.path.exists(clean_dir):
            logger.info("Creating clean images...")
            os.makedirs(clean_dir, exist_ok=True)
            _, testset, _ = DatasetManager.get_datasets(dataset_name)
            
            for idx, (img, label) in enumerate(testset):
                if idx >= 100:  # Limit to 100 images
                    break
                img_path = os.path.join(clean_dir, f"clean_image_{idx}_label_{label}.png")
                transforms.ToPILImage()(img).save(img_path)
            
        
        # Process images
        apc_processor = EnhancedAPCProcessor(model, self.device, architecture)
        
        # Process clean images
        logger.info("Processing clean images...")
        clean_metrics = self._process_directory(apc_processor, clean_dir, "Non Adversarial")
        
        # Process adversarial images
        if os.path.exists(adv_dir):
            logger.info("Processing adversarial images...")
            adv_metrics = self._process_directory(apc_processor, adv_dir, "Adversarial", clean_dir)
            
            # Calculate divergences
            logger.info("Calculating APC divergences...")
            divergences = self._calculate_divergences(clean_metrics, adv_metrics, apc_processor)
            
            # Combine all metrics
            all_metrics = clean_metrics + adv_metrics + divergences
        else:
            logger.warning(f"Adversarial directory {adv_dir} not found.")
            all_metrics = clean_metrics
        
        # Save metrics
        self._save_metrics_to_csv(all_metrics, 'all_adversarial_non_adversarial.csv')
        logger.info(f"APC metrics saved to all_adversarial_non_adversarial.csv")
        
        return all_metrics

    def _process_directory(self, apc_processor: EnhancedAPCProcessor, directory: str, 
                          label: str, clean_dir: str = None):
        """Process all images in a directory"""
        metrics = []
        
        for img_file in sorted(os.listdir(directory)):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(directory, img_file)
            
            try:
                # Load and transform image
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                # Extract label from filename
                try:
                    true_label = int(img_file.split('_')[-1].split('.')[0])
                except ValueError:
                    logger.warning(f"Could not extract label from {img_file}, skipping")
                    continue
                
                # Find corresponding clean image if processing adversarial
                clean_img_path = None
                if label == "Adversarial" and clean_dir:
                    # Try to find matching clean image
                    img_idx = img_file.split('_')[2] if len(img_file.split('_')) > 2 else img_file.split('_')[1]
                    clean_pattern = f"clean_image_{img_idx}_label_{true_label}.png"
                    clean_img_path = os.path.join(clean_dir, clean_pattern)
                    if not os.path.exists(clean_img_path):
                        clean_img_path = None
                
                # Process image
                metric = apc_processor.process_image(
                    img_path, input_tensor, true_label, label, clean_img_path
                )
                metrics.append(metric)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        return metrics

    def _calculate_divergences(self, clean_metrics: List[Dict], adv_metrics: List[Dict], 
                              apc_processor: EnhancedAPCProcessor):
        """Calculate divergences between clean and adversarial metrics"""
        divergences = []
        
        # Create mapping of clean metrics by image index
        clean_map = {}
        for metric in clean_metrics:
            img_name = metric['Image']
            try:
                img_idx = int(img_name.split('_')[2])
                clean_map[img_idx] = metric
            except:
                continue
        
        # Calculate divergences for each adversarial image
        for adv_metric in adv_metrics:
            try:
                adv_img_name = adv_metric['Image']
                adv_idx = int(adv_img_name.split('_')[2])
                
                if adv_idx in clean_map:
                    clean_metric = clean_map[adv_idx]
                    divergence = apc_processor.calculate_apc_divergence(clean_metric, adv_metric)
                    # Add metadata
                    divergence.update({
                        'Image': adv_img_name,
                        'Adversarial_or_Non_Adversarial': 'Divergence',
                        'Clean_Image': clean_metric['Image'],
                        'True_Label': adv_metric.get('True_Label', -1)
                    })
                    
                    divergences.append(divergence)
            except Exception as e:
                logger.error(f"Error calculating divergence for {adv_metric.get('Image', 'unknown')}: {e}")
                continue
        
        return divergences

    def _load_transform_image(self, img_path: str):
        """Load and transform image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(img_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def _save_metrics_to_csv(self, metrics: List[Dict], csv_file: str):
        """Save metrics to CSV file"""
        if not metrics:
            logger.warning("No metrics to save")
            return
        
        df = pd.DataFrame(metrics)
        
        # Handle list columns by converting to string
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        # Save to CSV
        if os.path.exists(csv_file):
            # Append to existing file
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(csv_file, index=False)
        else:
            # Create new file
            df.to_csv(csv_file, index=False)

    def generate_verification_metrics(self, dataset_name: str, architecture: str, 
                                    attack_types: List[str] = None):
        """Generate comprehensive verification metrics"""
        logger.info("Generating verification metrics...")
        
        # Initialize model
        _, _, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.initialize_model(self.device, dataset_name, architecture)
        
        # Create analyzer
        analyzer = VerificationMetricsAnalyzer(model, self.device, architecture)
        
        # Calculate metrics
        results = analyzer.calculate_verification_metrics(dataset_name, architecture, attack_types)
        
        return results

class VerificationMetricsAnalyzer:
    """Generate comprehensive verification metrics similar to the research paper format"""
    
    def __init__(self, model, device, architecture):
        self.model = model
        self.device = device
        self.apc_processor = EnhancedAPCProcessor(model, device, architecture)
        
    def calculate_verification_metrics(self, dataset_name: str, architecture: str, 
                                     attack_types: List[str] = None):
        """Calculate comprehensive verification metrics for different attack types"""
        
        if attack_types is None:
            attack_types = ['deepfool', 'fgsm', 'pgd', 'cw', 'bim']
        
        verification_results = []
        
        # First, calculate clean image metrics
        clean_metrics = self._calculate_clean_metrics(dataset_name, architecture)
        verification_results.append(clean_metrics)
        
        # Calculate metrics for each attack type
        for attack_type in attack_types:
            try:
                attack_metrics = self._calculate_attack_metrics(
                    dataset_name, architecture, attack_type, clean_metrics
                )
                if attack_metrics:
                    verification_results.append(attack_metrics)
            except Exception as e:
                logger.error(f"Error calculating metrics for {attack_type}: {e}")
                continue
        
        # Create and display results table
        self._create_verification_table(verification_results)
        
        return verification_results
    
    def _calculate_clean_metrics(self, dataset_name: str, architecture: str):
        """Calculate metrics for clean images"""
        logger.info("Calculating clean image metrics...")
        
        clean_dir = f"./{dataset_name.lower()}_{architecture}_clean_images"
        
        if not os.path.exists(clean_dir):
            logger.error(f"Clean images directory {clean_dir} not found!")
            return None
        
        # Calculate DNN accuracy on clean images
        dnn_accuracy = self._calculate_dnn_accuracy(clean_dir)
        
        return {
            'Input_Type': 'Clean',
            'DNN_Accuracy': dnn_accuracy,
            'APC_Trace_Divergence': 0.00,  # No divergence for clean images
            'Detection_Accuracy': 0.00,    # No adversarial examples to detect
            'Attack_Success_Rate': 0.00,
            'Average_Confidence': self._calculate_average_confidence(clean_dir),
            'Average_Loss': self._calculate_average_loss(clean_dir),
            'SSIM': 1.00  # Perfect similarity for clean images
        }
    
    def _calculate_attack_metrics(self, dataset_name: str, architecture: str, 
                                attack_type: str, clean_metrics: Dict):
        """Calculate metrics for specific attack type"""
        logger.info(f"Calculating metrics for {attack_type} attack...")
        
        adv_dir = f"./{dataset_name.lower()}_{architecture}_{attack_type}_images"
        clean_dir = f"./{dataset_name.lower()}_{architecture}_clean_images"
        
        if not os.path.exists(adv_dir):
            logger.warning(f"Adversarial images directory {adv_dir} not found!")
            return None
        
        # Calculate DNN accuracy on adversarial images
        dnn_accuracy = self._calculate_dnn_accuracy(adv_dir)
        
        # Calculate APC trace divergence
        apc_divergence = self._calculate_apc_trace_divergence(clean_dir, adv_dir)
        
        # Calculate detection accuracy using trained detector
        detection_accuracy = self._calculate_detection_accuracy(clean_dir, adv_dir)
        
        # Calculate attack success rate
        attack_success_rate = self._calculate_attack_success_rate(adv_dir)
        
        # Calculate average SSIM
        avg_ssim = self._calculate_average_ssim(clean_dir, adv_dir)
        
        return {
            'Input_Type': attack_type.upper(),
            'DNN_Accuracy': dnn_accuracy,
            'APC_Trace_Divergence': apc_divergence,
            'Detection_Accuracy': detection_accuracy,
            'Attack_Success_Rate': attack_success_rate,
            'Average_Confidence': self._calculate_average_confidence(adv_dir),
            'Average_Loss': self._calculate_average_loss(adv_dir),
            'SSIM': avg_ssim
        }
    
    def _calculate_dnn_accuracy(self, image_dir: str):
        """Calculate DNN accuracy on images in directory"""
        correct_predictions = 0
        total_predictions = 0
        
        for img_file in os.listdir(image_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                # Extract true label from filename
                true_label = int(img_file.split('_')[-1].split('.')[0])
                
                # Load and predict
                img_path = os.path.join(image_dir, img_file)
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predicted_label = torch.argmax(output, 1).item()
                
                if predicted_label == true_label:
                    correct_predictions += 1
                total_predictions += 1
                
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {e}")
                continue
        
        if total_predictions == 0:
            return 0.0
        
        accuracy = (correct_predictions / total_predictions) * 100
        return round(accuracy, 2)
    
    def _calculate_apc_trace_divergence(self, clean_dir: str, adv_dir: str):
        """Calculate APC trace divergence between clean and adversarial images"""
        try:
            # Load existing APC metrics if available
            csv_file = 'all_adversarial_non_adversarial.csv'
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                
                # Filter divergence rows
                divergence_rows = df[df['Adversarial_or_Non_Adversarial'] == 'Divergence']
                
                if len(divergence_rows) > 0:
                    # Calculate average divergence across key metrics
                    divergence_cols = [col for col in df.columns if col.endswith('_rel_diff')]
                    key_divergence_cols = [col for col in divergence_cols if any(
                        metric in col for metric in ['sparsity', 'activity', 'entropy', 'confidence']
                    )]
                    
                    if key_divergence_cols:
                        avg_divergence = divergence_rows[key_divergence_cols].mean().mean()
                        return round(avg_divergence, 2)  # Keep as percentage
            
            # If no existing data, calculate fresh divergence
            return self._calculate_fresh_apc_divergence(clean_dir, adv_dir)
            
        except Exception as e:
            logger.error(f"Error calculating APC divergence: {e}")
            return 0.0
    
    def _calculate_fresh_apc_divergence(self, clean_dir: str, adv_dir: str):
        """Calculate fresh APC divergence for a subset of images"""
        divergences = []
        
        # Get image pairs (limit to 10 for quick calculation)
        image_pairs = self._find_image_pairs(clean_dir, adv_dir)[:10]
        
        for clean_path, adv_path, img_idx, true_label in image_pairs:
            try:
                # Load images
                clean_tensor = self._load_transform_image(clean_path).to(self.device)
                adv_tensor = self._load_transform_image(adv_path).to(self.device)
                
                # Extract basic APC metrics
                clean_metrics = self._extract_basic_apc(clean_tensor)
                adv_metrics = self._extract_basic_apc(adv_tensor)
                
                # Calculate relative divergence
                divergence = 0
                count = 0
                for key in ['sparsity', 'activity', 'confidence']:
                    if key in clean_metrics and key in adv_metrics:
                        clean_val = clean_metrics[key]
                        adv_val = adv_metrics[key]
                        if clean_val != 0:
                            rel_diff = abs(adv_val - clean_val) / abs(clean_val)
                            divergence += rel_diff
                            count += 1
                
                if count > 0:
                    divergences.append(divergence / count)
                    
            except Exception as e:
                logger.warning(f"Error calculating divergence for pair {img_idx}: {e}")
                continue
        
        if divergences:
            return round(np.mean(divergences), 2)
        return 0.0
    
    def _extract_basic_apc(self, input_tensor: torch.Tensor):
        """Extract basic APC metrics quickly"""
        metrics = {}
        
        with torch.no_grad():
            # Register simple hooks
            activations = []
            
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations.append(output.detach())
            
            hooks = []
            for module in self.model.modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
            
            # Forward pass
            output = self.model(input_tensor)
            confidence = torch.max(nn.functional.softmax(output, dim=1)).item()
            
            # Calculate basic metrics
            if activations:
                # Sparsity (percentage of zero activations)
                total_zeros = sum(torch.count_nonzero(act == 0).item() for act in activations)
                total_elements = sum(act.numel() for act in activations)
                sparsity = (total_zeros / total_elements) * 100 if total_elements > 0 else 0
                
                # Activity (percentage of positive activations)
                total_positive = sum(torch.count_nonzero(act > 0).item() for act in activations)
                activity = (total_positive / total_elements) * 100 if total_elements > 0 else 0
                
                metrics.update({
                    'sparsity': sparsity,
                    'activity': activity,
                    'confidence': confidence
                })
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return metrics
    
    def _calculate_detection_accuracy(self, clean_dir: str, adv_dir: str):
        """Calculate detection accuracy using trained detector model"""
        try:
            # Check if detector models exist
            detector_files = ['xgboost_model.pkl', 'random_forest_model.pkl', 'dnn_model.h5']
            available_detectors = [f for f in detector_files if os.path.exists(f)]
            
            if not available_detectors:
                logger.warning("No trained detector models found. Training a quick detector...")
                return self._train_quick_detector(clean_dir, adv_dir)
            
            # Use the first available detector
            detector_file = available_detectors[0]
            
            if detector_file.endswith('.pkl'):
                detector = joblib.load(detector_file)
                return self._evaluate_ml_detector(detector, clean_dir, adv_dir)
            else:
                # Handle Keras model
                from tensorflow.keras.models import load_model
                detector = load_model(detector_file)
                return self._evaluate_dl_detector(detector, clean_dir, adv_dir)
                
        except Exception as e:
            logger.error(f"Error calculating detection accuracy: {e}")
            return 0.0
    
    def _train_quick_detector(self, clean_dir: str, adv_dir: str):
        """Train a quick detector for evaluation"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Extract features for a subset of images
            features, labels = self._extract_features_for_detection(clean_dir, adv_dir)
            
            if len(features) < 10:
                logger.warning(f"Not enough samples for training: {len(features)}")
                return 0.0
            
            logger.info(f"Training quick detector with {len(features)} samples and {features.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train quick RF classifier
            detector = RandomForestClassifier(n_estimators=50, random_state=42)
            detector.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            predictions = detector.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions) * 100
            
            logger.info(f"Quick detector accuracy: {accuracy:.2f}%")
            return round(accuracy, 2)
            
        except Exception as e:
            logger.error(f"Error training quick detector: {e}")
            return 0.0
    
    def _evaluate_ml_detector(self, detector, clean_dir: str, adv_dir: str):
        """Evaluate ML detector"""
        try:
            # Extract features using the same method as training
            features, labels = self._extract_features_for_detection(clean_dir, adv_dir)
            
            if len(features) < 5:
                logger.warning(f"Not enough samples for evaluation: {len(features)}")
                return 0.0
            
            logger.info(f"Evaluating ML detector with {len(features)} samples and {features.shape[1]} features")
            
            # Check if we have the expected number of features
            try:
                # Load preprocessors to check expected feature count
                scaler = joblib.load('scaler.pkl')
                feature_selector = joblib.load('feature_selector.pkl')
                
                # Apply preprocessing
                features_scaled = scaler.transform(features)
                features_selected = feature_selector.transform(features_scaled)
                
                # Make predictions
                predictions = detector.predict(features_selected)
                accuracy = accuracy_score(labels, predictions) * 100
                
                logger.info(f"ML detector accuracy: {accuracy:.2f}%")
                return round(accuracy, 2)
                
            except Exception as preprocessing_error:
                logger.error(f"Preprocessing error: {preprocessing_error}")
                # Fallback: train a new quick detector
                logger.info("Falling back to quick detector training...")
                return self._train_quick_detector(clean_dir, adv_dir)
                
        except Exception as e:
            logger.error(f"Error evaluating ML detector: {e}")
            return 0.0
    
    def _evaluate_dl_detector(self, detector, clean_dir: str, adv_dir: str):
        """Evaluate deep learning detector"""
        try:
            # Extract features using the same method as training
            features, labels = self._extract_features_for_detection(clean_dir, adv_dir)
            
            if len(features) < 5:
                logger.warning(f"Not enough samples for evaluation: {len(features)}")
                return 0.0
            
            logger.info(f"Evaluating DL detector with {len(features)} samples and {features.shape[1]} features")
            
            try:
                # Load preprocessors
                scaler = joblib.load('scaler.pkl')
                feature_selector = joblib.load('feature_selector.pkl')
                
                # Apply preprocessing
                features_scaled = scaler.transform(features)
                features_selected = feature_selector.transform(features_scaled)
                
                # Make predictions
                predictions = np.argmax(detector.predict(features_selected), axis=1)
                accuracy = accuracy_score(labels, predictions) * 100
                
                logger.info(f"DL detector accuracy: {accuracy:.2f}%")
                return round(accuracy, 2)
                
            except Exception as preprocessing_error:
                logger.error(f"Preprocessing error: {preprocessing_error}")
                # Fallback: train a new quick detector
                logger.info("Falling back to quick detector training...")
                return self._train_quick_detector(clean_dir, adv_dir)
                
        except Exception as e:
            logger.error(f"Error evaluating DL detector: {e}")
            return 0.0
    
    def _extract_features_for_detection(self, clean_dir: str, adv_dir: str):
        """Extract features for detection training"""
        features = []
        labels = []
        
        # Process clean images
        clean_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in clean_files[:20]:  # Limit for speed
            try:
                img_path = os.path.join(clean_dir, img_file)
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                # Extract true label from filename
                try:
                    true_label = int(img_file.split('_')[-1].split('.')[0])
                except ValueError:
                    continue
                
                # Extract FULL APC metrics (not just 3 basic ones)
                apc_metrics = self.apc_processor.process_image(
                    img_path, input_tensor, true_label, "Non Adversarial", None
                )
                
                # Convert metrics to feature vector (exclude metadata)
                excluded_keys = ['Image', 'Adversarial_or_Non_Adversarial', 'Predicted_Label', 'True_Label']
                feature_vector = []
                for key, value in apc_metrics.items():
                    if key not in excluded_keys and isinstance(value, (int, float)):
                        feature_vector.append(value)
                
                if len(feature_vector) > 0:
                    features.append(feature_vector)
                    labels.append(0)  # Clean = 0
                    
            except Exception as e:
                logger.warning(f"Error processing clean image {img_file}: {e}")
                continue
        
        # Process adversarial images
        adv_files = [f for f in os.listdir(adv_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in adv_files[:20]:  # Limit for speed
            try:
                img_path = os.path.join(adv_dir, img_file)
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                # Extract true label from filename
                try:
                    true_label = int(img_file.split('_')[-1].split('.')[0])
                except ValueError:
                    continue
                
                # Extract FULL APC metrics
                apc_metrics = self.apc_processor.process_image(
                    img_path, input_tensor, true_label, "Adversarial", None
                )
                
                # Convert metrics to feature vector (exclude metadata)
                excluded_keys = ['Image', 'Adversarial_or_Non_Adversarial', 'Predicted_Label', 'True_Label']
                feature_vector = []
                for key, value in apc_metrics.items():
                    if key not in excluded_keys and isinstance(value, (int, float)):
                        feature_vector.append(value)
                
                if len(feature_vector) > 0:
                    features.append(feature_vector)
                    labels.append(1)  # Adversarial = 1
                    
            except Exception as e:
                logger.warning(f"Error processing adversarial image {img_file}: {e}")
                continue
        
        if not features:
            return np.array([]), np.array([])
        
        # Ensure all feature vectors have the same length
        min_length = min(len(f) for f in features)
        features = [f[:min_length] for f in features]
        
        logger.info(f"Extracted {len(features)} samples with {min_length} features each")
        
        return np.array(features), np.array(labels)

    def _calculate_attack_success_rate(self, adv_dir: str):
        """Calculate attack success rate"""
        successful_attacks = 0
        total_attacks = 0
        
        for img_file in os.listdir(adv_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                # Extract true label from filename
                true_label = int(img_file.split('_')[-1].split('.')[0])
                
                # Load and predict
                img_path = os.path.join(adv_dir, img_file)
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predicted_label = torch.argmax(output, 1).item()
                
                # Attack is successful if prediction is wrong
                if predicted_label != true_label:
                    successful_attacks += 1
                total_attacks += 1
                
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {e}")
                continue
        
        if total_attacks == 0:
            return 0.0
        
        success_rate = (successful_attacks / total_attacks) * 100
        return round(success_rate, 2)
    
    def _calculate_average_confidence(self, image_dir: str):
        """Calculate average confidence for images in directory"""
        confidences = []
        
        for img_file in os.listdir(image_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                img_path = os.path.join(image_dir, img_file)
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    confidence = torch.max(nn.functional.softmax(output, dim=1)).item()
                    confidences.append(confidence)
                    
            except Exception as e:
                continue
        
        if confidences:
            return round(np.mean(confidences), 3)
        return 0.0
    
    def _calculate_average_loss(self, image_dir: str):
        """Calculate average loss for images in directory"""
        losses = []
        criterion = nn.CrossEntropyLoss()
        
        for img_file in os.listdir(image_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                # Extract true label
                true_label = int(img_file.split('_')[-1].split('.')[0])
                
                img_path = os.path.join(image_dir, img_file)
                input_tensor = self._load_transform_image(img_path).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    loss = criterion(output, torch.tensor([true_label], device=self.device))
                    losses.append(loss.item())
                    
            except Exception as e:
                continue
        
        if losses:
            return round(np.mean(losses), 3)
        return 0.0
    
    def _calculate_average_ssim(self, clean_dir: str, adv_dir: str):
        """Calculate average SSIM between clean and adversarial image pairs"""
        ssim_values = []
        image_pairs = self._find_image_pairs(clean_dir, adv_dir)
        
        for clean_path, adv_path, _, _ in image_pairs[:20]:  # Limit for speed
            try:
                ssim_val = ImageAnalyzer.calculate_ssim(clean_path, adv_path)
                if ssim_val > 0:
                    ssim_values.append(ssim_val)
            except:
                continue
        
        if ssim_values:
            return round(np.mean(ssim_values), 3)
        return 0.0
    
    def _find_image_pairs(self, clean_dir: str, adv_dir: str):
        """Find matching clean and adversarial image pairs"""
        pairs = []
        
        if not os.path.exists(clean_dir) or not os.path.exists(adv_dir):
            return pairs
        
        clean_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        adv_files = [f for f in os.listdir(adv_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for clean_file in clean_files:
            try:
                # Extract image index and label from clean filename
                parts = clean_file.split('_')
                if len(parts) >= 4:
                    img_idx = int(parts[2])
                    true_label = int(parts[4].split('.')[0])
                    
                    # Find corresponding adversarial image
                    matching_adv = None
                    for adv_file in adv_files:
                        if f"_{img_idx}_" in adv_file and f"_label_{true_label}" in adv_file:
                            matching_adv = adv_file
                            break
                    
                    if matching_adv:
                        clean_path = os.path.join(clean_dir, clean_file)
                        adv_path = os.path.join(adv_dir, matching_adv)
                        pairs.append((clean_path, adv_path, img_idx, true_label))
                        
            except (ValueError, IndexError):
                continue
        
        return pairs
    
    def _load_transform_image(self, img_path: str):
        """Load and transform image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(img_path).convert('RGB')
        return transform(image).unsqueeze(0)
    
    def _create_verification_table(self, results: List[Dict]):
        """Create and display verification metrics table"""
        if not results:
            logger.error("No results to display")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns for better presentation
        column_order = [
            'Input_Type', 'DNN_Accuracy', 'APC_Trace_Divergence', 
            'Detection_Accuracy', 'Attack_Success_Rate', 'Average_Confidence', 
            'Average_Loss', 'SSIM'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df_display = df[available_columns]
        
        # Format the table for better display
        print("\n" + "="*100)
        print("VERIFICATION METRICS TABLE")
        print("="*100)
        
        # Print table header
        header = f"{'Input Type':<12} {'DNN Acc(%)':<12} {'APC Div':<12} {'Det Acc(%)':<12} {'ASR(%)':<12} {'Conf':<8} {'Loss':<8} {'SSIM':<8}"
        print(header)
        print("-" * len(header))
        
        # Print each row
        for _, row in df_display.iterrows():
            input_type = str(row.get('Input_Type', 'N/A'))[:11]
            dnn_acc = f"{row.get('DNN_Accuracy', 0):.1f}%" if row.get('DNN_Accuracy') is not None else "N/A"
            apc_div = f"{row.get('APC_Trace_Divergence', 0):.2f}" if row.get('APC_Trace_Divergence') is not None else "N/A"
            det_acc = f"{row.get('Detection_Accuracy', 0):.1f}%" if row.get('Detection_Accuracy') is not None else "N/A"
            asr = f"{row.get('Attack_Success_Rate', 0):.1f}%" if row.get('Attack_Success_Rate') is not None else "N/A"
            conf = f"{row.get('Average_Confidence', 0):.3f}" if row.get('Average_Confidence') is not None else "N/A"
            loss = f"{row.get('Average_Loss', 0):.3f}" if row.get('Average_Loss') is not None else "N/A"
            ssim = f"{row.get('SSIM', 0):.3f}" if row.get('SSIM') is not None else "N/A"
            
            row_str = f"{input_type:<12} {dnn_acc:<12} {apc_div:<12} {det_acc:<12} {asr:<12} {conf:<8} {loss:<8} {ssim:<8}"
            print(row_str)
        
        print("="*100)
        
        # Save to CSV
        df_display.to_csv('verification_metrics_table.csv', index=False)
        logger.info("Verification metrics saved to verification_metrics_table.csv")
        
        # Create LaTeX table format
        self._create_latex_table(df_display)
        
        return df_display
    
    def _create_latex_table(self, df: pd.DataFrame):
        """Create LaTeX formatted table"""
        latex_file = 'verification_metrics_latex.txt'
        
        with open(latex_file, 'w') as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Verification Metrics}\n")
            f.write("\\label{tab:verification_metrics}\n")
            f.write("\\begin{tabular}{|l|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Input Type & DNN Accuracy (\\%) & APC Trace Divergence & Detection Accuracy (\\%) \\\\\n")
            f.write("\\hline\n")
            
            for _, row in df.iterrows():
                input_type = row.get('Input_Type', 'N/A')
                dnn_acc = f"{row.get('DNN_Accuracy', 0):.1f}\\%" if row.get('DNN_Accuracy') is not None else "N/A"
                apc_div = f"{row.get('APC_Trace_Divergence', 0):.2f}" if row.get('APC_Trace_Divergence') is not None else "N/A"
                det_acc = f"{row.get('Detection_Accuracy', 0):.1f}\\%" if row.get('Detection_Accuracy') is not None else "N/A"
                
                f.write(f"{input_type} & {dnn_acc} & {apc_div} & {det_acc} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"LaTeX table saved to {latex_file}")

class DetectorTrainer:
    """Handle detector model training with advanced features"""
    
    def __init__(self, csv_file: str = 'all_adversarial_non_adversarial.csv'):
        self.csv_file = csv_file
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file {self.csv_file} not found. Run APC extraction first.")
        
        df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Filter out divergence samples - we only want binary classification
        target_col = 'Adversarial_or_Non_Adversarial'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Keep only "Adversarial" and "Non Adversarial" samples
        binary_mask = df[target_col].isin(['Adversarial', 'Non Adversarial'])
        df_binary = df[binary_mask].copy()
        
        logger.info(f"Filtered to {len(df_binary)} samples for binary classification")
        logger.info(f"Class distribution: {df_binary[target_col].value_counts().to_dict()}")
        
        # Separate features and target
        y = df_binary[target_col]
        X = df_binary.drop([target_col, 'Image'], axis=1, errors='ignore')
        
        # Remove divergence-related columns and other metadata
        divergence_cols = [col for col in X.columns if 'divergence' in col.lower() or 
                          'relative_diff' in col or 'absolute_diff' in col or 'Clean_Image' in col]
        X = X.drop(divergence_cols, axis=1, errors='ignore')
        
        logger.info(f"Removed {len(divergence_cols)} divergence-related features")
        
        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop(col, axis=1)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Features: {list(X.columns[:10])}..." if len(X.columns) > 10 else f"Features: {list(X.columns)}")
        
        return X, y
    
    def train_models(self, test_size: float = 0.3, random_state: int = 42):
        """Train multiple detector models"""
        X, y = self.load_and_prepare_data()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self._perform_feature_selection(X_train_scaled, y_train, X.columns)
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        logger.info(f"Selected {X_train_selected.shape[1]} features out of {X_train_scaled.shape[1]}")
        
        # Train models
        self._train_traditional_models(X_train_selected, X_test_selected, y_train, y_test)
        self._train_deep_models(X_train_selected, X_test_selected, y_train, y_test)
        
        # Save models and preprocessors
        self._save_models_and_preprocessors()
        
        return X_test_selected, y_test
    
    def _perform_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 feature_names: List[str]):
        """Perform feature selection and analysis"""
        # Train random forest for feature importance
        rf_selector = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        rf_selector.fit(X_train, y_train)
        
        # Get feature importances
        importances = rf_selector.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title("Top 20 Feature Importances - Random Forest")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        
        # Select features using SelectFromModel
        self.feature_selector = SelectFromModel(
            rf_selector, threshold='median', prefit=True
        )
        
        # Get selected feature names
        selected_features = feature_names[self.feature_selector.get_support()]
        logger.info(f"Selected features: {list(selected_features)}")
        
        return selected_features
    
    def _train_traditional_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray):
        """Train traditional ML models"""
        models_config = {
            'XGBoost': xgb.XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric='logloss',
                n_estimators=100, max_depth=6, learning_rate=0.1
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, n_estimators=100, max_depth=10, n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', probability=True, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, name)
            metrics['training_time'] = training_time
            
            # Store model
            self.models[name] = {
                'model': model,
                'metrics': metrics
            }
            
            # Save individual model
            joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    def _train_deep_models(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray):
        """Train deep learning models"""
        # Should be binary classification now
        num_classes = 2
        logger.info(f"Training deep models for binary classification")
        
        # Prepare data for Keras
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        
        # Deep Neural Network
        dnn_model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])
        
        dnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Training Deep Neural Network...")
        history = dnn_model.fit(
            X_train, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test_cat),
            verbose=0
        )
        
        dnn_model.save('dnn_model.h5')
        dnn_predictions = np.argmax(dnn_model.predict(X_test), axis=1)
        dnn_proba = dnn_model.predict(X_test)[:, 1]
        
        dnn_metrics = self._evaluate_model(y_test, dnn_predictions, dnn_proba, "DNN")
        self.models['DNN'] = {
            'model': dnn_model,
            'metrics': dnn_metrics,
            'history': history.history
        }
        
        # LSTM Model
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        lstm_model = Sequential([
            LSTM(128, input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])
        
        lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Training LSTM...")
        lstm_history = lstm_model.fit(
            X_train_lstm, y_train_cat,
            epochs=30,
            batch_size=32,
            validation_data=(X_test_lstm, y_test_cat),
            verbose=0
        )
        
        lstm_model.save('lstm_model.h5')
        lstm_predictions = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
        lstm_proba = lstm_model.predict(X_test_lstm)[:, 1]
        
        lstm_metrics = self._evaluate_model(y_test, lstm_predictions, lstm_proba, "LSTM")
        self.models['LSTM'] = {
            'model': lstm_model,
            'metrics': lstm_metrics,
            'history': lstm_history.history
        }
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray = None, model_name: str = ""):
        """Comprehensive model evaluation"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score, roc_curve
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                
                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["auc"]:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
            except:
                metrics['auc'] = None
        
        # Print classification report
        print(f"\n--- {model_name} Model Evaluation ---")
        print(classification_report(y_true, y_pred))
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        if metrics.get('auc'):
            print(f"AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def _save_models_and_preprocessors(self):
        """Save models and preprocessors"""
        # Save traditional models (already saved individually)
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_selector, 'feature_selector.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        # Save model metrics
        metrics_summary = {}
        for name, model_info in self.models.items():
            if 'metrics' in model_info:
                metrics_summary[name] = {
                    k: v for k, v in model_info['metrics'].items() 
                    if k != 'confusion_matrix'
                }
        
        with open('model_metrics.json', 'w') as f:
            import json
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(metrics_summary, f, default=convert_numpy, indent=2)
        
        logger.info("Models and preprocessors saved successfully")
    
    def explain_models_with_shap(self, X_test: np.ndarray, feature_names: List[str]):
        """Generate SHAP explanations for models"""
        logger.info("Generating SHAP explanations...")
        
        # Ensure we don't have too many features for SHAP (can be slow)
        max_features_for_shap = 50
        if len(feature_names) > max_features_for_shap:
            logger.info(f"Too many features ({len(feature_names)}) for SHAP. Using top {max_features_for_shap} features.")
            # Use only the most important features
            if hasattr(self, 'feature_selector') and hasattr(self.feature_selector, 'estimator_'):
                importances = self.feature_selector.estimator_.feature_importances_
                top_indices = np.argsort(importances)[-max_features_for_shap:]
                X_test = X_test[:, top_indices]
                feature_names = feature_names[top_indices]
        
        # XGBoost SHAP
        if 'XGBoost' in self.models:
            try:
                logger.info("Generating XGBoost SHAP explanations...")
                xgb_model = self.models['XGBoost']['model']
                
                # Use a smaller sample for SHAP if dataset is large
                sample_size = min(100, X_test.shape[0])
                X_test_sample = X_test[:sample_size]
                
                explainer = shap.Explainer(xgb_model, X_test_sample)
                shap_values = explainer(X_test_sample)
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False, max_display=20)
                plt.tight_layout()
                plt.savefig('shap_summary_xgboost.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Waterfall plot for first prediction
                if len(X_test_sample) > 0:
                    plt.figure(figsize=(10, 8))
                    shap.waterfall_plot(explainer.expected_value, shap_values[0], 
                                       feature_names=feature_names, max_display=15, show=False)
                    plt.tight_layout()
                    plt.savefig('shap_waterfall_xgboost.png', dpi=300, bbox_inches='tight')
                    plt.show()
                
                logger.info("XGBoost SHAP explanations generated successfully")
                
            except Exception as e:
                logger.error(f"Error generating XGBoost SHAP explanations: {e}")
        
        # Random Forest SHAP
        if 'Random Forest' in self.models:
            try:
                logger.info("Generating Random Forest SHAP explanations...")
                rf_model = self.models['Random Forest']['model']
                
                # Use a smaller sample for SHAP
                sample_size = min(50, X_test.shape[0])  # Even smaller for TreeExplainer
                X_test_sample = X_test[:sample_size]
                
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X_test_sample)
                
                # For binary classification, use class 1 (adversarial)
                shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values_to_plot, X_test_sample, feature_names=feature_names, 
                                show=False, max_display=20)
                plt.tight_layout()
                plt.savefig('shap_summary_rf.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                logger.info("Random Forest SHAP explanations generated successfully")
                
            except Exception as e:
                logger.error(f"Error generating Random Forest SHAP explanations: {e}")
        
        logger.info("SHAP explanation generation completed")
    
    def analyze_feature_impact(self, num_features_range: List[int] = None):
        """Analyze impact of number of selected features on detection performance"""
        if num_features_range is None:
            num_features_range = [10, 20, 50, 100, 200, 500]
        
        X, y = self.load_and_prepare_data()
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get feature importances
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X_train_scaled, y_train)
        feature_importances = rf_selector.feature_importances_
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        
        results = []
        
        for num_features in num_features_range:
            if num_features > len(sorted_indices):
                continue
            
            # Select top features
            selected_indices = sorted_indices[:num_features]
            X_train_selected = X_train_scaled[:, selected_indices]
            X_test_selected = X_test_scaled[:, selected_indices]
            
            # Train XGBoost model
            model = xgb.XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            model.fit(X_train_selected, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'num_features': num_features,
                'accuracy': accuracy,
                'f1_score': f1
            })
            
            logger.info(f"Features: {num_features}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Plot results
        results_df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(results_df['num_features'], results_df['accuracy'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Accuracy')
        plt.title('Detection Accuracy vs Number of Features')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(results_df['num_features'], results_df['f1_score'], 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Number of Selected Features')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Number of Features')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        results_df.to_csv('feature_impact_analysis.csv', index=False)
        
        return results_df

class APCDivergenceAnalyzer:
    """Analyze APC divergence between clean and adversarial versions of same images"""
    
    def __init__(self, model, device, architecture):
        self.model = model
        self.device = device
        self.apc_processor = EnhancedAPCProcessor(model, device, architecture)
    
    def analyze_apc_divergence(self, clean_dir: str, adv_dir: str, output_file: str = "apc_divergence_analysis.csv"):
        """Analyze APC divergence between paired clean and adversarial images"""
        logger.info("Starting APC divergence analysis...")
        
        # Get clean and adversarial image pairs
        image_pairs = self._find_image_pairs(clean_dir, adv_dir)
        logger.info(f"Found {len(image_pairs)} clean-adversarial image pairs")
        
        if len(image_pairs) == 0:
            logger.error("No matching image pairs found!")
            return None
        
        divergence_results = []
        
        for clean_path, adv_path, img_idx, true_label in image_pairs:
            try:
                # Extract APC metrics for clean image
                clean_tensor = self._load_transform_image(clean_path).to(self.device)
                clean_metrics = self.apc_processor.process_image(
                    clean_path, clean_tensor, true_label, "Clean", None
                )
                
                # Extract APC metrics for adversarial image
                adv_tensor = self._load_transform_image(adv_path).to(self.device)
                adv_metrics = self.apc_processor.process_image(
                    adv_path, adv_tensor, true_label, "Adversarial", clean_path
                )
                
                # Calculate divergences for key APC metrics
                divergence = self._calculate_key_divergences(clean_metrics, adv_metrics, img_idx)
                divergence_results.append(divergence)
                
                logger.info(f"Processed pair {len(divergence_results)}/{len(image_pairs)}: {os.path.basename(clean_path)}")
                
            except Exception as e:
                logger.error(f"Error processing pair {clean_path} - {adv_path}: {e}")
                continue
        
        # Save results
        if divergence_results:
            df = pd.DataFrame(divergence_results)
            df.to_csv(output_file, index=False)
            logger.info(f"Divergence analysis saved to {output_file}")
            
            # Generate visualizations
            self._plot_apc_divergence(df)
            
            return df
        else:
            logger.error("No successful divergence calculations!")
            return None
    
    def _find_image_pairs(self, clean_dir: str, adv_dir: str):
        """Find matching clean and adversarial image pairs"""
        pairs = []
        
        if not os.path.exists(clean_dir) or not os.path.exists(adv_dir):
            logger.error(f"Directory not found: {clean_dir} or {adv_dir}")
            return pairs
        
        clean_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        adv_files = [f for f in os.listdir(adv_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for clean_file in clean_files:
            try:
                # Extract image index and label from clean filename
                # Expected format: clean_image_X_label_Y.png
                parts = clean_file.split('_')
                if len(parts) >= 4:
                    img_idx = int(parts[2])
                    true_label = int(parts[4].split('.')[0])
                    

                    matching_adv = None
                    for adv_file in adv_files:
                        if f"_{img_idx}_" in adv_file and f"_label_{true_label}" in adv_file:
                            matching_adv = adv_file
                            break
                    
                    if matching_adv:
                        clean_path = os.path.join(clean_dir, clean_file)
                        adv_path = os.path.join(adv_dir, matching_adv)
                        pairs.append((clean_path, adv_path, img_idx, true_label))
                        
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse filename {clean_file}: {e}")
                continue
        
        return pairs
    
    def _calculate_key_divergences(self, clean_metrics: Dict, adv_metrics: Dict, img_idx: int):
        """Calculate divergences for key APC metrics"""
        divergence = {
            'Image_Index': img_idx,
            'True_Label': clean_metrics.get('True_Label', -1),
            'Clean_Image': os.path.basename(clean_metrics.get('Image', '')),
            'Adversarial_Image': os.path.basename(adv_metrics.get('Image', ''))
        }
        
        # Key metrics to analyze
        key_metrics = [
            'Confidence', 'Loss', 'Output_Entropy', 'Inference_Time',
            'CPU_Usage_After', 'GPU_Util_After', 'GPU_Memory_After'
        ]
        
        # Add layer-wise metrics (sample from different layers)
        sample_layers = ['layer_0', 'layer_10', 'layer_20', 'layer_30', 'layer_40', 'layer_50']
        for layer in sample_layers:
            for metric_type in ['sparsity', 'activity', 'variance', 'entropy', 'l2_norm']:
                key_metrics.append(f'{layer}_{metric_type}')
        
        # Add aggregated metrics
        agg_metrics = ['avg_sparsity', 'avg_activity', 'avg_variance', 'std_sparsity', 'std_activity']
        key_metrics.extend(agg_metrics)
        
        # Calculate divergences
        for key in key_metrics:
            clean_val = clean_metrics.get(key, 0)
            adv_val = adv_metrics.get(key, 0)
            
            if isinstance(clean_val, (int, float)) and isinstance(adv_val, (int, float)):
                # Absolute difference
                abs_diff = abs(adv_val - clean_val)
                divergence[f'{key}_abs_diff'] = abs_diff
                
                # Relative difference (percentage change)
                if clean_val != 0:
                    rel_diff = abs_diff / abs(clean_val) * 100
                else:
                    rel_diff = abs_diff * 100
                divergence[f'{key}_rel_diff'] = rel_diff
                
                # Store original values for reference
                divergence[f'{key}_clean'] = clean_val
                divergence[f'{key}_adv'] = adv_val
        
        # Calculate SSIM if available
        if 'ssim' in adv_metrics:
            divergence['SSIM'] = adv_metrics['ssim']
        
        return divergence
    
    def _plot_apc_divergence(self, df: pd.DataFrame):
        """Generate comprehensive APC divergence visualizations"""
        logger.info("Generating APC divergence plots...")
        
        # 1. Overall divergence heatmap
        plt.figure(figsize=(15, 10))
        
        # Select key divergence metrics for heatmap
        divergence_cols = [col for col in df.columns if col.endswith('_rel_diff')]
        key_divergence_cols = divergence_cols[:20]  # Top 20 for readability
        
        if key_divergence_cols:
            heatmap_data = df[key_divergence_cols].T
            heatmap_data.columns = [f"Image_{i}" for i in range(len(heatmap_data.columns))]
            
            sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Relative Difference (%)'})
            plt.title('APC Divergence Heatmap: Clean vs Adversarial Images')
            plt.xlabel('Image Pairs')
            plt.ylabel('APC Metrics')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('apc_divergence_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Before/After APC Values Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        key_metrics = ['Confidence', 'Loss', 'Output_Entropy', 'avg_sparsity', 'avg_activity', 'Inference_Time']
        
        for i, metric in enumerate(key_metrics):
            if f'{metric}_clean' in df.columns and f'{metric}_adv' in df.columns:
                clean_vals = df[f'{metric}_clean']
                adv_vals = df[f'{metric}_adv']
                
                # Paired plot showing before/after
                x_pos = range(len(df))
                axes[i].plot(x_pos, clean_vals, 'o-', label='Clean', color='blue', alpha=0.7, linewidth=2)
                axes[i].plot(x_pos, adv_vals, 's-', label='Adversarial', color='red', alpha=0.7, linewidth=2)
                
                # Connect paired points with lines
                for j in range(len(df)):
                    axes[i].plot([j, j], [clean_vals.iloc[j], adv_vals.iloc[j]], 
                               'k--', alpha=0.3, linewidth=0.5)
                
                axes[i].set_title(f'{metric}: Clean vs Adversarial')
                axes[i].set_xlabel('Image Index')
                axes[i].set_ylabel(metric)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('apc_before_after_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Summary statistics visualization
        self._plot_summary_statistics(df)

    def _plot_summary_statistics(self, df: pd.DataFrame):
        """Plot summary statistics of APC changes"""
        plt.figure(figsize=(16, 10))
        
        # Key metrics to analyze
        key_metrics = ['Confidence_abs_diff', 'Loss_abs_diff', 'Output_Entropy_abs_diff', 
                      'avg_sparsity_rel_diff', 'avg_activity_rel_diff', 'Inference_Time_rel_diff']
        
        # Box plots
        plt.subplot(2, 3, 1)
        data_to_plot = [df[metric].dropna() for metric in key_metrics if metric in df.columns]
        labels = [metric.replace('_abs_diff', '').replace('_rel_diff', '').replace('_', ' ').title() 
                 for metric in key_metrics if metric in df.columns]
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.title('Distribution of APC Changes')
            plt.ylabel('Change Magnitude')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        # Violin plots
        plt.subplot(2, 3, 2)
        if len(data_to_plot) > 0:
            parts = plt.violinplot(data_to_plot, positions=range(1, len(data_to_plot) + 1))
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
            plt.title('Density of APC Changes')
            plt.ylabel('Change Magnitude')
            plt.grid(True, alpha=0.3)
        
        # Average changes bar plot
        plt.subplot(2, 3, 3)
        if data_to_plot:
            means = [np.mean(data) for data in data_to_plot]
            stds = [np.std(data) for data in data_to_plot]
            plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
            plt.title('Average APC Changes with Std Dev')
            plt.ylabel('Average Change')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 3, 4)
        if 'Confidence_abs_diff' in df.columns:
            sorted_conf = np.sort(df['Confidence_abs_diff'].dropna())
            y = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
            plt.plot(sorted_conf, y, 'b-', linewidth=2)
            plt.title('Cumulative Distribution: Confidence Changes')
            plt.xlabel('Confidence Change')
            plt.ylabel('Cumulative Probability')
            plt.grid(True, alpha=0.3)
        
        # Image-wise total change
        plt.subplot(2, 3, 5)
        change_cols = [col for col in df.columns if col.endswith('_abs_diff')]
        if change_cols:
            df['total_change'] = df[change_cols].sum(axis=1)
            plt.plot(range(len(df)), df['total_change'], 'ro-', alpha=0.7)
            plt.title('Total APC Change per Image')
            plt.xlabel('Image Index')
            plt.ylabel('Total Change Score')
            plt.grid(True, alpha=0.3)
        
        # Most affected metrics
        plt.subplot(2, 3, 6)
        if change_cols:
            metric_means = df[change_cols].mean().sort_values(ascending=False)
            top_metrics = metric_means.head(10)
            top_metrics.plot(kind='barh', color='coral')
            plt.title('Top 10 Most Affected APC Metrics')
            plt.xlabel('Average Change')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('apc_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _load_transform_image(self, img_path: str):
        """Load and transform image"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(img_path).convert('RGB')
        return transform(image).unsqueeze(0)



class AttackTester:
    """Utility class for testing and debugging adversarial attacks"""
    
    def __init__(self, framework_manager):
        self.framework = framework_manager
        self.device = framework_manager.device
    
    def test_all_attacks(self, dataset_name: str, architecture: str):
        """Test all available attacks with a small dataset"""
        logger.info("Testing all available attacks...")
        
        # Initialize model
        _, _, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.initialize_model(self.device, dataset_name, architecture)
        
        # Create attacker
        attacker = AdversarialAttacker(model, self.device)
        
        # Test each available attack
        results = {}
        for attack_name in attacker.list_available_attacks():
            logger.info(f"Testing {attack_name}...")
            
            try:
                success = self._test_single_attack(attacker, attack_name, dataset_name, architecture)
                results[attack_name] = {
                    'available': True,
                    'test_passed': success,
                    'info': self._get_attack_info(attack_name)
                }
                
                if success:
                    logger.info(f"✓ {attack_name}: PASSED")
                else:
                    logger.warning(f"⚠ {attack_name}: FAILED TEST")
                    
            except Exception as e:
                results[attack_name] = {
                    'available': False,
                    'test_passed': False,
                    'error': str(e),
                    'info': self._get_attack_info(attack_name)
                }
                logger.error(f"✗ {attack_name}: ERROR - {e}")
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def _test_single_attack(self, attacker, attack_name: str, dataset_name: str, architecture: str, num_test_samples: int = 5):
        """Test if an attack works with a small number of samples"""
        logger.info(f"Testing {attack_name} attack...")
        
        try:
            # Load a small batch of data
            _, testset, _ = DatasetManager.get_datasets(dataset_name)
            transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            testset.transform = transform
            testloader = DataLoader(testset, batch_size=2, shuffle=False)
            
            # Get first batch
            cln_data, true_label = next(iter(testloader))
            cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
            
            # Test attack
            attack = attacker.get_attack(attack_name)
            _, adv_examples, success = attack(attacker.fmodel, cln_data, true_label, epsilons=[0.1])
            
            if adv_examples[0] is not None:
                logger.info(f"✓ {attack_name} attack test successful!")
                return True
            else:
                logger.warning(f"✗ {attack_name} attack test failed - no adversarial examples generated")
                return False
                
        except Exception as e:
            logger.error(f" {attack_name} attack test failed: {e}")
            return False
    
    def _get_attack_info(self, attack_type: str):
        """Get information about a specific attack"""
        attack_info = {
            "fgsm": {
                "name": "Fast Gradient Sign Method",
                "norm": "L∞",
                "description": "Single-step gradient-based attack"
            },
            "pgd": {
                "name": "Projected Gradient Descent",
                "norm": "L∞",
                "description": "Multi-step iterative gradient-based attack"
            },
            "deepfool": {
                "name": "DeepFool",
                "norm": "L2",
                "description": "Minimal perturbation attack using linear approximation"
            },
            "cw": {
                "name": "Carlini & Wagner",
                "norm": "L2",
                "description": "Optimization-based attack with high transferability"
            },
            "jsma": {
                "name": "Jacobian-based Saliency Map Attack",
                "norm": "L0",
                "description": "Greedy pixel-wise perturbation attack"
            },
            "lbfgs": {
                "name": "L-BFGS Attack",
                "norm": "L2",
                "description": "Optimization-based attack using L-BFGS"
            },
            "bim": {
                "name": "Basic Iterative Method",
                "norm": "L∞",
                "description": "Iterative version of FGSM"
            }
        }
        
        return attack_info.get(attack_type, {"name": attack_type, "description": "Unknown attack"})
    
    def _print_test_summary(self, results):
        """Print a formatted summary of attack test results"""
        print("\n" + "="*80)
        print("ATTACK AVAILABILITY AND TEST SUMMARY")
        print("="*80)
        
        print(f"{'Attack Name':<25} {'Available':<12} {'Test Result':<15} {'Norm':<8} {'Description'}")
        print("-" * 80)
        
        for attack_name, result in results.items():
            available = "✓ Yes" if result['available'] else "✗ No"
            test_result = "✓ Pass" if result.get('test_passed', False) else "✗ Fail"
            
            info = result.get('info', {})
            norm = info.get('norm', 'N/A')
            description = info.get('description', 'Unknown')[:30] + '...' if len(info.get('description', '')) > 30 else info.get('description', 'Unknown')
            
            print(f"{attack_name:<25} {available:<12} {test_result:<15} {norm:<8} {description}")
        
        print("="*80)
        
        # Count working attacks
        working_attacks = sum(1 for r in results.values() if r.get('test_passed', False))
        total_attacks = len(results)
        
        print(f"Working attacks: {working_attacks}/{total_attacks}")
        print(f"Success rate: {working_attacks/total_attacks*100:.1f}%")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if working_attacks == 0:
            print("- No attacks are working. Check your Foolbox installation and model compatibility.")
        elif working_attacks < total_attacks:
            print("- Some attacks failed. This is normal - different Foolbox versions support different attacks.")
            print("- Use the working attacks for your experiments.")
        else:
            print("- All attacks are working! You have full attack capability.")
        
        print("\nWorking attacks you can use:")
        for attack_name, result in results.items():
            if result.get('test_passed', False):
                print(f"  - {attack_name}: {result['info'].get('description', 'No description')}")
    
    def benchmark_attacks(self, dataset_name: str, architecture: str, num_samples: int = 20):
        """Benchmark different attacks on a small dataset"""
        logger.info(f"Benchmarking attacks with {num_samples} samples...")
        
        # Initialize model
        _, _, num_classes = DatasetManager.get_datasets(dataset_name)
        model = ModelManager.initialize_model(self.device, dataset_name, architecture)
        
        # Create attacker
        attacker = AdversarialAttacker(model, self.device)
        
        # Get working attacks
        working_attacks = []
        for attack_name in attacker.list_available_attacks():
            if self._test_single_attack(attacker, attack_name, dataset_name, architecture):
                working_attacks.append(attack_name)
        
        if not working_attacks:
            logger.error("No working attacks found!")
            return None
        
        logger.info(f"Benchmarking {len(working_attacks)} working attacks: {working_attacks}")
        
        # Benchmark each attack
        benchmark_results = {}
        
        for attack_name in working_attacks:
            logger.info(f"Benchmarking {attack_name}...")
            
            try:
                start_time = time.time()
                asr = attacker.perform_attack(dataset_name, architecture, attack_name, num_samples)
                execution_time = time.time() - start_time
                
                benchmark_results[attack_name] = {
                    'asr': asr,
                    'execution_time': execution_time,
                    'samples_per_second': num_samples / execution_time if execution_time > 0 else 0,
                    'info': self._get_attack_info(attack_name)
                }
                
                logger.info(f"{attack_name}: ASR={asr:.2f}%, Time={execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Benchmarking failed for {attack_name}: {e}")
                benchmark_results[attack_name] = {
                    'asr': 0.0,
                    'execution_time': 0.0,
                    'samples_per_second': 0.0,
                    'error': str(e),
                    'info': self._get_attack_info(attack_name)
                }
        
        # Print benchmark summary
        self._print_benchmark_summary(benchmark_results, num_samples)
        
        # Save results
        self._save_benchmark_results(benchmark_results, dataset_name, architecture)
        
        return benchmark_results
    
    def _print_benchmark_summary(self, results, num_samples):
        """Print benchmark results summary"""
        print("\n" + "="*90)
        print("ATTACK BENCHMARK RESULTS")
        print("="*90)
        
        print(f"{'Attack':<20} {'ASR (%)':<12} {'Time (s)':<12} {'Samples/s':<12} {'Norm':<8} {'Status'}")
        print("-" * 90)
        
        for attack_name, result in results.items():
            asr = f"{result['asr']:.2f}" if 'asr' in result else "N/A"
            exec_time = f"{result['execution_time']:.2f}" if 'execution_time' in result else "N/A"
            samples_per_sec = f"{result['samples_per_second']:.2f}" if 'samples_per_second' in result else "N/A"
            norm = result['info'].get('norm', 'N/A')
            status = "✓ Success" if 'error' not in result else "✗ Failed"
            
            print(f"{attack_name:<20} {asr:<12} {exec_time:<12} {samples_per_sec:<12} {norm:<8} {status}")
        
        print("="*90)
        
        # Find best performing attacks
        successful_results = {k: v for k, v in results.items() if 'error' not in v and v['asr'] > 0}
        
        if successful_results:
            best_asr = max(successful_results.items(), key=lambda x: x[1]['asr'])
            fastest = max(successful_results.items(), key=lambda x: x[1]['samples_per_second'])
            
            print(f"\nBest ASR: {best_asr[0]} ({best_asr[1]['asr']:.2f}%)")
            print(f"Fastest: {fastest[0]} ({fastest[1]['samples_per_second']:.2f} samples/s)")
            
            # Recommendations
            print(f"\nRECOMMENDED ATTACKS:")
            print(f"- For highest success rate: {best_asr[0]}")
            print(f"- For fastest execution: {fastest[0]}")
            
            # Attack diversity recommendation
            norms = set(r['info'].get('norm', 'Unknown') for r in successful_results.values())
            print(f"- Available norms: {', '.join(norms)}")
            print(f"- Use different norms for comprehensive evaluation")
        else:
            print("\nNo successful attacks found!")
    
    def _save_benchmark_results(self, results, dataset_name, architecture):
        """Save benchmark results to file"""
        import json
        
        # Convert results to JSON-serializable format
        json_results = {}
        for attack_name, result in results.items():
            json_results[attack_name] = {
                'asr': result.get('asr', 0.0),
                'execution_time': result.get('execution_time', 0.0),
                'samples_per_second': result.get('samples_per_second', 0.0),
                'error': result.get('error', None),
                'info': result.get('info', {})
            }
        
        # Add metadata
        metadata = {
            'dataset': dataset_name,
            'architecture': architecture,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': json_results
        }
        
        filename = f'attack_benchmark_{dataset_name}_{architecture}.json'
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")
    
    def debug_attack_failure(self, attack_name: str, dataset_name: str, architecture: str):
        """Debug why a specific attack is failing"""
        logger.info(f"Debugging {attack_name} attack failure...")
        
        try:
            # Initialize model
            _, _, num_classes = DatasetManager.get_datasets(dataset_name)
            model = ModelManager.initialize_model(self.device, dataset_name, architecture)
            
            # Create attacker
            attacker = AdversarialAttacker(model, self.device)
            
            # Check if attack is available
            if attack_name not in attacker.list_available_attacks():
                logger.error(f"Attack {attack_name} is not available in this Foolbox version")
                logger.info("Available attacks:", list(attacker.list_available_attacks()))
                return False
            
            # Try to create attack instance
            try:
                attack = attacker.get_attack(attack_name)
                logger.info(f"✓ Attack instance created successfully: {type(attack)}")
            except Exception as e:
                logger.error(f"✗ Failed to create attack instance: {e}")
                return False
            
            # Load test data
            _, testset, _ = DatasetManager.get_datasets(dataset_name)
            transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            testset.transform = transform
            testloader = DataLoader(testset, batch_size=1, shuffle=False)
            
            # Get single sample
            cln_data, true_label = next(iter(testloader))
            cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
            
            logger.info(f"Test data shape: {cln_data.shape}")
            logger.info(f"True label: {true_label.item()}")
            
            # Check model prediction
            with torch.no_grad():
                prediction = model(cln_data)
                pred_label = torch.argmax(prediction, dim=1).item()
                confidence = torch.max(torch.softmax(prediction, dim=1)).item()
            
            logger.info(f"Model prediction: {pred_label}, Confidence: {confidence:.4f}")
            
            if pred_label != true_label.item():
                logger.warning("Model misclassifies clean sample - this may cause attack issues")
            
            # Try attack with minimal settings
            try:
                logger.info("Attempting attack with minimal epsilon...")
                _, adv_examples, success = attack(attacker.fmodel, cln_data, true_label, epsilons=[0.01])
                
                if adv_examples[0] is not None:
                    logger.info(" Attack generated adversarial example")
                    
                    # Check adversarial prediction
                    with torch.no_grad():
                        adv_prediction = model(adv_examples[0])
                        adv_pred_label = torch.argmax(adv_prediction, dim=1).item()
                        adv_confidence = torch.max(torch.softmax(adv_prediction, dim=1)).item()
                    
                    logger.info(f"Adversarial prediction: {adv_pred_label}, Confidence: {adv_confidence:.4f}")
                    logger.info(f"Attack success: {success[0][0].item()}")
                    
                    # Calculate perturbation magnitude
                    perturbation = (adv_examples[0] - cln_data).abs()
                    l2_norm = torch.norm(perturbation).item()
                    linf_norm = torch.max(perturbation).item()
                    
                    logger.info(f"Perturbation L2 norm: {l2_norm:.6f}")
                    logger.info(f"Perturbation L∞ norm: {linf_norm:.6f}")
                    
                    return True
                else:
                    logger.error(" Attack failed to generate adversarial examples")
                    return False
                    
            except Exception as e:
                logger.error(f"✗ Attack execution failed: {e}")
                return False
                    
        except Exception as e:
            logger.error(f"Debug failed with error: {e}")
            return False
        
def main():
    """Enhanced main function with comprehensive attack support"""
    IntroDisplay.show_intro()
    
    parser = argparse.ArgumentParser(description='SAMURAI: Enhanced Adversarial Detection Framework')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, 
                       choices=['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN','STL10'], 
                       help='Dataset to use (required for most operations)')
    parser.add_argument('--dataset1', type=str, 
                       choices=['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN','STL10'], 
                       help='Dataset to use (required for most operations)')
    parser.add_argument('--dataset2', type=str, 
                       choices=['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN','STL10'], 
                       help='Dataset to use (required for most operations)')
    parser.add_argument('--architecture', type=str, 
                       choices=['vgg16', 'vgg19', 'alexnet', 'resnet18', 'resnet34', 
                               'resnet50', 'resnet101', 'resnet152', 'densenet121', 
                               'densenet169', 'inception_v3', 'mobilenet_v2','VITA','VITB'],
                       help='Model architecture to use (required for most operations)')
    
    # Training arguments
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    # Attack arguments
    parser.add_argument('--attack', type=str, help='Perform specified attack (use --list_attacks to see available)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for attack/benchmark')
    
    # Analysis arguments
    parser.add_argument('--apc', action='store_true', help='Extract APC metrics')
    parser.add_argument('--train_detector', action='store_true', help='Train the ML detector model')
    parser.add_argument('--analyze_features', action='store_true', help='Analyze feature impact on detection')
    parser.add_argument('--explain_shap', action='store_true', help='Generate SHAP explanations')
    parser.add_argument('--analyze_divergence', action='store_true', help='Analyze APC divergence')
    parser.add_argument('--evaluate_detector', action='store_true', help='Evaluate the detector model')
    parser.add_argument('--verification_metrics', action='store_true', help='Generate verification metrics table')
    
    # New testing and debugging arguments
    parser.add_argument('--test_attacks', action='store_true', help='Test all available attacks')
    parser.add_argument('--benchmark_attacks', action='store_true', help='Benchmark all working attacks')
    parser.add_argument('--debug_attack', type=str, help='Debug specific attack failure')
    parser.add_argument('--list_attacks', action='store_true', help='List all available attacks')
    parser.add_argument('--attack_guide', action='store_true', help='Show comprehensive attack usage guide')

    parser.add_argument(
        '--extract_divergence',
        action='store_true',
        help='Run extract_divergence_metrics on two datasets and an architecture: --extract_divergence_metrics DATASET1 DATASET2 ARCH'
    )

    parser.add_argument(
        '--extract_class_divergence',
        action='store_true',
        help = 'Run extract divergence metrics on same dataset class wise and an architecture'
    )
    
    args = parser.parse_args()

    # Initialize framework
    framework = FrameworkManager()
    
    # Initialize system monitoring
    framework.system_monitor.initialize_nvml()

    try:
        # Show attack guide
        if args.attack_guide:
            print("""
SAMURAI FRAMEWORK - ADVERSARIAL ATTACK GUIDE
============================================

QUICK START:
1. Test available attacks: python samurai.py --test_attacks --dataset CIFAR10 --architecture resnet18
2. List attack details: python samurai.py --list_attacks --dataset CIFAR10 --architecture resnet18
3. Run specific attack: python samurai.py --attack pgd --dataset CIFAR10 --architecture resnet18

SUPPORTED ATTACKS:
- FGSM: Fast Gradient Sign Method (L∞)
- PGD: Projected Gradient Descent (L∞/L2)
- DeepFool: Minimal perturbation attack (L2/L∞)
- C&W: Carlini & Wagner optimization attack (L2)
- JSMA: Jacobian-based Saliency Map Attack (L0)
- L-BFGS: Limited-memory BFGS optimization attack (L2)
- BIM: Basic Iterative Method (L∞)

TROUBLESHOOTING:
- If attacks fail: Use --debug_attack <attack_name>
- If no attacks work: Check Foolbox installation
- For performance issues: Reduce --num_samples

EXAMPLE WORKFLOWS:
1. Full evaluation:
   python samurai.py --train --dataset CIFAR10 --architecture resnet18
   python samurai.py --test_attacks --dataset CIFAR10 --architecture resnet18
   python samurai.py --attack pgd --dataset CIFAR10 --architecture resnet18
   python samurai.py --apc --attack pgd --dataset CIFAR10 --architecture resnet18
   python samurai.py --train_detector
   python samurai.py --verification_metrics --dataset CIFAR10 --architecture resnet18

2. Quick attack test:
   python samurai.py --benchmark_attacks --dataset CIFAR10 --architecture resnet18 --num_samples 20
""")
            return
        
        # List available attacks
        if args.list_attacks:
            if not args.dataset or not args.architecture:
                parser.error("--list_attacks requires --dataset and --architecture")
            
            logger.info("Initializing model to check available attacks...")
            _, _, num_classes = DatasetManager.get_datasets(args.dataset)
            model = ModelManager.initialize_model(framework.device, args.dataset, args.architecture)
            attacker = AdversarialAttacker(model, framework.device)
            
            print("\n" + "="*80)
            print("AVAILABLE ADVERSARIAL ATTACKS")
            print("="*80)
            
            for attack in attacker.list_available_attacks():
                info = attacker.get_attack_info(attack)
                print(f"• {attack:15} - {info['description']}")
                print(f"  {'':15}   Norm: {info.get('norm', 'Unknown')}")
                print()
            
            print(f"Total available attacks: {len(attacker.list_available_attacks())}")
            print("Use --test_attacks to test these attacks")
            return
        
        # Test all attacks
        if args.test_attacks:
            if not args.dataset or not args.architecture:
                parser.error("--test_attacks requires --dataset and --architecture")
            
            logger.info("Testing all available attacks...")
            tester = AttackTester(framework)
            results = tester.test_all_attacks(args.dataset, args.architecture)
            
            # Save test results
            working_attacks = [k for k, v in results.items() if v.get('test_passed', False)]
            with open('working_attacks.txt', 'w') as f:
                f.write("Working attacks:\n")
                for attack in working_attacks:
                    f.write(f"- {attack}\n")
            
            logger.info(f"Test results saved. {len(working_attacks)} attacks are working.")
            return
        
        # Benchmark attacks
        if args.benchmark_attacks:
            if not args.dataset or not args.architecture:
                parser.error("--benchmark_attacks requires --dataset and --architecture")
            
            logger.info(f"Benchmarking attacks with {args.num_samples} samples...")
            tester = AttackTester(framework)
            results = tester.benchmark_attacks(args.dataset, args.architecture, args.num_samples)
            return
        
        # Debug specific attack
        if args.debug_attack:
            if not args.dataset or not args.architecture:
                parser.error("--debug_attack requires --dataset and --architecture")
            
            logger.info(f"Debugging {args.debug_attack} attack...")
            tester = AttackTester(framework)
            success = tester.debug_attack_failure(args.debug_attack, args.dataset, args.architecture)
            
            if success:
                logger.info(f"✓ Attack {args.debug_attack} is working correctly")
                print(f"\nYou can now use: python samurai.py --attack {args.debug_attack} --dataset {args.dataset} --architecture {args.architecture}")
            else:
                logger.error(f"✗ Attack {args.debug_attack} has issues")
                print(f"\nTry using a different attack. Use --list_attacks to see available options.")
            return

        # Check if dataset and architecture are required
        requires_model = args.train or args.attack or args.apc or args.verification_metrics
        if requires_model and (not args.dataset or not args.architecture):
            parser.error("--dataset and --architecture are required for the selected operation")

        # Train model if requested
        if args.train:
            logger.info(f"Training {args.architecture} on {args.dataset} for {args.epochs} epochs...")
            framework.train_model(args.dataset, args.architecture, args.epochs)

        if args.extract_class_divergence:
            if not args.dataset or not args.architecture:
                logger.info("Divergence Extraction Requires Dataset and architecture. Use --dataset <dataset_name> --architecture <architecture_name>")
                return
            logger.info(f"Extracting Class Divergence metrics for {args.dataset} and {args.architecture}...")
            metrics = framework.extract_class_divergence_metrics(args.dataset, args.architecture)
            logger.info(f"Extracted {len(metrics)} divergence samples")
        if args.extract_divergence:
            if not args.dataset1 or not args.dataset2:
                logger.info("Divergence Extraction Requires 2 Datasets. Use --dataset1 <dataset1_name> --dataset2 <dataset2_name>")
                return
            if not args.architecture:
                logger.info("Divergence Extraction Requires architecture. Use --architecture <architecture_name>")
                return
            logger.info(f"Extracting Divergence metrics for {args.dataset1} and {args.dataset2}...")
            metrics = framework.extract_divergence_metrics(args.dataset1, args.dataset2, args.architecture)
            logger.info(f"Extracted {len(metrics)} divergence samples")
        # Extract APC metrics if requested
        if args.apc:
            if not args.attack:
                logger.error("APC extraction requires an attack. Use --attack <attack_name>")
                return
            
            logger.info(f"Extracting APC metrics for {args.attack} attack...")
            metrics = framework.extract_apc_metrics(args.dataset, args.architecture, args.attack)
            logger.info(f"Extracted {len(metrics)} APC metric samples")

        # Perform attack if requested
        if args.attack:
            logger.info(f"Preparing {args.attack} attack on {args.dataset} using {args.architecture}...")
            
            # Initialize model and attacker
            _, _, num_classes = DatasetManager.get_datasets(args.dataset)
            model = ModelManager.initialize_model(framework.device, args.dataset, args.architecture)
            attacker = AdversarialAttacker(model, framework.device)
            
            # Check if attack is available
            if args.attack not in attacker.list_available_attacks():
                logger.error(f"Attack '{args.attack}' is not available!")
                logger.info("Available attacks:", attacker.list_available_attacks())
                logger.info("Use --test_attacks to test all attacks or --list_attacks for details")
                return
            
            # Test attack first
            logger.info(f"Testing {args.attack} attack before full execution...")
            if not attacker.test_attack(args.attack, args.dataset, args.architecture):
                logger.error(f"Attack {args.attack} failed initial test!")
                logger.info(f"Use --debug_attack {args.attack} to diagnose the issue")
                return
            
            logger.info(f"✓ Attack test passed. Running {args.attack} on {args.num_samples} samples...")
            
            # Perform the attack
            asr = attacker.perform_attack(args.dataset, args.architecture, args.attack, args.num_samples)
            logger.info(f"Attack completed! Success Rate: {asr:.2f}%")

        # Train detector models
        if args.train_detector:
            logger.info("Training adversarial detector models...")
            
            # Check if APC data exists
            if not os.path.exists('all_adversarial_non_adversarial.csv'):
                logger.error("No APC data found! Run --apc first to extract features.")
                logger.info("Example: python samurai.py --attack pgd --apc --dataset CIFAR10 --architecture resnet18")
                return
            
            detector_trainer = DetectorTrainer()
            X_test, y_test = detector_trainer.train_models()
            
            # Generate SHAP explanations if requested
            if args.explain_shap:
                try:
                    logger.info("Generating SHAP explanations...")
                    X, y = detector_trainer.load_and_prepare_data()
                    
                    label_encoder = detector_trainer.label_encoder
                    y_encoded = label_encoder.transform(y)
                    X_train, X_test_shap, y_train, y_test_shap = train_test_split(
                        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                    )
                    
                    scaler = detector_trainer.scaler
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test_shap)
                    
                    feature_selector = detector_trainer.feature_selector
                    X_test_selected = feature_selector.transform(X_test_scaled)
                    
                    selected_feature_mask = feature_selector.get_support()
                    selected_features = X.columns[selected_feature_mask]
                    
                    logger.info(f"Generating SHAP explanations for {len(selected_features)} selected features")
                    detector_trainer.explain_models_with_shap(X_test_selected, selected_features)
                    
                except Exception as e:
                    logger.error(f"Error generating SHAP explanations: {e}")
                    logger.info("Continuing without SHAP explanations...")

        # Analyze APC divergence if requested
        if args.analyze_divergence:
            if not args.dataset or not args.architecture or not args.attack:
                parser.error("--analyze_divergence requires --dataset, --architecture, and --attack")
            
            logger.info("Analyzing APC divergence between clean and adversarial images...")
            _, _, num_classes = DatasetManager.get_datasets(args.dataset)
            model = ModelManager.initialize_model(framework.device, args.dataset, args.architecture)
            
            clean_dir = f"./{args.dataset.lower()}_{args.architecture}_clean_images"
            adv_dir = f"./{args.dataset.lower()}_{args.architecture}_{args.attack}_images"
            
            if not os.path.exists(clean_dir) or not os.path.exists(adv_dir):
                logger.error(f"Required image directories not found!")
                logger.info(f"Clean dir: {clean_dir}")
                logger.info(f"Adversarial dir: {adv_dir}")
                logger.info(f"Run --attack {args.attack} first to generate images")
                return
            
            analyzer = APCDivergenceAnalyzer(model, framework.device, args.architecture)
            divergence_df = analyzer.analyze_apc_divergence(clean_dir, adv_dir)
            
            if divergence_df is not None:
                logger.info("APC divergence analysis completed successfully!")
            else:
                logger.error("APC divergence analysis failed!")

        # Analyze feature impact
        if args.analyze_features:
            logger.info("Analyzing feature impact on detection performance...")
            
            if not os.path.exists('all_adversarial_non_adversarial.csv'):
                logger.error("No APC data found! Run --apc first to extract features.")
                return
            
            detector_trainer = DetectorTrainer()
            results_df = detector_trainer.analyze_feature_impact()
            logger.info("Feature impact analysis completed")

        # Generate verification metrics if requested
        if args.verification_metrics:
            logger.info("Generating comprehensive verification metrics...")
            
            # Test which attacks are available first
            _, _, num_classes = DatasetManager.get_datasets(args.dataset)
            model = ModelManager.initialize_model(framework.device, args.dataset, args.architecture)
            attacker = AdversarialAttacker(model, framework.device)
            
            # Use available attacks for verification
            available_attacks = attacker.list_available_attacks()
            common_attacks = ['fgsm', 'pgd', 'deepfool', 'cw', 'bim']
            attack_types = [att for att in common_attacks if att in available_attacks]
            
            if not attack_types:
                attack_types = available_attacks[:3]  # Use first 3 available attacks
            
            logger.info(f"Using attacks for verification: {attack_types}")
            
            verification_results = framework.generate_verification_metrics(
                args.dataset, args.architecture, attack_types
            )
            
            logger.info("Verification metrics generation completed!")

        # Evaluate detector
        if args.evaluate_detector:
            logger.info("Evaluating trained detector models...")
            
            try:
                # Check if required files exist
                required_files = ['all_adversarial_non_adversarial.csv', 'scaler.pkl', 
                                'feature_selector.pkl', 'label_encoder.pkl']
                missing_files = [f for f in required_files if not os.path.exists(f)]
                
                if missing_files:
                    logger.error(f"Missing required files: {missing_files}")
                    logger.info("Run --train_detector first to train the models")
                    return
                
                # Load data
                df = pd.read_csv('all_adversarial_non_adversarial.csv')
                
                # Filter for binary classification
                binary_mask = df['Adversarial_or_Non_Adversarial'].isin(['Adversarial', 'Non Adversarial'])
                df_binary = df[binary_mask].copy()
                
                y = df_binary['Adversarial_or_Non_Adversarial']
                X = df_binary.drop(['Adversarial_or_Non_Adversarial', 'Image'], axis=1, errors='ignore')
                
                # Remove divergence columns
                divergence_cols = [col for col in X.columns if 'divergence' in col.lower() or 
                                 'relative_diff' in col or 'absolute_diff' in col or 'Clean_Image' in col]
                X = X.drop(divergence_cols, axis=1, errors='ignore')
                
                # Prepare data
                for col in X.columns:
                    if X[col].dtype == 'object':
                        try:
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                        except:
                            X = X.drop(col, axis=1)
                
                X = X.fillna(0).select_dtypes(include=[np.number])
                
                # Load preprocessors
                scaler = joblib.load('scaler.pkl')
                feature_selector = joblib.load('feature_selector.pkl')
                label_encoder = joblib.load('label_encoder.pkl')
                
                # Encode labels and split data
                y_encoded = label_encoder.transform(y)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                )
                
                # Transform data
                X_test_scaled = scaler.transform(X_test)
                X_test_selected = feature_selector.transform(X_test_scaled)
                
                print("\n" + "="*60)
                print("DETECTOR MODEL EVALUATION RESULTS")
                print("="*60)
                
                # Evaluate traditional ML models
                models_to_evaluate = [
                    ('xgboost_model.pkl', 'XGBoost'),
                    ('random_forest_model.pkl', 'Random Forest'),
                    ('svm_model.pkl', 'SVM'),
                    ('logistic_regression_model.pkl', 'Logistic Regression')
                ]
                
                for model_file, model_name in models_to_evaluate:
                    if os.path.exists(model_file):
                        model = joblib.load(model_file)
                        predictions = model.predict(X_test_selected)
                        
                        accuracy = accuracy_score(y_test, predictions)
                        f1 = f1_score(y_test, predictions, average='weighted')
                        precision = precision_score(y_test, predictions, average='weighted')
                        recall = recall_score(y_test, predictions, average='weighted')
                        
                        print(f"\n{model_name} Results:")
                        print(f"  Accuracy:  {accuracy:.4f}")
                        print(f"  F1 Score:  {f1:.4f}")
                        print(f"  Precision: {precision:.4f}")
                        print(f"  Recall:    {recall:.4f}")
                
                # Evaluate deep learning models
                if os.path.exists('dnn_model.h5'):
                    from tensorflow.keras.models import load_model
                    dnn_model = load_model('dnn_model.h5')
                    dnn_predictions = np.argmax(dnn_model.predict(X_test_selected), axis=1)
                    
                    accuracy = accuracy_score(y_test, dnn_predictions)
                    f1 = f1_score(y_test, dnn_predictions, average='weighted')
                    
                    print(f"\nDeep Neural Network Results:")
                    print(f"  Accuracy:  {accuracy:.4f}")
                    print(f"  F1 Score:  {f1:.4f}")
                
                if os.path.exists('lstm_model.h5'):
                    lstm_model = load_model('lstm_model.h5')
                    X_test_lstm = X_test_selected.reshape((X_test_selected.shape[0], 1, X_test_selected.shape[1]))
                    lstm_predictions = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
                    
                    accuracy = accuracy_score(y_test, lstm_predictions)
                    f1 = f1_score(y_test, lstm_predictions, average='weighted')
                    
                    print(f"\nLSTM Results:")
                    print(f"  Accuracy:  {accuracy:.4f}")
                    print(f"  F1 Score:  {f1:.4f}")
                
                print("\n" + "="*60)
                    
            except Exception as e:
                logger.error(f"Error evaluating detector: {e}")

        # Show completion message
        if any([args.train, args.attack, args.apc, args.train_detector, 
                args.analyze_features, args.analyze_divergence, args.verification_metrics, 
                args.evaluate_detector]):
            print("\n" + "="*60)
            print(" SAMURAI FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Provide next steps suggestions
            if args.train and not args.attack:
                print("Next steps:")
                print("1. Test attacks: --test_attacks --dataset {} --architecture {}".format(args.dataset, args.architecture))
                print("2. Run attack: --attack pgd --dataset {} --architecture {}".format(args.dataset, args.architecture))
            
            if args.attack and not args.apc:
                print("Next steps:")
                print("1. Extract APC: --apc --attack {} --dataset {} --architecture {}".format(args.attack, args.dataset, args.architecture))
            
            if args.apc and not args.train_detector:
                print("Next steps:")
                print("1. Train detector: --train_detector")
                print("2. Evaluate detector: --evaluate_detector")
            
            print("\nAll generated files and results are saved in the current directory.")
            print("="*60)

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Use --debug_attack <attack_name> to debug attack issues")
        logger.info("Use --attack_guide for comprehensive usage instructions")
        raise

if __name__ == '__main__':
    main()

