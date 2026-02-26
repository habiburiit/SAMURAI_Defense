# =============================================================================
# SAMURAI Framework - Model Accuracy Checker
# Usage: CUDA_VISIBLE_DEVICES=0 python check_accuracy.py \
#            --dataset CIFAR10 --architecture resnet18
# =============================================================================

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      type=str, default='CIFAR10')
parser.add_argument('--architecture', type=str, default='resnet18')
parser.add_argument('--gpu',          type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Dataset ──────────────────────────────────────────────────────────────────
dataset_map = {
    'CIFAR10':  (torchvision.datasets.CIFAR10,  10),
    'CIFAR100': (torchvision.datasets.CIFAR100, 100),
    'MNIST':    (torchvision.datasets.MNIST,    10),
    'SVHN':     (torchvision.datasets.SVHN,     10),
    'STL10':    (torchvision.datasets.STL10,    10),
}

dataset_class, num_classes = dataset_map[args.dataset]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(3) if args.dataset == 'MNIST' else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if args.dataset in ('SVHN', 'STL10'):
    testset = dataset_class(root='./data', split='test', download=True, transform=transform)
else:
    testset = dataset_class(root='./data', train=False, download=True, transform=transform)

testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# ─── Model ────────────────────────────────────────────────────────────────────
model_map = {
    'resnet18':   models.resnet18,
    'resnet34':   models.resnet34,
    'resnet50':   models.resnet50,
    'resnet101':  models.resnet101,
    'resnet152':  models.resnet152,
    'vgg16':      models.vgg16,
    'vgg19':      models.vgg19,
    'alexnet':    models.alexnet,
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'mobilenet_v2': models.mobilenet_v2,
}

model_fn = model_map[args.architecture]
model = model_fn(weights=None)

# Adjust final layer
if 'resnet' in args.architecture:
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif args.architecture in ('vgg16', 'vgg19', 'alexnet'):
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
elif 'densenet' in args.architecture:
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
elif args.architecture == 'mobilenet_v2':
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model_path = f'./{args.dataset.lower()}_{args.architecture}.pth'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ─── Evaluate ─────────────────────────────────────────────────────────────────
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs  = model(images)
        _, preds = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n{'='*40}")
print(f"  Dataset      : {args.dataset}")
print(f"  Architecture : {args.architecture}")
print(f"  Model Path   : {model_path}")
print(f"{'='*40}")
print(f"  Accuracy     : {accuracy:.2f}%")
print(f"  Correct      : {correct}/{total}")
print(f"{'='*40}")
