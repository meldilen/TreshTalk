import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights) # уже обучена на ImageNet, нужно дообучить только на наших классах
    model.fc = nn.Linear(model.fc.in_features, num_classes) # последний слой для нужного числа классов
    return model
