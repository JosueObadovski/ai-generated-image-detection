import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):

    # Carrega ResNet18 pré-treinada
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Substitui a última camada (fully connected)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model