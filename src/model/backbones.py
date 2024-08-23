from typing import Callable, Tuple

import torch.nn as nn

from torchvision.models import MobileNetV3, ResNet, Inception3, EfficientNet, DenseNet, VGG
from torchvision.models import mobilenet_v3_large, resnet50, inception_v3, efficientnet_b0, densenet121, vgg16
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.inception import Inception_V3_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.models.vgg import VGG16_Weights

BackboneGetter = Callable[[], Tuple[nn.Module, int]]


def get_mobilenet_v3_backbone() -> Tuple[MobileNetV3, int]:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, in_features


def get_resnet_50_backbone() -> Tuple[ResNet, int]:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


def get_inception3_backbone() -> Tuple[Inception3, int]:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


def get_efficientnet_b0_backbone() -> Tuple[EfficientNet, int]:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Identity()
    return model, in_features


def get_densenet121_backbone() -> Tuple[DenseNet, int]:
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Identity()
    return model, in_features


def get_vgg16_backbone() -> Tuple[VGG, int]:
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Identity()
    return model, in_features


backbone_getters: dict[str, BackboneGetter] = {
    "mobilenet-v3": get_mobilenet_v3_backbone,
    "resnet50": get_resnet_50_backbone,
    "inception3": get_inception3_backbone,
    "efficientnet": get_efficientnet_b0_backbone,
    "densenet121": get_densenet121_backbone,
    "vgg16": get_vgg16_backbone
}