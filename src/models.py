import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SmallCNN(nn.Module):
    """你当前那版：三层 conv+bn+pool + GAP"""
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class SmallCNNv2(nn.Module):
    """每个 block 两层卷积再池化"""
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # block1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # block2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # block3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class ResNet18Genre(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        # 3通道改1通道
        w = base.conv1.weight.data
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = w.mean(dim=1, keepdim=True)
        base.fc = nn.Linear(base.fc.in_features, n_classes)
        self.net = base

    def forward(self, x):
        return self.net(x)


def build_model(model_type: str, n_classes: int):
    if model_type == "small_cnn":
        return SmallCNN(n_classes)
    if model_type == "small_cnn_v2":
        return SmallCNNv2(n_classes)
    if model_type == "resnet18":
        return ResNet18Genre(n_classes)
    raise ValueError(f"Unknown model_type: {model_type}")
