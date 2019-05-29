import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        num_kernel = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet34, self).__init__()
        self.resnet34 = torchvision.models.resnet34(pretrained=pretrained)
        num_kernel = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.resnet34(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=pretrained)
        num_kernel = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.resnet50(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=pretrained)
        num_kernel = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.resnet101(x)


class ResNet152(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=pretrained)
        num_kernel = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.resnet152(x)
