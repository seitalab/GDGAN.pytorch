import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.densenet121 = (torchvision
                            .models
                            .densenet121(pretrained=pretrained))
        num_kernel = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.densenet121(x)


class DenseNet169(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNet169, self).__init__()
        self.densenet169 = (torchvision
                            .models
                            .densenet169(pretrained=pretrained))
        num_kernel = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.densenet169(x)


class DenseNet201(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNet201, self).__init__()
        self.densenet201 = (torchvision
                            .models
                            .densenet201(pretrained=pretrained))
        num_kernel = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.densenet201(x)
