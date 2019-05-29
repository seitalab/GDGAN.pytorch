import torch.nn as nn
import torchvision


class VGG11(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG11, self).__init__()
        self.vgg11 = torchvision.models.vgg11(pretrained=pretrained)
        num_kernel = self.vgg11.classifier[-1].in_features
        self.vgg11.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg11(x)


class VGG11_bn(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG11_bn, self).__init__()
        self.vgg11_bn = torchvision.models.vgg11_bn(pretrained=pretrained)
        num_kernel = self.vgg11_bn.classifier[-1].in_features
        self.vgg11_bn.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg11_bn(x)


class VGG13(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG13, self).__init__()
        self.vgg13 = torchvision.models.vgg13(pretrained=pretrained)
        num_kernel = self.vgg13.classifier[-1].in_features
        self.vgg13.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg13(x)


class VGG13_bn(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG13_bn, self).__init__()
        self.vgg13_bn = torchvision.models.vgg13_bn(pretrained=pretrained)
        num_kernel = self.vgg13_bn.classifier[-1].in_features
        self.vgg13_bn.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg13_bn(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        num_kernel = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg16(x)


class VGG16_bn(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG16_bn, self).__init__()
        self.vgg16_bn = torchvision.models.vgg16_bn(pretrained=pretrained)
        num_kernel = self.vgg16_bn.classifier[-1].in_features
        self.vgg16_bn.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg16_bn(x)


class VGG19(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=pretrained)
        num_kernel = self.vgg19.classifier[-1].in_features
        self.vgg19.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg19(x)


class VGG19_bn(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(VGG19_bn, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(pretrained=pretrained)
        num_kernel = self.vgg19_bn.classifier[-1].in_features
        self.vgg19_bn.classifier[-1] = nn.Linear(num_kernel, num_classes)

    def forward(self, x):
        return self.vgg19_bn(x)
