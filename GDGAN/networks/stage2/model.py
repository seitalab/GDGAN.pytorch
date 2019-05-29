import torch
import torch.nn as nn


class generator(nn.Module):
    def __init__(self, nc, image_size=28, class_num=10):
        super(generator, self).__init__()
        self.image_size = image_size

        self.conv = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * ((self.image_size // 4) ** 2) + class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * ((self.image_size // 4) ** 2)),
            nn.BatchNorm1d(128 * ((self.image_size // 4) ** 2)),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input, label):
        x = self.conv(input)
        x = x.view(-1, 128 * ((self.image_size // 4) ** 2))
        x = torch.cat([x, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    def __init__(self, nc, class_num=10, image_size=28):
        super(discriminator, self).__init__()
        self.image_size = image_size

        self.conv = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * ((self.image_size // 4) ** 2), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, class_num),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * ((self.image_size // 4) ** 2))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c
