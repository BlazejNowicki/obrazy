import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
        )
        self.first_relu = nn.LeakyReLU(0.02)

        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(num_features=64),
        )
        self.second_relu = nn.LeakyReLU(0.02)
        
        self.third_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(num_features=64),
        )
        self.third_relu = nn.LeakyReLU(0.2)

        self.fully_connected = nn.Sequential(
            nn.Flatten(0, -1),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
        )


    def forward(self, x):
        x = self.first_conv(x)
        x = nn.functional.normalize(x, p=2, eps=0.01)
        x = self.first_relu(x)

        x = self.second_conv(x)
        x = nn.functional.normalize(x, p=2, eps=0.01)
        x = self.second_relu(x)

        x = self.third_conv(x)
        x = nn.functional.normalize(x, p=2, eps=0.01)
        x = self.third_relu(x)
    
        x = self.fully_connected(x)

        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 1024),
            nn.Unflatten(0, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(256, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()

        )

    def forward(self, x):
        return self.model(x)



# img = Image.open("crawled_cakes/001_3f1c1895.jpg")
# img.resize(size=(32, 32))
# img

# dataset[0].unsqueeze(0)
# toPIL = transforms.ToPILImage()

# toPIL((dataset[3] + 1) / 2)


# x = dataset[0].unsqueeze(0)
# des = Descriminator()
# y = des(x)
# print(y)

# noise = torch.randn(64)
# generator = Generator()
# noise
# out = generator(noise)
# out.shape
# %%
