# %%
from dataset import CakeDataset
dataset = CakeDataset('crawled_cakes/')

# %%
import torch
from torch import nn
from PIL import Image

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from models import Descriminator

torch.manual_seed(111)

assert torch.cuda.is_available(), "Ten manewr by nas kosztował 10 lat"
device = "cuda"


img = Image.open("crawled_cakes/001_3f1c1895.jpg")
img.resize(size=(32, 32))


# %%
x = dataset[0].unsqueeze(0)
des = Descriminator()
y = des(x)
print(y.shape)
y
# %%
