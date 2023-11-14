from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import  torchvision.transforms as transforms


class CakeDataset(Dataset):
    def __init__(self, data_dir: str):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.images = []

        for file in os.listdir(data_dir):
            img = Image.open(data_dir + file)
            img = img.resize((32, 32))
            img = transform(img) # [0, 1]
            img = 2 * img - 1 # [-1, 1]
            self.images.append(img)

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> Image:
        return self.images[index]