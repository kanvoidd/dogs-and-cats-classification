import os
from torchvision.io import decode_image
from torchvision.transforms import ConvertImageDtype
from torch.utils.data import Dataset, DataLoader
from src.utils import get_images_with_extension
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class CatDogDataset(Dataset):
    def __init__(self, imgs_dir, transform=None, target_transform=None):
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.labels = []

        try:
            for label, class_name in enumerate(['cats', 'dogs']):
                class_dir = os.path.join(imgs_dir, class_name)

                for img_name in get_images_with_extension(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)
                    
        except Exception as e:
            print(f"Folders scanning error: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
