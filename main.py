import os
from src.config import TRAIN_DIRECTORY, TEST_DIRECTORY, IMAGE_MEAN, IMAGE_STD
from src.dataset import CatDogDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

def main():
    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

    train_dataset = CatDogDataset(TRAIN_DIRECTORY, transform=img_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[1].squeeze()
    # label = train_labels[1]

    # img = img.permute(1, 2, 0)
    # plt.imshow(img)
    # plt.show()
    # print(f"Label: {label}")
    
if __name__ == "__main__":
    main()