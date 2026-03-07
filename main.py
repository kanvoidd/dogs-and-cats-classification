import os
import src.config as config
from src.dataset import CatDogDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from src.model import CatDogCNN
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from src.train import train_catdog_classifier

def main():
    img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])

    train_dataset = CatDogDataset(config.TRAIN_DIRECTORY, transform=img_transform)
    val_dataset = CatDogDataset(config.VAL_DIRECTORY, transform=img_transform)


    train_loader = DataLoader(train_dataset, 
                              batch_size=32, 
                              shuffle=True, 
                              num_workers=2, 
                              pin_memory=True
    )

    val_loader = DataLoader(val_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=2, 
                            pin_memory=True
    )

    # Using GPU for more faster model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    history = train_catdog_classifier(model, train_loader, val_loader, criterion, optimizer, device, epochs=20)
    
if __name__ == "__main__":
    main()