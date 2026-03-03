import cv2
import torch
from torchvision import transforms
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import CATS_DIRECTORY, DOGS_DIRECTORY, TRAIN_CATS_DIRECTORY, TEST_CATS_DIRECTORY, TRAIN_DOGS_DIRECTORY, TEST_DOGS_DIRECTORY, JPG_EXTENSION_FILTER

def standardize_size_of_image(img_path: str, width: int, height: int):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Неудалось загрузить изображение {img_path}")
        return None

    if width < img.shape[1]:
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    else:
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    return resized_img

def to_tensor(img):
    transform = transforms.ToTensor()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb)

    return img_tensor

def normalize_image(img_tensor):
    mean = img_tensor.mean()
    std = img_tensor.std()
    normalized_tensor = (img_tensor - mean) / std

    return normalized_tensor

def preprocess_images(class_subset_dir: str, width: int, height: int):
    img_list = [
        img for img in os.listdir(class_subset_dir)
        if img.endswith(JPG_EXTENSION_FILTER) and os.path.isfile(class_subset_dir, img)
    ]

    img_path = os.path.join(class_subset_dir, img)

    img = standardize_size_of_image(img_path, width, height)
    img = to_tensor(img)
    img = normalize_image(img)

    