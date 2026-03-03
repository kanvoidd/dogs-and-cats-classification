from src.dataset import preprocess_images
from data.load_data import import_data
import os
from src.config import TRAIN_CATS_DIRECTORY, TEST_CATS_DIRECTORY, TRAIN_DOGS_DIRECTORY, TEST_DOGS_DIRECTORY

def main():
    preprocess_images(TRAIN_CATS_DIRECTORY, 128, 128)
    
if __name__ == "__main__":
    