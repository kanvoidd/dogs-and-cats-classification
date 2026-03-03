import kagglehub

import random
import shutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import CATS_DIRECTORY, DOGS_DIRECTORY


def import_data(test=0.2):
    # Don't download the dataset if it has already been downloaded.
    cached_path = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "shaunthesheep", "microsoft-catsvsdogs-dataset", "versions", "1")
    if os.path.exists(cached_path):
        path = cached_path
        print("Using cached dataset:", path)
    else:
        path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
        print("Downloaded dataset to:", path)


    os.makedirs("dataset/train/cats", exist_ok=True)
    os.makedirs("dataset/train/dogs", exist_ok=True)
    os.makedirs("dataset/test/cats", exist_ok=True)
    os.makedirs("dataset/test/dogs", exist_ok=True)


    cats_imgs = os.listdir(path + "\\" + os.path.normpath(CATS_DIRECTORY))
    dogs_imgs = os.listdir(path + "\\" + os.path.normpath(DOGS_DIRECTORY))
    random.shuffle(cats_imgs)
    random.shuffle(dogs_imgs)


    # Splitting cats images into train and test sets
    cats_split_idx = int(len(cats_imgs) * (1 - test))
    train_cats = cats_imgs[:cats_split_idx]
    test_cats = cats_imgs[cats_split_idx:]

    # Splitting dogs images into train and test sets
    dogs_split_idx = int(len(dogs_imgs) * (1 - test))
    train_dogs = dogs_imgs[:dogs_split_idx]
    test_dogs = dogs_imgs[dogs_split_idx:]


    # Distributing cats and dogs images into train and test folders
    for img in train_cats:
        src = os.path.join(path, os.path.normpath(CATS_DIRECTORY), img)
        dst = os.path.join("dataset/train/cats", img)

        if os.path.getsize(src) > 0:
            shutil.copy(src, dst)

    for img in test_cats:
        src = os.path.join(path, os.path.normpath(CATS_DIRECTORY), img)
        dst = os.path.join("dataset/test/cats", img)

        if os.path.getsize(src) > 0:
            shutil.copy(src, dst)
    
    for img in train_dogs:
        src = os.path.join(path, DOGS_DIRECTORY, img)
        dst = os.path.join("dataset/train/dogs", img)

        if os.path.getsize(src) > 0:
            shutil.copy(src, dst)

    for img in test_dogs:
        src = os.path.join(path, DOGS_DIRECTORY, img)
        dst = os.path.join("dataset/test/dogs", img)

        if os.path.getsize(src) > 0:
            shutil.copy(src, dst)


if __name__ == "__main__":
    import_data()