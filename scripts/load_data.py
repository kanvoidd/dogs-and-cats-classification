import kagglehub

from typing import Literal
import random
import shutil
import os
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import src.config as config


def distribute_data(
        path: str,
        images_list: list,
        subset_type: Literal["train", "test"],
        class_type: Literal["cats", "dogs"],
        class_directory: str
):
    """
    path: path to the initial dataset
    images_list: list of files for copying
    subset_type: 'train' or 'test'
    class_type: 'cats' or 'dogs'
    """
    valid_count = 0

    for img in images_list:
        src = os.path.join(path, class_directory, img)
        dst = os.path.join("dataset", subset_type, class_type, img)

        if os.path.getsize(src) == 0:
            print(f"Scipping empty file: {src}")
            continue

        try:
            with Image.open(dst) as test_img:
                test_img.verify()
        except Exception as e:
            print(f"Scipping corrupted file {src}: {e}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        valid_count += 1

    print(f"{subset_type} {class_type}: copied {valid_count} valid images (skipped {len(images_list) - valid_count})")

def get_list_of_images(path: str, class_directory: str):
    class_imgs = [
    img for img in os.listdir(os.path.join(path, class_directory))
    if img.endswith(config.JPG_EXTENSION_FILTER) and os.path.isfile(os.path.join(path, class_directory, img))
    ]

    return class_imgs

def train_val_test_split(image_list: list, train_ratio, val_ratio):
    train_end = int(len(image_list) * train_ratio)
    val_end = int(len(image_list) * (train_ratio + val_ratio))

    train_set = image_list[:train_end]
    val_set = image_list[train_end:val_end]
    test_set = image_list[val_end:]

    return train_set, val_set, test_set

def import_data(train=0.7):
    path = kagglehub.dataset_download(config.DATASET_PATH)
    print("Downloaded dataset to:", path)

    # Creating target folders
    for dir_path in [config.TRAIN_CATS_DIRECTORY, config.TRAIN_DOGS_DIRECTORY,
                     config.VAL_CATS_DIRECTORY, config.VAL_DOGS_DIRECTORY,
                     config.TEST_CATS_DIRECTORY, config.TEST_DOGS_DIRECTORY]:
        os.makedirs(dir_path, exist_ok=True)


    # Collecting lists of images
    cats_imgs = get_list_of_images(path, config.CATS_DIRECTORY)
    dogs_imgs = get_list_of_images(path, config.DOGS_DIRECTORY)

    random.shuffle(cats_imgs)
    random.shuffle(dogs_imgs)

    val = (1 - train) / 2
    train_cats, val_cats, test_cats = train_val_test_split(cats_imgs, train, val)
    train_dogs, val_dogs, test_dogs = train_val_test_split(dogs_imgs, train, val)

    # Distributing into the folders
    distribute_data(path, train_cats, "train", "cats", config.CATS_DIRECTORY)
    distribute_data(path, test_cats, "test", "cats", config.CATS_DIRECTORY)
    distribute_data(path, val_cats, "val", "cats", config.CATS_DIRECTORY)
    distribute_data(path, train_dogs, "train", "dogs", config.DOGS_DIRECTORY)
    distribute_data(path, test_dogs, "test", "dogs", config.DOGS_DIRECTORY)
    distribute_data(path, val_dogs, "val", "dogs", config.DOGS_DIRECTORY)

    print(f"\nThe Image sets were successfully loaded!")

if __name__ == "__main__":
    import_data()