import kagglehub

from typing import Literal
import random
import shutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import CATS_DIRECTORY, DOGS_DIRECTORY, JPG_EXTENSION_FILTER


def distribute_data(
        path: str,
        images_list: list,
        subset_type: Literal["train", "test"],
        class_type: Literal["cats", "dogs"],
        class_directory: str
):
    """
    path: путь к исходному датасету
    images_list: список файлов для копирования
    subset_type: 'train' или 'test'
    class_type: 'cats' или 'dogs'
    """
    for img in images_list:
        src = os.path.join(path, class_directory, img)
        dst = os.path.join("dataset", subset_type, class_type, img)

        if os.path.getsize(src) > 0:
            shutil.copy(src, dst)


def import_data(test=0.2):
    path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
    print("Downloaded dataset to:", path)


    os.makedirs("dataset/train/cats", exist_ok=True)
    os.makedirs("dataset/train/dogs", exist_ok=True)
    os.makedirs("dataset/test/cats", exist_ok=True)
    os.makedirs("dataset/test/dogs", exist_ok=True)


    cats_imgs = [
        img for img in os.listdir(os.path.join(path, CATS_DIRECTORY))
        if img.endswith(JPG_EXTENSION_FILTER) and os.path.isfile(os.path.join(path, CATS_DIRECTORY, img))
    ]

    dogs_imgs = [
                img for img in os.listdir(os.path.join(path, DOGS_DIRECTORY))
        if img.endswith(JPG_EXTENSION_FILTER) and os.path.isfile(os.path.join(path, DOGS_DIRECTORY, img))
    ]

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
    distribute_data(path, train_cats, "train", "cats", CATS_DIRECTORY)
    distribute_data(path, test_cats, "test", "cats", CATS_DIRECTORY)
    distribute_data(path, train_dogs, "train", "dogs", DOGS_DIRECTORY)
    distribute_data(path, test_dogs, "test", "dogs", DOGS_DIRECTORY)


if __name__ == "__main__":
    import_data()