import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import CATS_DIRECTORY, DOGS_DIRECTORY, TRAIN_CATS_DIRECTORY, TEST_CATS_DIRECTORY, TRAIN_DOGS_DIRECTORY, TEST_DOGS_DIRECTORY

img = cv2.imread(os.path.join("dataset", "train", "cats", "1.jpg"), cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def standartify_sizes_of_images(images_list: list, width: int, height: int, class_subset_dir: str):
    for img_name in images_list:
        img_path = os.path.join(class_subset_dir, img_name)

        img = cv2.imread(img_path)

        if width < img.shape[1]:
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(img_path, resized_img)