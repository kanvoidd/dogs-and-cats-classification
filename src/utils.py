import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import JPG_EXTENSION_FILTER

def get_images_with_extension(class_dir: str, extension: str = JPG_EXTENSION_FILTER):
    img_list = [
    img for img in os.listdir(class_dir)
    if img.endswith(extension) and os.path.isfile(os.path.join(class_dir, img))
    ]
    
    return img_list

