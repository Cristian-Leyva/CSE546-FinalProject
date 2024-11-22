import cv2
import os
import matplotlib.pyplot as plt

IMAGE_DIR = 'data/images'

def load_image(image_name, image_class):
    """ Load image by name and class """
    image_path = os.path.join(IMAGE_DIR, image_class, image_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load {image_path} correctly.")
            return None
        return image
    else:
        print(f"Image {image_name} not found in {image_path}")
        return None
    

def show_image(image):
    """ Show image and optionally save to disk"""
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()