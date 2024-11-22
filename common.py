import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def load_data():
    """ Load all data """
    df1 = pd.read_csv('data/Data.csv')
    df2 = pd.read_csv('data/extra_hard_samples.csv')
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    return df

def split_data(df):
    """ Split data into X/y training and testing sets from `amount` of total data """
    X = df.drop(['image_name', 'class'], axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test