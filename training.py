from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.utils import load_img
from keras.preprocessing.image import load_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import os

valid_formats = [".jpg",".jpeg",".png"]

def image_paths(root):
    image_paths = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()          
            if extension in valid_formats:
                image_path = os.path.join(dirpath, filename)
                image_paths.append(image_path)           
    return image_paths

image_paths = image_paths("SMILEs")
IMG_SIZE = (32,32)

def load_dataset(image_paths, target_size = IMG_SIZE):
    
    data = []
    labels = []
    
    for image_path in image_paths:
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)
        image = img_to_array(image)
        data.append(image)

        label = image_path.split(os.path.sep)[-3]
        label = 1 if label == "positives" else 0
        labels.append(label)

    return np.array(data) / 255.0, np.array(labels)
        
data, labels = load_dataset(image_paths)