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
        print(dirnames)

image_paths("SMILEs")