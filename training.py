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
IMG_SIZE = [32,32]

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


def build_model(input_shape = IMG_SIZE + [1]):
    model = Sequential([
        Conv2D(filters=32,
               kernel_size=(3,3),
               activation="relu",
               padding="same",
               input_shape=input_shape),
        MaxPool2D(2,2),
        Conv2D(filters=64,
               kernel_size=(3,3),
               activation="relu",
               padding="same",
               input_shape=input_shape),
        MaxPool2D(2,2),
        Flatten(),
        Dense(256,activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

labels, counts = np.unique(labels,return_counts=True)

counts = max(counts)/counts
class_weights = dict(zip(labels,counts))

(X_train, X_test, y_train, y_test) = train_test_split(data, labels,
                                                      test_size=0.2,
                                                      stratify=labels,
                                                      random_state=42)

(X_train, X_valid, y_train, y_valid) = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      stratify=y_train,
                                                      random_state=42) 

model = build_model()
EPOCHS = 20
history = model.fit(X_train, y_train,
                    validation_data=(X_valid,y_valid),
                    class_weight=class_weights,
                    batch_size=64,
                    epochs=EPOCHS)
model.save("model")