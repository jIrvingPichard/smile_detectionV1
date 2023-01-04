import keras
from keras.preprocessing.image import load_img

# load the image
img = load_img('dog.jpg')

# find more about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)

# show the image
img.show()
