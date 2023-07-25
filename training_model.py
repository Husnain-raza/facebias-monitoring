## run on colab to directl download data from kaggle (dont forget uploading your kaggle.json file)
!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d 'jessicali9530/celeba-dataset'

!unzip /content/celeba-dataset.zip
!pip install aif360


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# The dataset path
DATA_PATH = ""

# Load attribute data (sample of 5000 images to keep the model small)
data = pd.read_csv(DATA_PATH + "list_attr_celeba.csv").sample(5000).reset_index(drop=True)


def load_images(data, path):
    image_array = []
    for i in range(len(data)):
        img = cv2.imread(path + data['image_id'][i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color space
        img = cv2.resize(img, (64, 64))  # resize images to make it smaller
        image_array.append(img)
    return np.array(image_array)

images = load_images(data, DATA_PATH + "img_align_celeba/img_align_celeba/")

# Normalize pixel values between 0 and 1
images = images / 255.0

# Convert attribute values into 0 and 1 (originally -1 and 1)
data.replace(to_replace=-1, value=0, inplace=True)

# Define target - for gender bias
target = data['Male'].values


# Split the data
x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=0.2, random_state=42)


# Load the ResNet-50 model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

# Create a new model on top of the output of one (or several) layers from the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Freeze the base model
base_model.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stopping])


# Saving the data
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

# Loading the data
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
