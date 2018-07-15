from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPool2D
from keras.layers.advanced_activations import ReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import tensorflow as tf
import numpy as np
import cv2


def get_training_data():
    data_dirs = ['/home/adarsh/software/udacity/bc_data/driving_log.csv',
                 '/home/adarsh/software/udacity/bc_data/data/driving_log.csv']

    image_paths = []
    angles = []

    for data_dir in data_dirs:
        csvfile = open(data_dir, 'r')
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',')
        reader = list(reader)
        for idx, item in enumerate(reader):
            if idx == 0:
                continue
            image_paths.append(item[0])
            angles.append(float(item[3]))
            image_paths.append(item[1])
            angles.append(float(item[3]) + 0.25)
            image_paths.append(item[2])
            angles.append(float(item[3]) - 0.25)

    image_paths = np.array(image_paths)
    angles = np.array(angles)

    return image_paths, angles


def preprocess(image):
    img = cv2.GaussianBlur(image, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    return img


def training_data_generator(paths, angles, validation=False):
    paths, angles = shuffle(paths, angles)

    batch_size = 256




model = Sequential()

model.add(Conv2D(24, (5, 5), strides=(1,1), padding='valid', activation=ReLU, kernel_regularizer=l2(0.001)))

model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(BatchNormalization())

model.add(Conv2D(36, (5, 5), strides=(1,1), padding='valid', activation=ReLU, kernel_regularizer=l2(0.001)))

model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(BatchNormalization())

model.add(Conv2D(48, (5, 5), strides=(1,1), padding='valid', activation=ReLU, kernel_regularizer=l2(0.001)))

model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(BatchNormalization())

model.add(Dropout(0.50))

model.add(Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation=ReLU, kernel_regularizer=l2(0.001)))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation=ReLU, kernel_regularizer=l2(0.001)))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.50))

model.add(Dense(100, activation=ReLU, activity_regularizer=l2(0.001)))

model.add(Dropout(0.50))

model.add(Dense(50, activation=ReLU, activity_regularizer=l2(0.001)))

model.add(Dropout(0.50))

model.add(Dense(10, activation=ReLU, activity_regularizer=l2(0.001)))

model.add(Dense(1))

model.compile(optimizer=Adam(lr=1e-4), loss='mse')

image_paths, angles = get_training_data()

image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles, test_size=0.10)

image_paths_train, image_paths_valid, angles_train, angles_valid = train_test_split(image_paths_train, angles_train, test_size=0.10)

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

# history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040,
#                               nb_epoch=5, verbose=2, callbacks=[checkpoint])

model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)