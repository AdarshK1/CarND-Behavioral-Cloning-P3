import math
import random

from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPool2D
from keras.layers.advanced_activations import ReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, BaseLogger, TensorBoard, ProgbarLogger

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import tensorflow as tf
import numpy as np
import cv2


def get_training_data():
    data_dirs = ['/home/adarsh/software/udacity/bc_data/data/driving_log.csv', '/home/adarsh/software/udacity/bc_data/driving_log.csv', ]

    image_paths = []
    angles = []

    for idx1, data_dir in enumerate(data_dirs):
        csvfile = open(data_dir, 'r')
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',')
        reader = list(reader)
        for idx2, item in enumerate(reader):
            if idx2 == 0:
                continue
            if idx1 == 0:
                image_paths.append(("/home/adarsh/software/udacity/bc_data/data/" + item[0]))
                image_paths.append(("/home/adarsh/software/udacity/bc_data/data/" + item[1]))
                image_paths.append(("/home/adarsh/software/udacity/bc_data/data/" + item[2]))
            else:
                image_paths.append(item[0])
                image_paths.append(item[1])
                image_paths.append(item[2])

            angles.append(float(item[3]))
            angles.append(float(item[3]) + 0.25)
            angles.append(float(item[3]) - 0.25)

    image_paths = np.array(image_paths)
    angles = np.array(angles)

    return image_paths, angles


def preprocess(image):
    img = cv2.GaussianBlur(image, (3, 3), 0)
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_LINEAR)

    return img



def random_distort(image):
    if random.random() < 0.5:
        height, width, channels = image.shape
        horiz = 2 * height / 5
        shift = np.random.randint(-height / 10, height / 10)
        points1 = np.float32([[0, horiz], [width, horiz], [0, height], [width, height]])
        points2 = np.float32([[0, horiz + shift], [width, horiz + shift], [0, height], [width, height]])
        mat = cv2.getPerspectiveTransform(points1, points2)
        image = cv2.warpPerspective(image, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return image.astype(np.uint8)


def training_data_generator(paths, angles, validation=False):
    paths, angles = shuffle(paths, angles)

    batch_size = 128

    x, y = ([], [])

    while True:
        for j in range(len(angles)):
            img = cv2.imread(paths[j])
            angle = angles[j]
            img = preprocess(img)

            if not validation:
                img = random_distort(img)
                x.append(img)
                y.append(angle)

                if np.abs(angle) > 0.40:
                    flipped = cv2.flip(img, 1)
                    angle *= -1
                    x.append(flipped)
                    y.append(angle)

            x.append(img)
            y.append(angle)

            if random.random() < 0.5:
                flipped = cv2.flip(img, 1)
                angle *= -1
                x.append(flipped)
                y.append(angle)

            if len(y) == batch_size:
                yield (np.array(x), np.array(y))
                x, y = ([], [])


image_paths, angles = get_training_data()

image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles, test_size=0.10)

image_paths_train, image_paths_valid, angles_train, angles_valid = train_test_split(image_paths_train, angles_train, test_size=0.10)

training_gen = training_data_generator(image_paths_train, angles_train, validation=False)
validation_gen = training_data_generator(image_paths_valid, angles_valid, validation=True)
testing_gen = training_data_generator(image_paths_test, angles_test, validation=True)

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66, 200, 3)))

model.add(Conv2D(24, (5, 5), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(0.001)))

model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(BatchNormalization())

model.add(Conv2D(36, (5, 5), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(0.001)))

model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(BatchNormalization())

model.add(Conv2D(48, (5, 5), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(0.001)))

model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

model.add(BatchNormalization())

# model.add(Dropout(0.50))

model.add(Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation='relu', kernel_regularizer=l2(0.001)))

model.add(BatchNormalization())

model.add(Flatten())

# model.add(Dropout(0.25))

model.add(Dense(100, activation='relu', activity_regularizer=l2(0.001)))

model.add(Dropout(0.25))

model.add(Dense(50, activation='relu', activity_regularizer=l2(0.001)))

model.add(Dropout(0.25))

model.add(Dense(10, activation='relu', activity_regularizer=l2(0.001)))

model.add(Dense(1))

tb = TensorBoard(log_dir="tb_logs/", histogram_freq=1, write_graph=True, write_images=True)


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('./models/model_2_{epoch:02d}.h5')

logger = BaseLogger(stateful_metrics=None)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        print("began")

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        print('batch ended', logs)
        print(logs.get('loss'))

losshist = LossHistory()

progbar = ProgbarLogger(count_mode="samples", stateful_metrics=None)

history = model.fit_generator(training_gen, validation_data=validation_gen, validation_steps=2560, steps_per_epoch=23040, epochs=10, verbose=2, callbacks=[checkpoint, logger, tb, losshist, progbar])

model.save_weights('./models/model_2_.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)