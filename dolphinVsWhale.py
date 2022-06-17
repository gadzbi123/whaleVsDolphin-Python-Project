from numpy import random
from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os 

from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import cv2


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tfds.disable_progress_bar()
tf.get_logger().setLevel('INFO')


train_path = 'ds/train'  # path for training images
valid_path = 'ds/valid'  # path for validation images
test_path = 'ds/test'  # path for testing images

# process and scale images and put them in batches with set classes, batches sized 10 images each
# batches contain images and their labels
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['dolphin', 'whale'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['dolphin', 'whale'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['dolphin', 'whale'], batch_size=10,
                         shuffle=False)

def testModel(model):
    loss, acc = model.evaluate(test_batches, verbose=2)
    return ("Tested model, accuracy: {:5.2f}%".format(100 * acc))

def testImage(loadedModel,imagePath):

    # load and convert image to RGB
    single_sample = cv2.imread(imagePath)
    single_sample = cv2.cvtColor(single_sample, cv2.COLOR_BGR2RGB)

    # preprocess image
    single_sample = tf.keras.applications.vgg16.preprocess_input(single_sample)
    single_sample = cv2.resize(single_sample, (224, 224))
    single_sample = np.array(single_sample).reshape((1, 224, 224, 3))

    # prediction
    prediction = loadedModel.predict(single_sample)
    predictionResult = ''
    if prediction[0][0] < prediction[0][1]:
        accuracy = prediction[0][1] * 100
        formatted_float = "{:.2f}".format(accuracy)
        predictionResult = 'whale ' + formatted_float + '%'
    else:
        accuracy = prediction[0][0] * 100
        formatted_float = "{:.2f}".format(accuracy)
        predictionResult = 'dolphin ' + formatted_float + '%'
    return predictionResult

def loadModel(path):
    loaded_model = keras.models.load_model(path)
    return loaded_model

# making training model
def createModel():
    # model convolutional build
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),  # 112
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),  # 56
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),  # 28
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Dropout(0.25),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),  # 14
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Dropout(0.25),  # makes sure not to overtrain
        Flatten(),  # 1d array
        Dense(units=2, activation='relu'),
        Dense(units=2, activation='softmax')
    ])

    #compile a model (takes about 20 min)
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    #validate model with valid_vatches
    model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=10,
            verbose=2
            )

    # save created model
    model.save('./models/current_model')

    return model

createModel()