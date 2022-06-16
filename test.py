from numpy import random
from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
tfds.disable_progress_bar()
tf.get_logger().setLevel('INFO')

train_path = 'ds/train' #path for training images
valid_path = 'ds/valid' #path for validation images
test_path = 'ds/test' #path for testing images

#process and scale images and put them in batches with set classes, batches sized 10 images each
#batches contain images and their labels
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'not cat'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'not cat'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'not cat'], batch_size=10, shuffle=False)

#show processed images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#test model that was saved before
def testLoadedModel(test_batch, loaded_model):
    loaded_model.summary()
    loss, acc = loaded_model.evaluate(test_batch, verbose=2)
    print("Tested model, accuracy: {:5.2f}%".format(100 * acc))

def predictRandomImage(loaded_model):
    index = np.random.randint(4000,4599)

    #load and convert image to RGB
    single_sample = cv2.imread('ds/single_testing/'+str(index)+'.jpg')
    single_sample = cv2.cvtColor(single_sample, cv2.COLOR_BGR2RGB)

    #show image
    plt.imshow(single_sample)
    plt.axis('off')
    plt.show()

    #preprocess image
    single_sample = tf.keras.applications.vgg16.preprocess_input(single_sample)
    single_sample = cv2.resize(single_sample, (224,224))
    single_sample = np.array(single_sample).reshape((1,224,224,3))

    #prediction
    prediction = loaded_model.predict(single_sample)
    if prediction[0][0]<prediction[0][1]:
        accuracy= prediction[0][1]*100
        formatted_float = "{:.2f}".format(accuracy)
        print('not cat '+ formatted_float+'%')
    else :
        accuracy= prediction[0][0]*100
        formatted_float = "{:.2f}".format(accuracy)
        print('cat '+ formatted_float +'%')
        

loaded_model = keras.models.load_model('./my_model_current')

testLoadedModel(test_batches, loaded_model)

predictRandomImage(loaded_model)

imgs, labels = next(train_batches)

plotImages(imgs)
#print labels
print(labels)

#training model

#model convolutional build
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'same'),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.25),
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding = 'same'),
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(units=2, activation='softmax')
])

#model summary
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)

#test the trained model
loss, acc = model.evaluate(test_batches, verbose=2)
print("Tested model, accuracy: {:5.2f}%".format(100 * acc))

#save created model
model.save('./my_model_current')

