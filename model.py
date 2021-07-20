import os
import csv
from itertools import islice
import cv2
import numpy as np
import sklearn
from keras.preprocessing import image as ki
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Lambda, Cropping2D, Dropout, Input, Convolution2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.backend import tf as ktf
from sklearn.utils import shuffle
import matplotlib.image as mpimg

input_shape = (160, 320, 3)
ouput_shape = (160, 320, 1)
normalized_shape = (80, 320, 1)

EPOCHS = 1000
BS = 100

samples = []
with open('.//data//driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader, 1, None):
        samples.append(line)

#split the data set into trainning data and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#define augmented images functions
def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


#define the genrators since the memory of the computer is not enough
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(1):
                #three camera images readed
#                     print (batch_sample[i])
#                     print (len(batch_sample[i]))
                    
#                     if len(batch_sample[i]) != 38 :
#                         continue;
                        
#                     name = './/data//IMG//' + batch_sample[i].split('/')[-1]
                    name = './data/IMG/' + batch_sample[0].split('/')[-1]
#                     print (name)
                    #read BGR format image
                    center_image = cv2.imread(name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
#                     center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2GRAY) 
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
            
                    # add random_flip data into training samples
#                     center_image3, center_angle3 = random_flip(center_image, center_angle)
#                     images.append(center_image3)
#                     angles.append(center_angle3)
            
#                     name = './data/IMG/' + batch_sample[1].split('/')[-1]
# #                     print (name)
#                     center_image = cv2.imread(name)
#                     center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
#                     center_angle = float(batch_sample[3]) + 0.2

#                     name = './data/IMG/' + batch_sample[2].split('/')[-1]
# #                     print (name)
#                     center_image = cv2.imread(name)
#                     center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
#                     center_angle = float(batch_sample[3]) - 0.2
            
                    #change samples into array
                    X_train = np.array(images)
                    y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BS)
validation_generator = generator(validation_samples, batch_size=BS)

def color2gray(images):
#     images = images.astype(float)
#     return np.dot(images, [[0.2989],[0.5870],[0.1140]])
#     arr = ki.img_to_array(images)
#     np.dot(arr, [[0.2989],[0.5870],[0.1140]])
#     return ki.array_to_img(arr)
#     print (images.ndim)
    coff = ktf.constant([[0.2989],[0.5870],[0.1140]])
#     print (images.shape)
#     print (images.dtype)
    return K.dot(images, coff)


def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (32, 100))


# Keras model refered to NVIDIA model. But I added Cropping2D and made some of changes.
model = Sequential()
#cropping the images to remove the noise from image, Keras onyl need to feed input shape in fisrt layer.
model.add(Cropping2D(cropping = ((65, 20), (2,2)), input_shape = (160, 320, 3)))
#resize the image data to accelerate the traning step.
model.add(Lambda(resize_img))
#normalize the data
model.add(Lambda(lambda x: x/127.5 - 1))
#convolution layer with 12 filters(size (5,5)), stride = (2,2)
model.add(Convolution2D(12, 5, 5, activation='relu', subsample=(2,2)))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#convolution layer with 32 filters(size (5,5)), stride = (2,2)
model.add(Convolution2D(32, 5, 5, activation='relu', subsample=(2,2)))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#convolution layer with 48 filters(size (5,5)), stride = (2,2)
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
#flatten the layer into fully connected layer
model.add(Flatten())
#add dense layer , and the number of node become 100
model.add(Dense(100, activation='relu'))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#add dense layer , and the number of node become 50
model.add(Dense(50, activation='relu'))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#add dense layer , and the number of node become 10
model.add(Dense(10, activation='relu'))
#add dense layer , and the number of node become 1
model.add(Dense(1))
model.summary()

#train the model with generator
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
                    steps_per_epoch = 1, \
                    validation_data = validation_generator, \
                    validation_steps = 1, \
                    epochs = EPOCHS)

#save model as model.h5
model.save('model.h5')
