# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
## Imports
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

# Training parameters
batch_size = 128
num_classes = 10
epochs = 1

# Data preparation

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

Index_train = []
Index_test = []
for i in range(0,6):
    temp = np.random.choice(len(x_train), len(x_train), replace = False)
    temp2 = np.random.choice(len(x_test), len(x_test), replace = False)
    Index_train.append(temp)
    Index_test.append(temp2)
    
    
xtrain =  np.array([x_train[Index_train[0]], x_train[Index_train[1]], x_train[Index_train[2]], x_train[Index_train[3]], x_train[Index_train[4]], x_train[Index_train[5]]])
ytrain =  np.array([y_train[Index_train[0]], y_train[Index_train[1]], y_train[Index_train[2]], y_train[Index_train[3]], y_train[Index_train[4]], y_train[Index_train[5]]])

xtest =  np.array([x_test[Index_test[0]], x_test[Index_test[1]], x_test[Index_test[2]], x_test[Index_test[3]], x_test[Index_test[4]], x_test[Index_test[5]]])
ytest =  np.array([y_test[Index_test[0]], y_test[Index_test[1]], y_test[Index_test[2]], y_test[Index_test[3]], y_test[Index_test[4]], y_test[Index_test[5]]])

x_train = np.swapaxes(xtrain,0,1)
x_test = np.swapaxes(xtest,0,1)



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train_cnn = np.zeros((6,60000,10))
# convert class vectors to binary class matrices
y_test_cnn = np.zeros((6,10000,10))
for i in range(0,6):
    y_train_cnn[i] = keras.utils.to_categorical(ytrain[i], num_classes)
    y_test_cnn[i] = keras.utils.to_categorical(ytest[i], num_classes)


y_train = np.swapaxes(y_train_cnn,0,1)
y_test = np.swapaxes(y_test_cnn,0,1)

# Model definition
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


sequences_input = Input(shape=(6,28,28,1))
process_seq = TimeDistributed(model)(sequences_input)

model = Model(input=[sequences_input], output=process_seq)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Training loop

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





