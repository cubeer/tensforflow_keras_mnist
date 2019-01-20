# -*- encoding: utf-8 -*-
'''
@File    :   tf.keras.py
@Author  :   cubeer.com
@Desc    :   利用CNN模型，进行衣服分类和识别
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib.pylab as plt
import time

batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28


def read_data_sets(folder):
    _mnist = input_data.read_data_sets(folder, one_hot=True)
    return _mnist


fashion_mnist = read_data_sets("./fashion-mnist/data/fashion/")

# 准备数据
x_train, y_train = fashion_mnist.train.images, fashion_mnist.train.labels
x_test, y_test = fashion_mnist.test.images, fashion_mnist.test.labels
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train /= 255
x_test /= 255
print('input_shape:', input_shape)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print('-------------------------------')

# 构建卷积神经网络
model = Sequential()
''' 
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        kernel_initializer='he_normal',
        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))  #0.5
model.add(Dense(num_classes, activation='softmax'))
'''
model.add(
    Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='relu',
        kernel_initializer='he_normal',
        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adadelta(),
    loss=tf.keras.metrics.categorical_crossentropy,
    metrics=['accuracy'])

model.summary()
model.save('fashion_mnist.h5')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('-------------------------------')


# plot
class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[history])

#validation accuracy and loss, from the train history.
score = model.evaluate(x_test, y_test, verbose=0)

# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
