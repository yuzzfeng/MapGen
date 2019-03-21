'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
#import matplotlib
#import matplotlib.pyplot as plt
#
#import os
#import numpy as np
#import pandas as pd
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#
#from keras import models
#from keras.models import load_model
#from keras.layers import Input
#from keras.models import Model
#
#from data_helper import readImg
#
#def update_gan_generator_to_any_size(old_model):
#    # Remove the top layer and add input with no limit
#    old_model.layers.pop(0) 
#    newInput = Input(shape=(None, None, 1)) # New image input
#    newOutputs = old_model(newInput)
#    newModel = Model(newInput, newOutputs)
#
#    return newModel
#
#def IoUcheck(img_input, img_output):
#    # Pixelwise IoU score
#    logic_and = np.sum(np.logical_and(img_output, img_input))
#    logic_or = np.sum(np.logical_or(img_output, img_input))
#    return logic_and/logic_or
#
#def rescaleImg(image_arr):
#    # Rescale the image to 8 x n
#    if image_arr.shape[0] % 8 != 0:
#        n = image_arr.shape[0] % 8
#        new_x = image_arr.shape[0] - n
#    else:
#        new_x = image_arr.shape[0]
#
#    if image_arr.shape[1] % 8 != 0:
#        n = image_arr.shape[1] % 8
#        new_y = image_arr.shape[1] - n
#    else:
#        new_y = image_arr.shape[1]
#
#    image_arr = image_arr[:new_x, :new_y]
#
#    return image_arr
#
#def save_prediction(img, out_path, out_filename, subfix):
#    fig = plt.figure(figsize=(img.shape[1] / 1000, img.shape[0] / 1000), dpi=100, frameon=False)
#    ax = plt.Axes(fig, [0., 0., 1., 1.])
#    ax.set_axis_off()
#    fig.add_axes(ax)
#
#    plt.imshow(img, cmap='gray')
#    fig.savefig(out_path + out_filename + subfix, dpi=1000)
#    
#def evaluate(image_arrA, image_arrB):
#    
#    target_names = ['0', '1']
#    
#    y_true = image_arrB.flatten().astype(bool) 
#    y_pred = image_arrA.flatten().astype(bool)
#        
#    Accuracy = accuracy_score(y_true, y_pred)
#    IntOverUnion = IoUcheck(y_true, y_pred)
#    conf = confusion_matrix(y_true, y_pred)
#    report = classification_report(y_true, y_pred, target_names=target_names)
#        
#    print('Acc:', Accuracy)
#    print('Error:', 1 - Accuracy)
#    print('IoU:', IntOverUnion)
#    print(conf)
#    #print(report)
#
#    return Accuracy, IntOverUnion, conf, report
#
#def model_predict(newModel, input_image, num_runs):
#    # Predict with model for n times
#    m,n = input_image.shape
#    for i in range(num_runs):
#        input_image = np.reshape(input_image, (1, m, n, 1))
#        input_image = newModel.predict([input_image])
#        input_image = np.reshape(input_image,(m, n)) > 0.5
#    return input_image
#
#def predict_eval(SavedModel, img_range, testPath, fn_input, fn_target, out_path, scale, nr = 1):
#
#    image_arr = readImg(testPath + fn_input)
#    image_tar = readImg(testPath + fn_target)
#    
#    if len(img_range) == 4: # If range was set
#        xmin, xmax, ymin, ymax = img_range
#        image_arr = image_arr[xmin:xmax, ymin:ymax]
#        image_tar = image_tar[xmin:xmax, ymin:ymax]
#    
#    print("- " + str(scale) +"k", 'Example: ', image_arr.shape)
#    evals_orig = evaluate(image_arr, image_tar)
#    
#    image_arr = rescaleImg(image_arr)
#    image_tar = rescaleImg(image_tar)
#    
#    print("+ " + str(scale) +"k", 'Prediction: ', image_arr.shape)
#    pred = model_predict(SavedModel, image_arr, num_runs = nr)
#    evals_pred = evaluate(pred > 0.5, image_tar)
#    
#    save_prediction(pred, out_path, fn_input[:-4], '_' + str(scale) + '_' + str(nr) + '_out.png')
#    return [evals_orig, evals_pred]
#
#def predict_only(SavedModel, img_range, testPath, fn_input, out_path, scale, nr = 1):
#
#    image_arr = readImg(testPath + fn_input)
#    
#    if len(img_range) == 4: # If range was set
#        xmin, xmax, ymin, ymax = img_range
#        image_arr = image_arr[xmin:xmax, ymin:ymax]
#    
#    image_arr = rescaleImg(image_arr)
#    
#    print("+ " + str(scale) +"k", 'Prediction: ', image_arr.shape)
#    pred = model_predict(SavedModel, image_arr, num_runs = nr)
#    
#    save_prediction(pred > 0.5, out_path, fn_input[:-4], '_' + str(scale) + '_' + str(nr) + '_out.png')
#    
#path = "Y:/yu/"
#
#if 1: # 256 Input r
#    scale = 15
#    modelPath = path + "tmp_results/predictions/2019-03-09 13-18-43_15/"
#    out_evaluation = path + "tmp_results/Evaluations/Runet_15k_256/"
#    
#    modelname = "weights.h5"
#    saved_model = models.load_model(modelPath + modelname)
#    saved_model = update_gan_generator_to_any_size(saved_model)
#
#if 1:    
#    tester_path = path + "tmp_data/Data/Testing_large/4270/"
#    all_records = []
#    records = predict_eval(saved_model, [0,2400,500,2900], tester_path, r"geb_clip_4270.png", r"geb"+str(scale)+"_clip_4270.png", out_evaluation, scale, nr = 1)
#    all_records.extend(records)