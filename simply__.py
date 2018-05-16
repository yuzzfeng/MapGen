# -*- coding: utf-8 -*-

##  ROAD NETWORK EXTRACTION USING CNN  ##
##  SERCAN CAKIR MASTER'S THESIS       ##
##  SUBMISSION DATE: 02.11.2017        ##

# Import libraries
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os
import numpy as np
from numpy import random

np.random.seed(7)
import keras
from keras.models import Sequential
from keras.callbacks import History
from keras.layers.core import Dropout
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import backend as K


# function to read .tif image files
from osgeo import gdal


def readImg(img):
    # Read heatmap
    image = gdal.Open(img)
    # Get band of heatmap, it's gray scale image!
    img_band = image.GetRasterBand(1)
    # Read the image as array
    image = img_band.ReadAsArray()
    # Normalize the pixel values in the range 0-1 acc. to max. normalization
    image = (image - image.min()) / (image.max() - image.min())
    return image.astype('float32')


# Load Strava heat map and target road for Hannover ####

# function to create image patches
from skimage.util.shape import view_as_windows
import numpy as np


def imagePatches(img1, p_w, p_h, stride):
    img_1 = view_as_windows(img1, (p_w, p_h), stride)
    a, b, h, w = img_1.shape
    img_1_1 = np.reshape(img_1, (a * b, p_w, p_h))
    return img_1_1





# functions to remove fully black images from heatmap and target data, and all the correspondences
def removeBlackImg(img_patch):
    patch_list = []
    patch_list_new = []
    for i in range(len(img_patch)):
        patch_list.append(img_patch[i])
        if patch_list[i].max() != 0:
            patch_list_new.append(img_patch[i])
    return patch_list_new


def removeCorrespondence(road, heat):  # remove roads if heats are black
    patch_road_list = []
    patch_heat_list = []
    patch_road_list_new = []
    for i in range(len(road)):
        patch_road_list.append(road[i])
        patch_heat_list.append(heat[i])
        if patch_heat_list[i].max() != 0:
            patch_road_list_new.append(road[i])
    return patch_road_list_new



from sklearn.model_selection import train_test_split



##### Building the CNN archıtecture with "Sequential Model" (model looks like autoencoder)
def create_model(optimizer, input_shape):
    model = Sequential()
    ### Encoding (down-sampling) ###
    model.add(Conv2D(filters=24, kernel_size=(5, 5),
                     strides=(2, 2), padding='same',
                     activation='relu', input_shape=input_shape, kernel_initializer='random_uniform',
                     name="down_conv_1"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2"))
    model.add(Dropout(0.1))
    ###############################################################################
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', name="down_conv_2"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_3"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_4"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_5"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_6"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_7"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8"))
    model.add(Dropout(0.1))
    ###############################################################################
    model.add(UpSampling2D(size=(2, 2), name='up_samp_1'))

    model.add(Conv2D(filters=64, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_1"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_9"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_10"))
    model.add(Dropout(0.1))
    ###############################################################################
    model.add(UpSampling2D(size=(2, 2), name='up_samp_2'))

    model.add(Conv2D(filters=24, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_2"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=12, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_11"))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=1, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='sigmoid', name="flat_conv_12"))
    # model.add(Activation(our_activation))
    model.add(Dropout(0.1))

    # Compile model with Adam optimizer and binary cross entropy loss function
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


# model_ex1.summary()

##################################################################################################################################
#class Covariance(keras.callbacks.Callback):
#
#    def on_train_begin(self, logs={}):
#        self.avg_cov1 = []
#        self.avg_cov2 = []
#
#    def on_epoch_end(self, epoch, logs={}):
#        x_val1, y_true1 = x_train_sim, y_train_sim
#        y_pred1 = self.model.predict(x_val1)
#
#        x_val2, y_true2 = x_test_sim, y_test_sim
#        y_pred2 = self.model.predict(x_val2)
#
#        cov_1 = []
#        max_cov1 = []
#        for i in range(len(y_pred1)):
#            y_pred_clip1 = np.reshape(y_pred1[i][1:(len(y_pred1[1]) - 1), 1:(len(y_pred1[1]) - 1)],
#                                      ((len(y_pred1[1]) - 2) ** 2, 1))  # (len(y_pred1)-1)=31
#            y_true_1 = imagePatches(np.reshape(y_true1[i], (len(y_pred1[1]), len(y_pred1[1]))), (len(y_pred1[1]) - 2),
#                                    (len(y_pred1[1]) - 2), 1)
#            for k in range(0, 9):
#                cov_1.append(sum((y_pred_clip1 - y_pred_clip1.mean()) * (
#                            np.reshape(y_true_1[k], ((len(y_pred1[1]) - 2) ** 2, 1)) - (
#                        np.reshape(y_true_1[k], ((len(y_pred1[1]) - 2) ** 2, 1)).mean()))) / len(
#                    y_pred_clip1))  # 900
#        for j in range(0, 9 * len(y_pred1), 9):
#            max_cov1.append(max(cov_1[j:j + 9]))
#
#        cov_2 = []
#        max_cov2 = []
#        for i in range(len(y_pred2)):
#            y_pred_clip2 = np.reshape(y_pred2[i][1:(len(y_pred2[1]) - 1), 1:(len(y_pred2[1]) - 1)],
#                                      ((len(y_pred2[1]) - 2) ** 2, 1))  # (len(y_pred1)-1)=31
#            y_true_2 = imagePatches(np.reshape(y_true2[i], (len(y_pred2[1]), len(y_pred2[1]))), (len(y_pred2[1]) - 2),
#                                    (len(y_pred2[1]) - 2), 1)
#            for k in range(0, 9):
#                cov_2.append(sum((y_pred_clip2 - y_pred_clip2.mean()) * (
#                            np.reshape(y_true_2[k], ((len(y_pred2[1]) - 2) ** 2, 1)) - (
#                        np.reshape(y_true_2[k], ((len(y_pred2[1]) - 2) ** 2, 1)).mean()))) / len(
#                    y_pred_clip2))  # 900
#        for j in range(0, 9 * len(y_pred2), 9):
#            max_cov2.append(max(cov_2[j:j + 9]))
#        print('\n avg_cov: {} - val_avg_cov: {}\n'.format(np.mean(max_cov1), np.mean(max_cov2)))
#
#        self.avg_cov1.append(np.mean(max_cov1))
#        self.avg_cov2.append(np.mean(max_cov2))


##################################################################################################################################
class LearningRateTracker(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.lr_list = []

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        # lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        lr = K.eval(
            optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay)))))
        print('\n LR: {}\n'.format(lr))
        self.lr_list.append(lr)


##################################################################################################################################
class SaveWeights(keras.callbacks.Callback):  # Saves weights after each 25 epochs
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 49 == 0:
            model_json = self.model.to_json()
            with open("model_" + str(epoch) + ".json", "w") as json_file:
                json_file.write(model_json)
            self.model.save_weights("weights_model_" + str(epoch) + ".h5")
            print("Saved model-weights to disk")


##################################################################################################################################


# Order the image dimension acc. to TensorFlow (batc_hsize, rows, cols, channels)
K.set_image_dim_ordering('tf')

# set the working directory
#os.chdir(r'F:\sercan\input_images')
PATH = os.getcwd()
#plt.gray()
#plt.show()


# load simulated heat map (TRAJECTORY SIMULATION) and target road for Hannover ####
sim_heatmap_hannover = readImg(r"input2.tif")
sim_road_hannover = readImg(r"output2.tif")

p_size_1 = 128
# hm: heatmap
sim_hm_patches_32 = imagePatches(sim_heatmap_hannover, p_size_1, p_size_1, p_size_1)
sim_road_patches_32 = imagePatches(sim_road_hannover, p_size_1, p_size_1, p_size_1)


#p_size_2 = 200
## hm: heatmap
#sim_hm_patches_200 = imagePatches(sim_heatmap_hannover, p_size_2, p_size_2, p_size_2)
#sim_road_patches_200 = imagePatches(sim_road_hannover, p_size_2, p_size_2, p_size_2)
#plt.imshow(np.reshape(sim_hm_patches_200[41], (200,200)))
#plt.show()

sim_road_patches_32_new = removeCorrespondence(sim_road_patches_32, sim_hm_patches_32)
sim_hm_patches_32_new = removeCorrespondence(sim_hm_patches_32, sim_road_patches_32)
sim_road_patches_32_new_new = removeBlackImg(sim_road_patches_32)


####
#sim_road_patches_200_new = removeCorrespondence(sim_road_patches_200, sim_hm_patches_200)
#sim_hm_patches_200_new = removeCorrespondence(sim_hm_patches_200, sim_road_patches_200)
#sim_road_patches_200_new_new = removeBlackImg(sim_road_patches_200)


### experience 1 - simulated hm
index_list_sim = list(range(len(sim_hm_patches_32_new)))
random.shuffle(index_list_sim)

idx_sim = 100
index_list_test_sim = index_list_sim[-idx_sim:]
index_list_test_sim.sort()
sim_hm_test = [sim_hm_patches_32_new[i] for i in index_list_test_sim]
sim_road_test = [sim_road_patches_32_new_new[i] for i in index_list_test_sim]

index_list_train_sim = index_list_sim[:-idx_sim]
index_list_train_sim.sort()
sim_hm_train = [sim_hm_patches_32_new[i] for i in index_list_train_sim]
sim_road_train = [sim_road_patches_32_new_new[i] for i in index_list_train_sim]

x_train_sim = np.reshape(sim_hm_train, (len(sim_hm_train), p_size_1, p_size_1, 1))
y_train_sim = np.reshape(sim_road_train, (len(sim_road_train), p_size_1, p_size_1, 1))
x_test_sim = np.reshape(sim_hm_test, (len(sim_hm_test), p_size_1, p_size_1, 1))
y_test_sim = np.reshape(sim_road_test, (len(sim_road_test), p_size_1, p_size_1, 1))

# save image patch arrays
np.save("x_train_sim.npy", x_train_sim)
np.save("y_train_sim.npy", y_train_sim)
np.save("x_test_sim.npy", x_test_sim)
np.save("y_test_sim.npy", y_test_sim)

# plt.imshow(np.reshape(x_test_sim[2], (p_size_1,p_size_1)))
# plt.imshow(np.reshape(y_test_sim[2], (p_size_1,p_size_1)))



input_shape1 = x_train_sim[0].shape

opt1 = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

model_ex1 = create_model(opt1, input_shape1)


##### Train the model
#covariance1 = Covariance()
History1 = History()
hist1 = model_ex1.fit(x_train_sim, y_train_sim,
                      batch_size=16,
                      epochs=500,
                      verbose=1,
                      shuffle=True,
                      callbacks=[History1],
                      validation_data=(x_test_sim, y_test_sim))

### Save history
History1_loss = History1.history['loss']
History1_acc = History1.history['acc']
History1_val_loss = History1.history['val_loss']
History1_val_acc = History1.history['val_acc']

thefile1 = open('History1_loss.txt', 'w')
for item in History1_loss:
    thefile1.write("%s\n" % item)

thefile2 = open('History1_acc.txt', 'w')
for item in History1_acc:
    thefile2.write("%s\n" % item)

thefile3 = open('History1_val_loss.txt', 'w')
for item in History1_val_loss:
    thefile3.write("%s\n" % item)

thefile4 = open('History1_val_acc.txt', 'w')
for item in History1_val_acc:
    thefile4.write("%s\n" % item)

### Save model
model_json1 = model_ex1.to_json()
with open("model_ex1.json", "w") as json_file:
    json_file.write(model_json1)
model_ex1.save_weights("weights_model_ex1.h5")
print("Saved model to disk")

################################################################


#### Plot history of average covariance - accuracy and loss of the models
#plt.plot(History1.history['loss'])
#plt.plot(History1.history['val_loss'])
#plt.title('loss & val_loss')
#plt.legend(['train', 'test'], loc='upper right')
#plt.plot(History1.history['acc'])
#plt.plot(History1.history['val_acc'])
#plt.title('acc & val_acc')
#plt.legend(['train', 'test'], loc='upper right')
#
#plt.plot(covariance1.avg_cov1)
#plt.plot(covariance1.avg_cov2)
#plt.title('average covariance')
#plt.legend(['train', 'test'], loc='lower left')
#
#plt.show()

## Test trained models on test images


image_arr = readImg(r"testexampleinput2.tif")

b = 128 #32  # patch size
image_patches_1 = view_as_windows(image_arr, (b, b), b)

###### Image reconstruction from patches ######
test_image_1 = []
for i in range(image_patches_1.shape[1]):
    test_image_1.append(image_patches_1[0, i])

test_image_all = []
for i in range(image_patches_1.shape[1]):
    for k in range(image_patches_1.shape[0]):
        test_image_all.append(np.reshape(image_patches_1[k, i], (b, b)))

test_image_pred_all = []
for i in range(image_patches_1.shape[0] * image_patches_1.shape[1]):
    # change the model upon request
    test_image_pred_all.append(np.reshape(model_ex1.predict(np.reshape(test_image_all[i], (1, b, b, 1))), (b, b)))
    print(i)

conc1 = []
for m in range(0, image_patches_1.shape[0] * image_patches_1.shape[1], image_patches_1.shape[0]):
    conc1.append(np.concatenate((test_image_pred_all[m:m + image_patches_1.shape[0]]), axis=0))

for n in range(image_patches_1.shape[0]):
    conc2 = (np.concatenate((conc1), axis=1))

fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(conc2)
#ax.imshow(conc2, aspect='normal')
fig.savefig("strava_pred", dpi=1000)


##### function to calculate evaluation parameters (F1-Score, Precision, Recall) ######
def evaluation(model, x_test, y_test, patch_size):
    precision = []
    recall = []
    f1Score = []
    import math
    for k in range(len(x_test_sim)):
        y_pred = model.predict(x_test_sim[k:k + 1])
        y_pred = np.reshape(y_pred, (32 * 32))

        y_true = y_test_sim[k:k + 1]
        y_true = np.reshape(y_true, (32 * 32))

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        y_pred = np.round(y_pred)
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == 1:
                TP += 1
            elif y_pred[i] == y_true[i] == 0:
                TN += 1
            elif y_pred[i] == 1 and y_true[i] != y_pred[i]:
                FP += 1
            elif y_pred[i] == 0 and y_true[i] != y_pred[i]:
                FN += 1

        precision.append(TP / (TP + FP + K.epsilon()))  # completeness
        recall.append(TP / (TP + FN))  # correctness
        beta = 1
        f1Score.append((math.pow(beta, 2) + 1) * TP / ((math.pow(beta, 2) + 1) * TP + math.pow(beta, 2) * FN + FP))
        # eval_list = [precision,  recall, f1Score]

    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(precision)
    avg_f1score = sum(f1Score) / len(precision)
    avg_eval_param = [avg_precision, avg_recall, avg_f1score]
    return avg_eval_param


#  evaluation(model_ex1, x_test_sim, y_test_sim, 32)


# function to load a saved model
def LoadModel(model_json):
    from keras.models import model_from_json
    json_file = open(model_json)
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model