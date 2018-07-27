# start auf keko: nohub python3 <file.py> > train.log & bg
# um es im hintergrund laufen zu lassen 
# coding: utf-8

# In[1]:




## Map Generalization for Polygons using Autoencode-like strucutures
## Adatped based on Master Thesis of SERCAN CAKIR "ROAD NETWORK EXTRACTION USING CNN"
## Author: Yu Feng, yuzz.feng@gmail.com
## 1. Version Author: SERCAN CAKIR

## Changes:
## 1. Two conv layers were added before the first down convlusional layer
## 2. Output can be any size during the evaluation

import matplotlib
matplotlib.use('Agg') # necessary for linux kernal
#get_ipython().magic(u'matplotlib inline')
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
from keras.layers import MaxPooling2D, Conv2DTranspose
from keras import backend as K
import tensorflow as tf

from osgeo import gdal
from sklearn.model_selection import train_test_split
from skimage.util.shape import view_as_windows

from PIL import Image
import time

from time import gmtime, strftime
timestr = strftime("%Y-%m-%d %H-%M-%S", gmtime())

# function to read .tif image files
def readImg(img):
    # Read heatmap
    image = gdal.Open(img)
    #print("imabe in readimg : ",image)
    # Get band of heatmap, it's gray scale image!
    img_band = image.GetRasterBand(1)
    # Read the image as array
    image = img_band.ReadAsArray()
    # Normalize the pixel values in the range 0-1 acc. to max. normalization
    image = (image - image.min()) / (image.max() - image.min())
    return image.astype('float32')


# function to create image patches
def imagePatches(img1, p_w, p_h, stride):
    img_1 = view_as_windows(img1, (p_w, p_h), stride)
    a, b, h, w = img_1.shape
    img_1_1 = np.reshape(img_1, (a * b, p_w, p_h))
    return img_1_1

# functions to remove fully black images from heatmap and target data, and all the correspondences
# remove white == 255
def removeBlackImg(img_patch):
    patch_list = []
    patch_list_new = []
    for i in range(len(img_patch)):
        patch_list.append(img_patch[i])
        if i == 2:
            print("patch i: nur weiss oder schwarz ?",patch_list[i].max(), patch_list[i].min())
        if patch_list[i].max() != patch_list[i].min():
            patch_list_new.append(img_patch[i])
    return patch_list_new

# remove roads if heats are black
def removeCorrespondence(road, heat):  
    patch_road_list = []
    patch_heat_list = []
    patch_road_list_new = []
    for i in range(len(road)):
        patch_road_list.append(road[i])
        patch_heat_list.append(heat[i])
        if patch_heat_list[i].max() != 255:
            patch_road_list_new.append(road[i])
    return patch_road_list_new
# remove roads if heats are black

# remove correspondence, if one of both patches contains only one Grayvalue
def removeCorrespondenceAll(road, heat):  
    print("len road / heat: ", len(road), len(heat))
    patch_road_list = []
    patch_heat_list = []
    for i in range(len(road)):
        #patch_road_list.append(road[i])
        #patch_heat_list.append(heat[i])
        if road[i].max() != road[i].min():
            if heat[i].max() != heat[i].min():
                #print("appending image patches heat and road",i)
                patch_road_list.append(road[i])
                patch_heat_list.append(heat[i])
    return patch_road_list, patch_heat_list

# from yu
def check_and_create(out_dir):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)

def prediction_independent(model_ex1, image_arr):
    
    conc2 = np.reshape(model_ex1.predict(np.reshape(image_arr, (1, image_arr.shape[0], image_arr.shape[1], 1))), 
                   (image_arr.shape[0], image_arr.shape[1]))
    return conc2

# cut the image to avoid shape error
def cut_image(image_arr):
    
    print("Original:", image_arr.shape)
    
    if image_arr.shape[0] % 4 != 0:
        n = image_arr.shape[0] % 4
        new_x = image_arr.shape[0] - n
    else:
        new_x = image_arr.shape[0]

    if image_arr.shape[1] % 4 != 0:
        n = image_arr.shape[1] % 4
        new_y = image_arr.shape[1] - n
    else:
        new_y = image_arr.shape[1]
    
    image_arr = image_arr[:new_x, :new_y]
    print("Clipped:", image_arr.shape)

from sklearn.metrics import accuracy_score
def testIndependet(fn, inpath, outpath):
    print("testin on image ",fn)
    image_arr = readImg(inpath + fn)
    print(image_arr.shape)
    
    if image_arr.shape[0] % 4 != 0:
        n = image_arr.shape[0] % 4
        new_x = image_arr.shape[0] - n
    else:
        new_x = image_arr.shape[0]


    if image_arr.shape[1] % 4 != 0:
        n = image_arr.shape[1] % 4
        new_y = image_arr.shape[1] - n
    else:
        new_y = image_arr.shape[1]

    image_arr = image_arr[:new_x, :new_y]
    print("image arr")
    print(image_arr.shape)
    
    conc2 = np.reshape(model_ex1.predict(np.reshape(image_arr, (1, image_arr.shape[0], image_arr.shape[1], 1))), 
                       (image_arr.shape[0], image_arr.shape[1]))
    
    print("accuracy score: ")
    print(accuracy_score(image_arr.flatten().astype(bool), (conc2 > 0.5).flatten()))
    
    fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    plt.imshow(conc2)
    fig.savefig(outpath + fn[:-4] + '_out.png', dpi=1000)
    
    
    # print image in bw (threshold 0.5)
    fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    conc2 = conc2 > 0.5
    plt.imshow(conc2, cmap='gray')
    fig.savefig(outpath + fn[:-4] + '_out_bw.png', dpi=1000)
    print("all images drawn and printed",outpath + fn[:-4] + '_out_bw.png')


############ Path Setting ##############

trainPath = r"./Data/geb10/"
#trainPath = r"./Data/Training_Validation/" 
testPath = r"./Data/Testing/" 
#tmpPath = r"./Data/tmp_data2/"
tmpPath = r"./Data/tmp_data/"


outPath = r"Prediction/"
check_and_create(outPath)
check_and_create(outPath + timestr)
outPath = outPath + timestr + "/"


# In[2]:


# function to load a saved model
def LoadModel(model_json):
    from keras.models import model_from_json
    json_file = open(model_json)
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


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


# In[3]:

_EPSILON = 10e-8

def IoU(yTrue,yPred):  
    I = tf.multiply(yTrue, yPred, name="intersection")
    U = yTrue + yPred - I + _EPSILON
    
    IoU = tf.reduce_sum(I) / tf.reduce_sum(U)
    return -tf.log(IoU + _EPSILON) 

#In das model brauchen Sie nur das loss name ändern, z.B. 

# Compile model with Adam optimizer and binary cross entropy loss function
#Standard Loss: 

#model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics=['acc'])

#Custom Loss: 
#model.compile(optimizer=optimizer, loss=IoU, metrics=['acc'])



##### Building the CNN archıtecture with "Sequential Model" 
##### (model looks like autoencoder)
def create_model(optimizer, input_shape):
    model = Sequential()
    
    droprate = 0.3

    model.add(Conv2D(filters=24, kernel_size=(5, 5),
              strides=(1, 1), padding='same',
              activation='relu', input_shape=input_shape, kernel_initializer='random_uniform',
              name="flat_conv_a"))
    model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=24, kernel_size=(5, 5),
              strides=(1, 1), padding='same',
              activation='relu',name="flat_conv_b"))
    model.add(Dropout(droprate))
    
#    model.add(Conv2D(filters=24, kernel_size=(3, 3),
#              strides=(1, 1), padding='same',
#              activation='relu',name="flat_conv_c"))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#    model.add(Dropout(droprate))
    
    ## Encoding (down-sampling) ###   
    model.add(Conv2D(filters=24, kernel_size=(5, 5),
                     strides=(2, 2), padding='same',
                     activation='relu', #input_shape=input_shape, kernel_initializer='random_uniform',
                     name="down_conv_1"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2"))
    model.add(Dropout(droprate))
    ##############################################################################
    
#    model.add(Conv2D(filters=24, kernel_size=(3, 3),
#              strides=(1, 1), padding='same',
#              activation='relu',name="down_conv_2"))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#    model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', name="down_conv_2"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_3"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_4"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_5"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_6"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_7"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8"))
    model.add(Dropout(droprate))
    ###############################################################################
    model.add(UpSampling2D(size=(2, 2), name='up_samp_1'))
    
#    model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), 
#                              padding='same', activation='softmax'))
    

    model.add(Conv2D(filters=64, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_1"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_9"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_10"))
    model.add(Dropout(droprate))
    ###############################################################################
    model.add(UpSampling2D(size=(2, 2), name='up_samp_2'))
    
#    model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), # Lead the accuracy to 0.78
#                              padding='same', activation='softmax'))

    model.add(Conv2D(filters=24, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_2"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=12, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_11"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=1, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='sigmoid', name="flat_conv_12"))
    # model.add(Activation(our_activation))
    model.add(Dropout(droprate))

    # Compile model with Adam optimizer and binary cross entropy loss function
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['acc'])
    #model.compile(optimizer=optimizer, loss=IoU, metrics=['acc'])

    return model

#model.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics=['acc'])

#Custom Loss: 
#model.compile(optimizer=optimizer, loss=IoU, metrics=['acc'])


#model_ex1.summary()


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


# In[14]:


# Order the image dimension acc. to TensorFlow (batc_hsize, rows, cols, channels)
K.set_image_dim_ordering('tf')

# set the working directory
#os.chdir(r'F:\sercan\input_images')
PATH = os.getcwd()
#plt.gray()
#plt.show()

# load simulated heat map (TRAJECTORY SIMULATION) and target road for Hannover ####
#sim_heatmap_hannover = readImg(r"input2.tif")
#sim_road_hannover = readImg(r"output2.tif")

print("trainPath: ",trainPath)
#2traininput_inv.png


def generate_input(First):
    # image_input  = trainPath + r"2traininput_inv.png"
    # image_output = trainPath + r"2trainoutput_inv.png"
    # typify
    ##image_input  = trainPath + r"input2.tif"
    ##image_output = trainPath + r"output2.tif"
    p_size_1 = 128

    if First: 
        # geb25 - target scale 1:25.000
        image_input  = trainPath + r"geb.png"
        image_output = trainPath + r"geb10.png"
        #image_input  = trainPath + r"gebtif2_in.tif"
        #image_output = trainPath + r"gebtif2_out.tif"
    
        
        print("output image: ",image_output)
        print("image input: ",image_input)
        
        #fns_input = [trainPath + r"input2.tif"]
        #fns_output = [trainPath + r"output2.tif"]
        
        sim_heatmap_hannover = readImg(image_input)
        sim_road_hannover = readImg(image_output)
        
        #sim_heatmap_hannover = readImg("Data/Training_Validation/geb1_inp_inv.tif")
        #sim_road_hannover = readImg("Data/Training_Validation/geb1_out_inv.tif")
        
        # p_size_1 = 128
        
        # create overlapping training data
    #    sim_hm_patches_overlap = imagePatches(sim_heatmap_hannover, p_size_1, p_size_1, int(p_size_1 / 2))
    #    sim_road_patches_overlap = imagePatches(sim_road_hannover, p_size_1, p_size_1, int(p_size_1 / 2))
        sim_hm_patches_overlap = imagePatches(sim_heatmap_hannover, p_size_1, p_size_1, int(p_size_1))
        sim_road_patches_overlap = imagePatches(sim_road_hannover, p_size_1, p_size_1, int(p_size_1))
        
        
        #sim_road_patches_overlap_new = removeCorrespondence(sim_road_patches_overlap, sim_hm_patches_overlap)
        #sim_hm_patches_overlap_new = removeCorrespondence(sim_hm_patches_overlap, sim_road_patches_overlap)
        #sim_road_patches_overlap_new_new = removeBlackImg(sim_road_patches_overlap)
        
        sim_road_patches_32_new_new , sim_hm_patches_32_new = removeCorrespondenceAll(sim_road_patches_overlap, sim_hm_patches_overlap)
        
        
        #sim_hm_patches_32_new = sim_hm_patches_32_new + sim_hm_patches_overlap_new
        #sim_road_patches_32_new_new = sim_road_patches_32_new_new + sim_road_patches_overlap_new_new
        
        #sim_hm_patches_32_new = sim_hm_patches_overlap_new
        #sim_road_patches_32_new_new = sim_road_patches_overlap_new_new
        
        print("removed non-filled patches: ", )
        num_img = len(sim_hm_patches_32_new)
        #print("anzahl bilder (vor dem löschen von schwarzen) ", num_img)
        #print("anzahl bilder (nach dem löschen von schwarzen) ", len(sim_road_patches_32_new_new))
        
        print("removed patches with either pure black or white pixels ")
        
        sim_road_patches_used = sim_road_patches_32_new_new
        sim_hm_patches_used = sim_hm_patches_32_new
        p_size_used = p_size_1
        
        print("trained with patch size : ",p_size_used)
        print("check: road/sim-patches : ",len(sim_road_patches_used), len(sim_hm_patches_used))
        
        ### experience 1 - simulated hm
        #index_list_sim = list(range(len(sim_hm_patches_32_new)))
        #index_list_sim = list(range(len(sim_hm_patches_used)))
        #random.shuffle(index_list_sim)
     
        idx_sim = 1000
        if len(sim_road_patches_used) < idx_sim*10:
            idx_sim = int(0.1*len(sim_road_patches_used))
            
        print("idx_sim is: ",idx_sim)
            
        # 30 % trainingsdaten
        # idx_sim = int(len(sim_road_patches_used)*0.3)
        print("number of test patches     : ",idx_sim)
        
        print("number of training patches : ",len(sim_road_patches_used)-idx_sim)
        
        #old
        ### experience 1 - simulated hm
        index_list_sim_all = list(range(len(sim_hm_patches_32_new)))
        random.shuffle(index_list_sim_all)
        
        # maximum number of patches 20.000 - or original number of patches
        max_patches = 20000
        max_patches = min(max_patches,len(sim_hm_patches_32_new))
        print("number of patches used: ",max_patches)
        
        index_list_sim = index_list_sim_all[:max_patches]
        
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
        print("storing image patches")
        np.save("x_train_sim.npy", x_train_sim)
        np.save("y_train_sim.npy", y_train_sim)
        np.save("x_test_sim.npy", x_test_sim)
        np.save("y_test_sim.npy", y_test_sim)
        
    else:
        # load image patch arrays
        print("loading stored image patches ")        
        x_train_sim = np.load("x_train_sim.npy")
        y_train_sim = np.load("y_train_sim.npy")
        x_test_sim = np.load("x_test_sim.npy")
        y_test_sim = np.load("y_test_sim.npy")
    
    # plt.imshow(np.reshape(x_test_sim[2], (p_size_1,p_size_1)))
    # plt.imshow(np.reshape(y_test_sim[2], (p_size_1,p_size_1)))
    
    plt.figure()
    plt.imshow(np.reshape(x_test_sim[1], (p_size_1,p_size_1)))
    print("figure plotted")
    plt.figure()
    plt.imshow(np.reshape(y_test_sim[1], (p_size_1,p_size_1)))
    plt.figure()
    plt.imshow(np.reshape(x_test_sim[6], (p_size_1,p_size_1)))
    print("figure plotted")
    plt.figure()
    plt.imshow(np.reshape(y_test_sim[6], (p_size_1,p_size_1)))
        
    input_shape1 = (None, None, 1) #x_train_sim[0].shape
    print('Input Shape of the models', x_train_sim.shape)
    return input_shape1, x_train_sim, y_train_sim, x_test_sim, y_test_sim

First = True
# store image patch arrays
start_training = True
print("start training - or use existing data ",start_training)
 
if start_training:    
    input_shape1, x_train_sim, y_train_sim, x_test_sim, y_test_sim = generate_input(First)

    print('Input Shape of the models after data read', x_train_sim.shape)

    opt1 = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model_ex1 = create_model(opt1, input_shape1)
    print("model 1 created : ",model_ex1)


# In[ ]:


##### Train the model
#covariance1 = Covariance()
if start_training: 
    start = time.time()
    print("start training")

    History1 = History()
    hist1 = model_ex1.fit(x_train_sim, y_train_sim,
                      #batch_size=16,
                      batch_size=16,
                      #epochs=500,
                      #epochs=60,
                      epochs = 80,
                      verbose=1,
                      shuffle=True,
                      callbacks=[History1],
                      validation_data=(x_test_sim, y_test_sim))

    print("Training reached epoch's end. It took: {} min.".format((time.time()-start)/60))

    ### Save history
    History1_loss = History1.history['loss']
    History1_acc = History1.history['acc']
    History1_val_loss = History1.history['val_loss']
    History1_val_acc = History1.history['val_acc']

    thefile1 = open('History1_loss.txt', 'w')
    for item in History1_loss:
        thefile1.write("%s\n" % item)
    thefile1.close()

    thefile2 = open('History1_acc.txt', 'w')
    for item in History1_acc:
        thefile2.write("%s\n" % item)
    thefile2.close()

    thefile3 = open('History1_val_loss.txt', 'w')
    for item in History1_val_loss:
        thefile3.write("%s\n" % item)
    thefile3.close()

    thefile4 = open('History1_val_acc.txt', 'w')
    for item in History1_val_acc:
        thefile4.write("%s\n" % item)
    thefile4.close()

    ### Save model
    model_json1 = model_ex1.to_json()
    with open(tmpPath + "model_ex1.json", "w") as json_file:
        json_file.write(model_json1)
    model_ex1.save_weights(tmpPath + "weights_model_ex1.h5")
    print("Saved model to disk")
    
    ### Plot history of average covariance - accuracy and loss of the models
    plt.figure()
    plt.plot(History1.history['loss'])
    plt.plot(History1.history['val_loss'])
    plt.title('loss & val_loss')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig("loss", dpi=1000)
    
    plt.figure()
    plt.plot(History1.history['acc'])
    plt.plot(History1.history['val_acc'])
    plt.title('acc & val_acc')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig("acc", dpi=1000)
    plt.close()
    
    
else:
    print("already trained -- use saved model ")
#    model_ex1 = LoadModel(tmpPath + "models\model_ex1.json")
#    model_ex1.load_weights(tmpPath + "models\weights_model_ex1.h5")
    model_ex1 = LoadModel(r"Data/tmp_data/trainGeb10/model_ex1.json")
    model_ex1.load_weights(r"Data/tmp_data/trainGeb10/weights_model_ex1.h5")
    


# In[ ]:


#image_arr = readImg(r"Data/Training_Validation/geb0_inp.tif")
##image_arr = readImg(r"testexampleinput2-inv.tif")
#
#print("image shape is: ",image_arr.shape)
#
##conc2 = np.reshape(model_ex1.predict(np.reshape(image_arr, (1, image_arr.shape[0], image_arr.shape[1], 1))), 
##                   (image_arr.shape[0], image_arr.shape[1] ))
#
#pred = model_ex1.predict(np.reshape(image_arr, (1, image_arr.shape[0], image_arr.shape[1], 1)))
#print("prediction done; shape of result is: ", pred.shape)
#print(pred.shape[0], pred.shape[1], pred.shape[2])
#
#conc2 = np.reshape(pred,(pred.shape[0], pred.shape[1]))
#
#fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
#
##fig = plt.figure(figsize=(pred.shape[1] / 1000, pred.shape[0] / 1000), dpi=100, frameon=False)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#plt.imshow(conc2)
##ax.imshow(conc2, aspect='normal')
#
## Save the image in memory in PNG format
#
##name = "file"+str(num)
#filename = "OutputTest.png"
#fig.savefig(filename, dpi=1000)
#print("file saved")
#
#
## Load this image into PIL
##png2 = Image.open(filename)
#
## Save as TIFF
##png2.save("outputTest.tiff")
##filename.close()



# In[22]:



    

#testIndependet(r"testexampleinput2.tif", testPath, outPath)
# 2753, 4400)
#testIndependet(r"input2.tif", testPath, outPath)
#testIndependet(r"FTest1_input_inv.png", testPath, outPath)
#testIndependet(r"FTest2_input_inv.png", testPath, outPath)
#testIndependet(r"geb11_inp_inv.tif", testPath, outPath)
#testIndependet(r"2traininput_inv-ex.png", testPath, outPath)
print("testing model ------------------------------ ")
# testIndependet(r"gebt2_in.tif", testPath, outPath)
#testIndependet(r"gebt.tif", testPath, outPath)
#testIndependet(r"geb12_1_inp.tif", testPath, outPath)
#testIndependet(r"geb12_2_inp.tif", testPath, outPath)
#testIndependet(r"geb12_3_inp.tif", testPath, outPath)

testIndependet(r"h1.tif", testPath, outPath)
testIndependet(r"h2.tif", testPath, outPath)
testIndependet(r"h3.tif", testPath, outPath)
testIndependet(r"h4.tif", testPath, outPath)



#geb25
#testIndependet(r"gebtif2_in.tif", testPath, outPath)
#testIndependet(r"geb_1.png", testPath, outPath)
#testIndependet(r"geb_2.png", testPath, outPath)
#testIndependet(r"geb_3.png", testPath, outPath)
#testIndependet(r"geb_4.png", testPath, outPath)

#testIndependet(r"geb_5.png", testPath, outPath)
#testIndependet(r"geb_6.png", testPath, outPath)
#testIndependet(r"geb_7.png", testPath, outPath)
#testIndependet(r"geb_8.png", testPath, outPath)
#testIndependet(r"geb_9.png", testPath, outPath)


# typify
# testIndependet(r"gebtif2_1_in.tif", testPath, outPath)
#testIndependet(r"geb12_inp.tif", testPath, outPath) --> zu groß ?
#testIndependet(r"geb1_inp_large.tif", testPath, outPath)

#testIndependet(r"geb1_inp_inv.tif", testPath, outPath)


# In[18]:

#
#### Plot history of average covariance - accuracy and loss of the models
#plt.figure()
#plt.plot(History1.history['loss'])
#plt.plot(History1.history['val_loss'])
#plt.title('loss & val_loss')
#plt.legend(['train', 'test'], loc='upper right')
#plt.savefig("loss", dpi=1000)
#
#plt.figure()
#plt.plot(History1.history['acc'])
#plt.plot(History1.history['val_acc'])
#plt.title('acc & val_acc')
#plt.legend(['train', 'test'], loc='upper right')
#plt.savefig("acc", dpi=1000)


#
#plt.plot(covariance1.avg_cov1)
#plt.plot(covariance1.avg_cov2)
#plt.title('average covariance')
#plt.legend(['train', 'test'], loc='lower left')
#
#plt.show()

## Test trained models on test images
# use patches:
#image_arr = readImg(r"testexampleinput2.tif")
def test_image_patches(fn, inpath, outpath):
    print("testing on image ",fn)
    image_arr = readImg(inpath + fn)
    print(image_arr.shape)
       
    b = 128 #32  # patch size
    image_patches_1 = view_as_windows(image_arr, (b, b), b)
    
    ###### Image reconstruction from patches ######
    test_image_1 = []
    for i in range(image_patches_1.shape[1]):
        test_image_1.append(image_patches_1[0, i])
    print("number of image patches ",len(image_patches_1))
    print("is impatch 0 ",image_patches_1.shape[0])
    print("is impatch 1 ",image_patches_1.shape[1])
    print("test image 1 : ", len(test_image_1))
    print("image_arr.shape[0] ist :",image_arr.shape[0] )
    print("image_arr.shape[0] ist :",image_arr.shape[1] )
    print("lenx : ",image_arr.shape[0]/b)
    print("leny : ",image_arr.shape[1]/b)
    
    test_image_all = []
    for i in range(image_patches_1.shape[1]):
        for k in range(image_patches_1.shape[0]):
            print(" i, k : ",i,k)
            test_image_all.append(np.reshape(image_patches_1[k, i], (b, b)))
    
    print("länge testimageall ",len(test_image_all))
    
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
    
    #print("len von image_array: ",len(image_arr))
    #print("len von conc2: ",len(conc2))
    
    #print("accuracy score: ")
    #print(accuracy_score(image_arr.flatten().astype(bool), (conc2 > 0.5).flatten()))

    fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(conc2)
    #ax.imshow(conc2, aspect='normal')
    fig.savefig(outpath + fn[:-4] + '_pred.png',dpi=1000)
    
    conc2 = conc2 > 0.5
    plt.imshow(conc2, cmap='gray')
    fig.savefig(outpath + fn[:-4] + '_pred_bw.png', dpi=1000)
    print("all images drawn and printed",outpath + fn[:-4] + '_pred_bw.png')

    # fig.savefig("strava_pred", dpi=1000)

##test_image_patches(r"gebtif2_in.tif", testPath, outPath)
#print("generate output using test-patches: ")
#test_image_patches(r"geb_test2.png", testPath, outPath)
#test_image_patches(r"geb_test1.png", testPath, outPath)
#start = time.time()
#print("start prediction")
#
#test_image_patches(r"geb_test3.png", testPath, outPath)
#test_image_patches(r"geb_clip.png", testPath, outPath)
#print("prediction took: {} min.".format((time.time()-start)/60))


#  evaluation(model_ex1, x_test_sim, y_test_sim, 32)