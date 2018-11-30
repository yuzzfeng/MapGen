import os
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal
from skimage.util.shape import view_as_windows
from sklearn.metrics import accuracy_score

from keras import models


def prediction_independent(model_ex1, image_arr):
    
    conc2 = np.reshape(model_ex1.predict(np.reshape(image_arr, (1, image_arr.shape[0], image_arr.shape[1], 1))), 
                   (image_arr.shape[0], image_arr.shape[1]))
    return conc2


_EPSILON = 10e-8
def IoU(yTrue,yPred):  
    
    I = tf.multiply(yTrue, yPred, name="intersection")
    U = yTrue + yPred - I + _EPSILON
    
    IoU = tf.reduce_sum(I) / tf.reduce_sum(U)
    return -tf.log(IoU + _EPSILON) + binary_crossentropy(yTrue,yPred)

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

# function to read .tif image files
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

# function to read .tif image files
def readImgInv(img):
    # Read heatmap
    image = gdal.Open(img)
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
def removeBlackImg(img_patch):
    patch_list = []
    patch_list_new = []
    for i in range(len(img_patch)):
        patch_list.append(img_patch[i])
        if patch_list[i].max() != 0:
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
        if patch_heat_list[i].max() != 0:
            patch_road_list_new.append(road[i])
    return patch_road_list_new

def check_and_create(out_dir):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
        
def IoUcheck(img_input, img_output):

    logic_and = np.sum(np.logical_and(img_output, img_input))
    logic_or = np.sum(np.logical_or(img_output, img_input))

    return logic_and/logic_or

# cut the image to avoid shape error
##def rescaleImg(image_arr):
##    
##    print("Original:", image_arr.shape)
##    
##    if image_arr.shape[0] % 4 != 0:
##        n = image_arr.shape[0] % 4
##        new_x = image_arr.shape[0] - n
##    else:
##        new_x = image_arr.shape[0]
##
##    if image_arr.shape[1] % 4 != 0:
##        n = image_arr.shape[1] % 4
##        new_y = image_arr.shape[1] - n
##    else:
##        new_y = image_arr.shape[1]
##    
##    image_arr = image_arr[:new_x, :new_y]
##    print("Clipped:", image_arr.shape)
##    return image_arr

def rescaleImg(image_arr):
    
    if image_arr.shape[0] % 8 != 0:
        n = image_arr.shape[0] % 8
        new_x = image_arr.shape[0] - n
    else:
        new_x = image_arr.shape[0]

    if image_arr.shape[1] % 8 != 0:
        n = image_arr.shape[1] % 8
        new_y = image_arr.shape[1] - n
    else:
        new_y = image_arr.shape[1]

    image_arr = image_arr[:new_x, :new_y]
    
    return image_arr

# function to load a saved model
def LoadModel(model_json):
    from keras.models import model_from_json
    json_file = open(model_json)
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def testIndependet(fn, target, inpath, outpath, model_ex1):
    
    image_arr = readImg(inpath + fn)
    #print(image_arr.shape)
    image_arr = rescaleImg(image_arr)
    #print(image_arr.shape)
    
    image_tar = readImg(inpath + target)
    #print(image_tar.shape)
    image_tar = rescaleImg(image_tar)
    #print(image_tar.shape)
    
    conc2 = np.reshape(model_ex1.predict(np.reshape(image_arr, (1, image_arr.shape[0], image_arr.shape[1], 1))), 
                       (image_arr.shape[0], image_arr.shape[1]))
    
    acc = accuracy_score(image_tar.flatten().astype(bool), (conc2 > 0.5).flatten())
    iou = IoUcheck(image_tar.flatten().astype(bool), (conc2 > 0.5).flatten())
    print('accuracy:', acc)
    print('IoU:', iou)
    
    fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    plt.imshow(conc2)
    fig.savefig(outpath + fn[:-4] + '_out.png', dpi=1000)
    
    
    fig = plt.figure(figsize=(image_arr.shape[1] / 1000, image_arr.shape[0] / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    conc2 = conc2 > 0.5
    plt.imshow(conc2, cmap='gray')
    fig.savefig(outpath + fn[:-4] + '_out_bw.png', dpi=1000)
    
    return acc, iou
    
def predict_15k(tmpPath, testPath, outPath, 
                fn_input, fn_target):
    
    
    image_arrA = readImg(testPath + fn_input)
    image_arrB = readImg(testPath + fn_target)

    print("15k", 'Example: ')

    Accuracy = accuracy_score(image_arrB.flatten().astype(bool), 
                              image_arrA.flatten().astype(bool))

    IntOverUnion = IoUcheck(image_arrB.flatten().astype(bool), 
                            image_arrA.flatten().astype(bool))
    
    print('accuracy:', Accuracy)
    print('IoU:', IntOverUnion)

    if "weights.hdf5" in os.listdir(tmpPath):
        model_ex1 = models.load_model(tmpPath + "weights.hdf5")
    else:
        model_ex1 = LoadModel(tmpPath + "model_ex1.json")
        model_ex1.load_weights(tmpPath + "weights_model_ex1.h5")
    
    acc, iou = testIndependet(fn_input, fn_target, testPath, outPath, model_ex1)
    
    return [[Accuracy, IntOverUnion], [acc, iou]]


def predict_scale(tmpPath, testPath, outPath, scale):
    
    fns = os.listdir(testPath)
    fn_input  = [fn for fn in fns if 'geb_clip' in fn][0]
    fn_target = [fn for fn in fns if str(scale) in fn][0]
    print('Input and Target:', fn_input, fn_target)

    image_arrA = readImg(testPath + fn_input)
    image_arrB = readImg(testPath + fn_target)

    print(str(scale), 'Example: ')

    Accuracy = accuracy_score(image_arrB.flatten().astype(bool), 
                              image_arrA.flatten().astype(bool))

    IntOverUnion = IoUcheck(image_arrB.flatten().astype(bool), 
                            image_arrA.flatten().astype(bool))
    
    print('accuracy:', Accuracy)
    print('IoU:', IntOverUnion)

    if "weights.hdf5" in os.listdir(tmpPath):
        model_ex1 = models.load_model(tmpPath + "weights.hdf5")
    else:
        model_ex1 = LoadModel(tmpPath + "model_ex1.json")
        model_ex1.load_weights(tmpPath + "weights_model_ex1.h5")

    
    
    acc, iou = testIndependet(fn_input, fn_target, testPath, outPath, model_ex1)
    
    return [[Accuracy, IntOverUnion], [acc, iou]]


def save_hist(History1, outPath):

    ### Save history
    History1_loss = History1.history['loss']
    History1_acc = History1.history['acc']
    History1_val_loss = History1.history['val_loss']
    History1_val_acc = History1.history['val_acc']


    thefile1 = open(outPath + 'History1_loss.txt', 'w')
    for item in History1_loss:
        thefile1.write("%s\n" % item)
    thefile1.close()

    thefile2 = open(outPath + 'History1_acc.txt', 'w')
    for item in History1_acc:
        thefile2.write("%s\n" % item)
    thefile2.close()

    thefile3 = open(outPath + 'History1_val_loss.txt', 'w')
    for item in History1_val_loss:
        thefile3.write("%s\n" % item)
    thefile3.close()

    thefile4 = open(outPath + 'History1_val_acc.txt', 'w')
    for item in History1_val_acc:
        thefile4.write("%s\n" % item)
    thefile4.close()

    ### Plot history of average covariance - accuracy and loss of the models
    plt.figure()
    plt.plot(History1.history['loss'])
    plt.plot(History1.history['val_loss'])
    plt.title('loss & val_loss')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(outPath + "loss", dpi=1000)

    plt.figure()
    plt.plot(History1.history['acc'])
    plt.plot(History1.history['val_acc'])
    plt.title('acc & val_acc')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(outPath + "acc", dpi=1000)

    #plt.figure()
    #plt.plot(History1.history['IoU'])
    #plt.plot(History1.history['val_IoU'])
    #plt.title('IoU & val_IoU')
    #plt.legend(['train', 'test'], loc='lower right')
    #plt.savefig(outPath + "IoU", dpi=1000)


def save_model(model_ex1, outPath):
    ### Save model
    model_json1 = model_ex1.to_json()
    with open(outPath + "model_ex1.json", "w") as json_file:
        json_file.write(model_json1)
    model_ex1.save_weights(outPath + "weights_model_ex1.h5")
    print("Saved model to disk")

    
    
    
    
    
    
# New versions on 19.07.2018
    
def predict_patches(model_ex1, image_arr, b = 256):
    
    print("Input shape:", image_arr.shape)
    
    sub_b = int(b/4)
    image_arr = np.pad(image_arr, (sub_b, sub_b), 'reflect')
    image_patches_1 = view_as_windows(image_arr, (b, b), int(b/2))

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
    
    start = int(b/4)
    end = int(3 * b /4)
    
    conc1 = []
    for m in range(0, image_patches_1.shape[0] * image_patches_1.shape[1], image_patches_1.shape[0]):
        g = [img[start:end, start:end] for img in test_image_pred_all[m:m + image_patches_1.shape[0]]]
        conc1.append(np.concatenate(g, axis=0))

    for n in range(image_patches_1.shape[0]):
        conc2 = (np.concatenate((conc1), axis=1))
    
    print("Output shape:", conc2.shape)
    return conc2

def normalImg(image_arr, size = 256):
    
    if image_arr.shape[0] % size != 0:
        n = image_arr.shape[0] % size
        new_x = image_arr.shape[0] - n
    else:
        new_x = image_arr.shape[0]

    if image_arr.shape[1] % size != 0:
        n = image_arr.shape[1] % size
        new_y = image_arr.shape[1] - n
    else:
        new_y = image_arr.shape[1]

    image_arr = image_arr[:new_x, :new_y]
    
    return image_arr

def indepTest(fn, target, inpath, outpath, model_ex1):
    
    image_arr = readImg(inpath + fn)
    #print(image_arr.shape)
    image_arr = normalImg(image_arr)
    #print(image_arr.shape)
    
    image_tar = readImg(inpath + target)
    #print(image_tar.shape)
    image_tar = normalImg(image_tar)
    #print(image_tar.shape)
    
    conc2 = predict_patches(model_ex1, image_arr)
    
    acc = accuracy_score(image_tar.flatten().astype(bool), (conc2 > 0.5).flatten())
    iou = IoUcheck(image_tar.flatten().astype(bool), (conc2 > 0.5).flatten())
    print('accuracy:', acc)
    print('IoU:', iou)
    
    plt.imsave(outpath + fn[:-4] + '_out.png', conc2)
    bw = conc2 > 0.5
    plt.imsave(outpath + fn[:-4] + '_out_bw.png', bw, cmap = 'gray')
    
    return acc, iou


def predict_and_compare(tmpPath, testPath, outPath, scale = 15):

    fns = os.listdir(testPath)
    fn_input  = [fn for fn in fns if 'geb_clip' in fn][0]
    fn_target = [fn for fn in fns if str(scale) in fn][0]
    print('Input and Target:', fn_input, fn_target)

    image_arrA = readImg(testPath + fn_input)
    image_arrB = readImg(testPath + fn_target)

    print(str(scale) + "k", 'Example: ')
    
    Accuracy = accuracy_score(image_arrB.flatten().astype(bool), 
                              image_arrA.flatten().astype(bool))

    IntOverUnion = IoUcheck(image_arrB.flatten().astype(bool), 
                            image_arrA.flatten().astype(bool))
    
    print('accuracy:', Accuracy)
    print('IoU:', IntOverUnion)

    if "weights.hdf5" in os.listdir(tmpPath):
        model_ex1 = models.load_model(tmpPath + "weights.hdf5")
    else:
        model_ex1 = LoadModel(tmpPath + "model_ex1.json")
        model_ex1.load_weights(tmpPath + "weights_model_ex1.h5")
    
    acc, iou = indepTest(fn_input, fn_target, testPath, outPath, model_ex1)
    
    return [[Accuracy, IntOverUnion], [acc, iou]]
    
    