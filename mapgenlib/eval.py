import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from osgeo import gdal

def readImg(img):
    # function to read .tif image files
    
    # Read heatmap
    image = gdal.Open(img)
    # Get band of heatmap, it's gray scale image!
    img_band = image.GetRasterBand(1)
    # Read the image as array
    image = img_band.ReadAsArray()
    # Normalize the pixel values in the range 0-1 acc. to max. normalization
    image = (image - image.min()) / (image.max() - image.min())
    return image.astype('float32')

def predict_eval(SavedModel, img_range, testPath, fn_input, fn_target, out_path, scale, nr = 1):
    # Predict and compare with ground truth
    
    image_arr = readImg(testPath + fn_input)
    image_tar = readImg(testPath + fn_target)
    
    if len(img_range) == 4: # If range was set
        xmin, xmax, ymin, ymax = img_range
        image_arr = image_arr[xmin:xmax, ymin:ymax]
        image_tar = image_tar[xmin:xmax, ymin:ymax]
    
    print("- " + str(scale) +"k", 'Example: ', image_arr.shape)
    evals_orig = evaluate(image_arr, image_tar)
    
    image_arr = rescaleImg(image_arr)
    image_tar = rescaleImg(image_tar)
    
    print("+ " + str(scale) +"k", 'Prediction: ', image_arr.shape)
    pred = model_predict(SavedModel, image_arr, num_runs = nr)
    evals_pred = evaluate(pred > 0.5, image_tar)
    
    save_prediction(pred, out_path, fn_input[:-4], '_' + str(scale) + '_' + str(nr) + '_out.png')
    return [evals_orig, evals_pred]

def predict_only(SavedModel, img_range, testPath, fn_input, out_path, scale, nr = 1):
    # Predict only
    
    image_arr = readImg(testPath + fn_input)
    
    if len(img_range) == 4: # If range was set
        xmin, xmax, ymin, ymax = img_range
        image_arr = image_arr[xmin:xmax, ymin:ymax]
    
    image_arr = rescaleImg(image_arr)
    
    print("+ " + str(scale) +"k", 'Prediction: ', image_arr.shape)
    pred = model_predict(SavedModel, image_arr, num_runs = nr)
    
    save_prediction(pred, out_path, fn_input[:-4], '_' + str(scale) + '_' + str(nr) + '_out.png')
    
def model_predict(newModel, input_image, num_runs):
    # Predict with model for n times
    m,n = input_image.shape
    for i in range(num_runs):
        input_image = np.reshape(input_image, (1, m, n, 1))
        input_image = newModel.predict([input_image])
        input_image = np.reshape(input_image,(m, n)) > 0.5
    return input_image

def rescaleImg(image_arr):
    # Rescale the image to 8n x 8n, because 3x downsampling
    
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

def IoUcheck(img_input, img_output):
    # Pixelwise IoU score
    
    logic_and = np.sum(np.logical_and(img_output, img_input))
    logic_or = np.sum(np.logical_or(img_output, img_input))
    return logic_and/logic_or

def evaluate(image_arrA, image_arrB):
    # Calculate the evaluation metrics
    
    target_names = ['0', '1']
    
    y_true = image_arrB.flatten().astype(bool) 
    y_pred = image_arrA.flatten().astype(bool)
        
    Accuracy = accuracy_score(y_true, y_pred)
    IoU = IoUcheck(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
        
    print('Acc & IoU:', Accuracy, IoU)
    print('Error:', 1 - Accuracy)
    print(conf)

    return Accuracy, IoU, conf, report

def save_prediction(img, out_path, out_filename, subfix):
    
    fig = plt.figure(figsize=(img.shape[1] / 1000, img.shape[0] / 1000), dpi=100, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(img, cmap='gray')
    fig.savefig(out_path + out_filename + subfix, dpi=1000)