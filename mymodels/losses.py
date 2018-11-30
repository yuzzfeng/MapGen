import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy

_EPSILON = 10e-8

def IoU(yTrue,yPred):  
    
    I = tf.multiply(yTrue, yPred, name="intersection")
    U = yTrue + yPred - I + _EPSILON
    
    IoU = tf.reduce_sum(I) / tf.reduce_sum(U)
    return -tf.log(IoU + _EPSILON) + binary_crossentropy(yTrue,yPred)
    
    #IoU = tf.divide(I, U, name='IoU')
    #L = -tf.log(IoU + _EPSILON)
    #return tf.reduce_mean(L)
    
def MSE_CROSS(yTrue,yPred):
    return binary_crossentropy(yTrue,yPred) + K.abs(K.sum(yTrue) - K.sum(yPred))

def linf_loss(yTrue,yPred):
    return K.max(K.abs(yTrue - yPred))

def dice_coef(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: dice_coeff -- A metric that accounts for precision and recall
                           on the scale from 0 - 1. The closer to 1, the
                           better.
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)


def dice_coef_loss(y_true, y_pred):
    '''
    Params: y_true -- the labeled mask corresponding to an rgb image
            y_pred -- the predicted mask of an rgb image
    Returns: 1 - dice_coeff -- a negation of the dice coefficient on
                               the scale from 0 - 1. The closer to 0, the
                               better.
    Citation (MIT License): https://github.com/jocicmarko/
                            ultrasound-nerve-segmentation/blob/
                            master/train.py
    '''
    return 1-dice_coef(y_true, y_pred)


