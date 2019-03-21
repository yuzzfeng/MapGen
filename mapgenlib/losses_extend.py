import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy

smooth = 1.

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
    
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
    
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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
    return 1 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred):
    
    dice = dice_coef(y_true, y_pred)
    iou = dice / (2 - dice)
    return iou

def iou_loss(y_true, y_pred):
    
    return 1 - iou_coef(y_true, y_pred)


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Citation:
        https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed