"""
Residual u-net for mapgen in keras

Reference: https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py
"""

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
    

def res_block(x, nb_filters, strides, increase = False):
    
    # This implementation used the double 3x3 structure and followed the Identity Mappings
    # He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027.
    
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)
    
    if increase:
        shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x

    res_path = Add()([shortcut, res_path])
    return res_path


def decoder(x, from_encoder):
    
    main_path = UpSampling2D(size=(2, 2))(x)
    #main_path = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path) # Add 2019.03.07
    
    main_path = Concatenate(axis=3)([main_path, from_encoder[2]])
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)], increase = True)

    main_path = UpSampling2D(size=(2, 2))(main_path)
    #main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path) # Add 2019.03.07
    
    main_path = Concatenate(axis=3)([main_path, from_encoder[1]])
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)], increase = True)

    main_path = UpSampling2D(size=(2, 2))(main_path)
    #main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path) # Add 2019.03.07
    
    main_path = Concatenate(axis=3)([main_path, from_encoder[0]])
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)], increase = True)

    return main_path


def encoder(x):
    
    to_decoder = []

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = Add()([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2), (1, 1)], increase = True)
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)], increase = True)
    to_decoder.append(main_path)

    return to_decoder


def build_res_unet(img_shape):
    
    """Residual U-Net Generator"""
        
    inputs = Input(shape=img_shape)
    to_decoder = encoder(inputs)
    path = res_block(to_decoder[2], [256, 256], [(2, 2), (1, 1)], increase = True) # 3x
    path = res_block(path, [256, 256], [(1, 1), (1, 1)]) # Number of block of bottleneck = 1
    path = res_block(path, [256, 256], [(1, 1), (1, 1)]) # Try to add one 2019.01.14, achieved best result ever
    path = decoder(path, from_encoder=to_decoder)
    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path) 

    return Model(input=inputs, output=path)
    