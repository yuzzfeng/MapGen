# Reference: https://github.com/DuFanXin/deep_residual_unet/blob/master/res_unet.py

import keras
from keras import backend as K

from keras.models import Model
from keras.models import Sequential

from keras.layers.core import Dropout
from keras.layers import Input, Conv2D, Dropout, UpSampling2D, Activation, Concatenate, Add
from keras.layers import MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = Add()([shortcut, res_path])
    return res_path

def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = Concatenate(axis=3)([main_path, from_encoder[2]])
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path) 
    main_path = Concatenate(axis=3)([main_path, from_encoder[1]])
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = Concatenate(axis=3)([main_path, from_encoder[0]])
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)])

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

    main_path = res_block(main_path, [64, 64], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def build_res_unet(input_shape):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [256, 256], [(2, 2), (1, 1)]) # 3x
    
    path = res_block(path, [256, 256], [(1, 1), (1, 1)]) # Yu.add - in 2018-12-02 16-09-04_15 only once

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return Model(input=inputs, output=path)


##### Building the CNN archıtecture with "Model" 
def create_model_residual(optimizer, input_shape, drop_rate = 0.1):

    model = build_res_unet(input_shape)
    model.compile(optimizer=optimizer,
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    
    return model