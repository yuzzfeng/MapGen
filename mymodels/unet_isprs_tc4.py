import keras
from keras import backend as K

from keras.models import Model, Sequential

from keras.layers.core import Dropout
from keras.layers import Layer, Input, Conv2D, Dropout, UpSampling2D, Activation, Concatenate, Add, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation

from keras import regularizers
from keras.regularizers import Regularizer
from keras.callbacks import History


##### Building the CNN archıtecture with "Model" - skip connections were added
def create_model_add_skips_2(optimizer, input_shape, drop_rate = 0.1): # ISPRS TCIV

    
    i = Input(shape=input_shape)
    
    
    x = Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              activation='relu', input_shape=input_shape, kernel_initializer='random_uniform',
              name="flat_conv_a")(i)
    first_skip = Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              activation='relu',name="flat_conv_b")(x)
    x = Conv2D(filters=24, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu',
                     name="down_conv_1")(first_skip)
    #x = Dropout(drop_rate)(x) ################################################# First Drop
    
    
    
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1")(x)
    second_skip = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', name="down_conv_2")(second_skip)
    #x = Dropout(drop_rate)(x) ################################################# Second Drop
    
    
    
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_3")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_4")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="xx_conv_3")(x)
    #x = Dropout(drop_rate)(x) ################################################# Third Drop
    
    
    
    x = Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_5")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_6")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_7")(x)
    x = Dropout(drop_rate)(x) ################################################# 4th Drop
    
    

    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="xx_conv_0")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8b")(x)
    #x = Dropout(drop_rate)(x) ################################################# 5th Drop
    
    
    x = UpSampling2D(size=(2, 2), name='up_samp_1')(x)
    x = Conv2D(filters=64, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_1")(x)
    concat = Concatenate()([second_skip, x])
    x = Conv2D(filters=64, kernel_size=(1, 1),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_9")(concat)
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_10")(x)
    #x = Dropout(drop_rate)(x) ################################################# 6th Drop
    
    
    x = UpSampling2D(size=(2, 2), name='up_samp_2')(x)
    x = Conv2D(filters=24, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_2")(x)
    concat2 = Concatenate()([first_skip, x])
    x = Conv2D(filters=12, kernel_size=(1, 1),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_11")(concat2)
    x = Conv2D(filters=12, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_11b")(x)
    #x = Dropout(drop_rate)(x) ################################################# 7th Drop
    
    
    o = Conv2D(filters=1, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='sigmoid', name="flat_conv_12")(x)
    model = Model(inputs=i, outputs=o)

    # Compile model with Adam optimizer and binary cross entropy loss function
    #model.compile(optimizer=optimizer,
    #              loss = 'binary_crossentropy', #loss=IoU, #'binary_crossentropy'
    #              metrics=['acc', IoU])
    
    model.compile(optimizer=optimizer,
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    
    return model
