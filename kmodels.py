import keras
from keras import backend as K

from keras.models import Model
from keras.models import Sequential

from keras.layers.core import Dropout
from keras.layers import Input, Conv2D, Dropout, UpSampling2D, Activation, Concatenate
from keras.layers import MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation

from keras.callbacks import History

##### Building the CNN archıtecture with "Model" - skip connections were added
def create_model_add_skips_3(optimizer, input_shape, drop_rate = 0.1):

    
    i = Input(shape=input_shape)
    
    
    x = Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              activation='relu', input_shape=input_shape, kernel_initializer='random_uniform',
              name="flat_conv_a")(i)
    first_skip = Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              activation='relu',name="flat_conv_b")(x)
    x = Conv2D(filters=24, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu',
                     name="down_conv_1")(first_skip)
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pool_1')(x)
    #x = Dropout(drop_rate)(x) ################################################# First Drop
    
    
    
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1")(x)
    second_skip = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="down_conv_2")(second_skip)
    x = MaxPooling2D(pool_size=2, strides=2, name='max_pool_2')(x)
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



##### Building the CNN archıtecture with "Model" - skip connections were added
def create_model_add_skips_2(optimizer, input_shape, drop_rate = 0.1):

    
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




##### Building the CNN archıtecture with "Model" - skip connections were added
def create_model_add_skips(optimizer, input_shape, drop_rate = 0.3):

    
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
    x = Dropout(drop_rate)(x) ################################################# First Drop
    
    
    
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1")(x)
    second_skip = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', name="down_conv_2")(second_skip)
    x = Dropout(drop_rate)(x) ################################################# Second Drop
    
    
    
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_3")(x)
    third_skip = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_4")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', name="down_conv_3")(third_skip)
    x = Dropout(drop_rate)(x) ################################################# Third Drop
    
    
    
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
    
    
    x = UpSampling2D(size=(2, 2), name='up_samp_0')(x)
    x = Conv2D(filters=128, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     activation='relu', name="up_conv_0")(x)
    concat0 = Concatenate()([third_skip, x])
    x = Conv2D(filters=128, kernel_size=(1, 1),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8")(concat0)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8b")(x)
    x = Dropout(drop_rate)(x) ################################################# 5th Drop
    
    
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
    x = Dropout(drop_rate)(x) ################################################# 6th Drop
    
    
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
    x = Dropout(drop_rate)(x) ################################################# 7th Drop
    
    
    o = Conv2D(filters=1, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='sigmoid', name="flat_conv_12")(x)
    model = Model(inputs=i, outputs=o)

    # Compile model with Adam optimizer and binary cross entropy loss function
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model



## Building the CNN archıtecture with "Sequential Model" (model looks like autoencoder)
## Version with batch normalozation - Do not benifit that much

def create_model_batch(optimizer, input_shape):
    
    model = Sequential()
    droprate = 0.1
    
    model.add(Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              input_shape=input_shape, kernel_initializer='random_uniform', name="flat_conv_a"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same', name="flat_conv_b"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    ## Encoding (down-sampling) ###   
    model.add(Conv2D(filters=24, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     name="down_conv_1"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_1"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_2"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    
    ## Encoding (down-sampling) ### 
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     name="down_conv_2"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_3"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_4"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_5"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_6"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_6a"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_6b"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_6c"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_7"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_8"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    
    ###############################################################################
    model.add(UpSampling2D(size=(2, 2), name='up_samp_1'))
    
    model.add(Conv2D(filters=64, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     name="up_conv_1"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_9"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_10"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))
    ###############################################################################
    model.add(UpSampling2D(size=(2, 2), name='up_samp_2'))

    model.add(Conv2D(filters=24, kernel_size=(4, 4),
                     strides=(1, 1), padding='same',
                     name="up_conv_2"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=12, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     name="flat_conv_11"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
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
    return model



##### Building the CNN archıtecture with "Sequential Model" 
##### (model looks like autoencoder)
def create_model(optimizer, input_shape):
    model = Sequential()
    
    droprate = 0.1

    model.add(Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              activation='relu', input_shape=input_shape, kernel_initializer='random_uniform',
              name="flat_conv_a"))
    #model.add(Dropout(droprate))
    
    model.add(Conv2D(filters=24, kernel_size=(3, 3),
              strides=(1, 1), padding='same',
              activation='relu',name="flat_conv_b"))
    #model.add(Dropout(droprate))
    
#    model.add(Conv2D(filters=24, kernel_size=(3, 3),
#              strides=(1, 1), padding='same',
#              activation='relu',name="flat_conv_c"))
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
#    model.add(Dropout(droprate))
    
    ## Encoding (down-sampling) ###   
    model.add(Conv2D(filters=24, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', #input_shape=input_shape, kernel_initializer='random_uniform',
                     name="down_conv_1"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1"))
    #model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2"))
    #model.add(Dropout(droprate))
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
    #model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_4"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_5"))
    #model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_6"))
    model.add(Dropout(droprate))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_7"))
    #model.add(Dropout(droprate))

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
    #model.add(Dropout(droprate))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_10"))
    #model.add(Dropout(droprate))
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
    #model.add(Dropout(droprate))

    model.add(Conv2D(filters=1, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='sigmoid', name="flat_conv_12"))
    # model.add(Activation(our_activation))
    #model.add(Dropout(droprate))

    # Compile model with Adam optimizer and binary cross entropy loss function
    model.compile(optimizer=optimizer,
                  loss = 'binary_crossentropy', #'mean_squared_error', #'binary_crossentropy', #loss=IoU, #'binary_crossentropy'
                  metrics=['acc']) #metrics=['acc', IoU])
    
    
    return model


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