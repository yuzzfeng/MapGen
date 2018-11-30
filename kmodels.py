import keras
from keras import backend as K

from keras.models import Model
from keras.models import Sequential

from keras.layers.core import Dropout
from keras.layers import Input, Conv2D, Dropout, UpSampling2D, Activation, Concatenate, Add
from keras.layers import MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation

from keras.callbacks import History

from keras import regularizers
from keras.regularizers import Regularizer

from keras.layers import Layer
import keras.backend as K


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = Add([shortcut, res_path])
    return res_path

def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = Concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = Concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = Concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path

def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = Add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def build_res_unet(input_shape):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return Model(input=inputs, output=path)



##### Building the CNN archıtecture with "Model" - skip connections were added
def create_model_residual(optimizer, input_shape, drop_rate = 0.1):

    
    model = build_res_unet(input_shape)
    
    model.compile(optimizer=optimizer,
                  loss = 'binary_crossentropy',
                  metrics=['acc'])
    
    return model


'''
Comparision of the loss functions

'binary_crossentropy' > dice_coef

'''

class Round(Layer):

    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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


class linf_reg(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        output = self.layer.get_output(True)
        return K.sum(K.max(K.abs(output), axis=0))
        
        #loss += self.l1 * K.sum(K.mean(K.abs(output), axis=0))
        #loss += self.l2 * K.sum(K.mean(K.square(output), axis=0))
        #return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}
    
class linf_reg_weight(Regularizer):
        """Regularizer for L1 and L2 regularization.
        # Arguments
            l1: Float; L1 regularization factor.
            l2: Float; L2 regularization factor.
        """

        def __init__(self, l1=0., l2=0.):
            self.l1 = K.cast_to_floatx(l1)
            self.l2 = K.cast_to_floatx(l2)

        def __call__(self, x):
            return K.max(K.abs(x))

        def get_config(self):
            return {'l1': float(self.l1),
                    'l2': float(self.l2)}

# Current testing
def create_model_add_skips_linfnorm(optimizer, input_shape, drop_rate = 0.1):
    
    def linf_reg(x):
        return K.max(K.abs(x))
    
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
                     activation='relu', name="down_conv_1", kernel_regularizer=linf_reg)(first_skip)
    #x = Dropout(drop_rate)(x) ################################################# First Drop
    
    
    
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_1")(x)
    second_skip = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_2")(x)
    x = Conv2D(filters=64, kernel_size=(3, 3),
                     strides=(2, 2), padding='same',
                     activation='relu', name="down_conv_2", kernel_regularizer=linf_reg)(second_skip)
    #x = Dropout(drop_rate)(x) ################################################# Second Drop
    
    
    
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_3")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_4")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="xx_conv_3", kernel_regularizer=linf_reg)(x)
    #x = Dropout(drop_rate)(x) ################################################# Third Drop
    
    
    
    x = Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_5")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_6")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_7", kernel_regularizer=linf_reg)(x)
    x = Dropout(drop_rate)(x) ################################################# 4th Drop
    
    

    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="xx_conv_0")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu', name="flat_conv_8b", kernel_regularizer=linf_reg)(x)
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
                     activation='relu', name="flat_conv_10", kernel_regularizer=linf_reg)(x)
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
                     activation='relu', name="flat_conv_11b", kernel_regularizer=linf_reg)(x)
    #x = Dropout(drop_rate)(x) ################################################# 7th Drop
    
    
    o = Conv2D(filters=1, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='sigmoid', name="flat_conv_12", kernel_regularizer=linf_reg)(x)
    
    o = Round()(o)
    
    model = Model(inputs=i, outputs=o)

    # Compile model with Adam optimizer and binary cross entropy loss function
    #model.compile(optimizer=optimizer,
    #              loss = 'binary_crossentropy', #loss=IoU, #'binary_crossentropy'
    #              metrics=['acc', IoU])
    
    #def linf_loss(yTrue,yPred):
    #    return K.max(K.abs(yTrue - yPred))

    model.compile(optimizer=optimizer,
                  loss = 'binary_crossentropy', #dice_coef_loss, #'sparse_categorical_crossentropy', #'mse', #'binary_crossentropy',
                  metrics =  ['acc'])
    
    return model


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