from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta
from keras import losses
from keras.utils import to_categorical
import keras.backend as K


def getDiscriminator(img_shape, out_activation = 'sigmoid'):
        
        model = Sequential()

        model.add(Conv2D(16, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dropout(0.5))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Dropout(0.5))
        
        model.add(Flatten())
        model.add(Dense(1, activation = out_activation))
        model.summary()
        
        return model

def getGenerator(img_shape, out_activation = 'sigmoid'):
 
        model = Sequential()

        # Encoder

        # Down sample
        model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding="same", 
                         input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Down sample
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Down sample
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        
        # Decoder
        
        # Up sample
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(256, kernel_size=(4, 4), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        # Up sample
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(4, 4), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        # Up sample
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(48, kernel_size=(4, 4), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(24, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        
        model.add(Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        model.add(Activation(out_activation)) 

        model.summary()
        
        return model