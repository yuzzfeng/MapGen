from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D

"""
Discriminator of all kind 

Reference: https://github.com/eriklindernoren/Keras-GAN
"""


def build_discriminator_patchgan_cycle(img_shape, df):
    """
    PatchGAN:
    
    Change: InstanceNormalization -> BatchNormalization
    
    Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py
    Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/discogan/discogan.py
    Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/pixelda/pixelda.py
    
    Similar to pix2pix, but without conditioning input image
    https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
    """
    def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization(momentum=0.8)(d)
            return d

    img = Input(shape=img_shape)
    
    ## Difference to pix2pix
    #img_B = Input(shape=img_shape)
    # Concatenate image and conditioning image by channels to produce input
    #combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)


def build_discriminator_patchgan_srgan(img_shape, df):
    """
    PatchGAN variant: used on srgan similar cyclegan but with sigmoid output

    https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py

    """

    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Input img
    d0 = Input(shape=img_shape)

    d1 = d_block(d0, df, bn=False)
    d2 = d_block(d1, df, strides=2)
    d3 = d_block(d2, df*2)
    d4 = d_block(d3, df*2, strides=2)
    d5 = d_block(d4, df*4)
    d6 = d_block(d5, df*4, strides=2)
    d7 = d_block(d6, df*8)
    d8 = d_block(d7, df*8, strides=2)

    d9 = Dense(df*16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(d0, validity)

def build_discriminator_simple_dc(img_shape):
    """
    Simple binary GAN: used in dcgan

    https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
    """
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def build_discriminator_critic(img_shape):
    """
    Simple binary GAN: used in wgan and wgan_gp
    
    Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
    Reference: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
    """
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

