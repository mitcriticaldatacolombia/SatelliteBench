""" Setup Environment """
import tensorflow as tf
from tensorflow.keras import metrics

import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


""" Constants: """

# Image size
#target_size = (224, 224, 12)
#latent_dim = 1024 # Number of latent dim parameters

""" Autoencoder: """
def get_Autoencoder(model_path=None, backbone=None, target_size = (224, 224, 12), latent_dim = 1024, loss='mean_squared_error', optimizer='adam', model_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()]):
                    
    """ Encoder Part: """
    input_data = Input(shape=target_size, name='encoder_input')

    # Conv block 1
    encoder = Conv2D(32, 3, activation="relu", strides=2, padding="same", name="conv_1")(input_data)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    # Conv block 2
    encoder = Conv2D(64, 3, activation="relu", strides=2, padding="same", name="conv_2")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    # Conv block 3
    encoder = Conv2D(128, 3, activation="relu", strides=2, padding="same", name="conv_3")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = LeakyReLU()(encoder)

    conv_shape = K.int_shape(encoder) #Shape of conv to be provided to decoder
    encoder = Flatten()(encoder)
    
    """ Latent Distribution and Sampling """
    latent_encoding = Dense(latent_dim, activation='relu')(encoder)
    
    encoder_model = Model(input_data, latent_encoding)
    
    # return encoder_model

    """ Decoder Part """
    decoder_input = Input(shape=(latent_dim), name='decoder_input')
    decoder = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3])(decoder_input)

    decoder = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)

    # Transpose Conv block 1
    decoder = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same", name="deconv_1")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    # Transpose Conv block 2
    decoder = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name="deconv_2")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    # Transpose Conv block 1
    decoder = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name="deconv_3")(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = LeakyReLU()(decoder)

    decoder_output = Conv2DTranspose(target_size[2], 3, activation='relu', padding="same")(decoder)
    decoder_model = Model(decoder_input, decoder_output)
    
    
    """ Model: """
    encoded = encoder_model(input_data)
    decoded = decoder_model(encoded)
    
    autoencoder = Model(input_data, decoded)
    
    # Compile AE
    if metrics:
        autoencoder.compile(loss=loss, optimizer=optimizer)
    else:
        autoencoder.compile(loss=loss, 
                            optimizer=optimizer, 
                            metrics=model_metrics)
        
    if model_path:
        # Load weights
        autoencoder.load_weights(model_path, by_name=True)
    
    model = keras.Sequential()
    if backbone:
        for layer in autoencoder.layers[:-1]: # just exclude last layer from copying
            model.add(layer)    
        
        model.trainable = False
        
        return model
    return autoencoder

