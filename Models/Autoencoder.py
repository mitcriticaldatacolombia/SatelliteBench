""" Setup Environment """
import tensorflow as tf
from tensorflow.keras import metrics

import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

from transformers import TFViTModel
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50V2
from tensorflow.keras.applications.convnext import ConvNeXtBase, ConvNeXtSmall, ConvNeXtTiny


import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()



""" Constants: """

# Image size
#target_size = (224, 224, 12)
#latent_dim = 1024 # Number of latent dim parameters

""" VIT """
class ViTLayer(tf.keras.layers.Layer):
    def __init__(self, backbone, **kwargs):
        super(ViTLayer, self).__init__(**kwargs)
        self.backbone = backbone
        
    def build(self, input_shape):
        self.vit = TFViTModel.from_pretrained(self.backbone)
        
    def call(self, inputs):
        out = self.vit(inputs)['pooler_output']
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vit.config.hidden_size)


class CustomViT(TFViTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(3, kernel_size=1, padding="same")

    def call(self, inputs, **kwargs):
        # Convert the input to a tensor with the correct data type (float32)
        #inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # Expand the input from 12 to 3 channels using the Conv2D layer
        x = self.conv(inputs)
        # Pass the input through the rest of the ViT model
        return super().call(x, **kwargs)
    
    
""" Autoencoder: """
def get_Autoencoder(model_path=None, backbone=None, target_size = (224, 224, 3), latent_dim = 1024, loss='mean_squared_error', optimizer='adam', model_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()], encoder_backbone=None):
                    
    """ Encoder Part: """
    input_data = Input(shape=target_size, name='encoder_input')
    
    if not encoder_backbone:

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
        
    elif encoder_backbone == 'vit':
        if target_size[2] == 12:
            input_data_3chanels = Conv2D(3, kernel_size=1, padding="same", name="conv_1", input_shape=target_size)(input_data) 
            chanel_first_inputs = tf.keras.layers.Permute((3,1,2))(input_data_3chanels)
        else:
            chanel_first_inputs = tf.keras.layers.Permute((3,1,2))(input_data)

        vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")(chanel_first_inputs)
            
        encoder = vit['pooler_output']#['last_hidden_state']
        conv_shape = (None, 28, 28, 128)
        
    elif encoder_backbone == 'ResNet50V2':
        if target_size[2] != 3:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights=None)(input_data)
        encoder = tf.keras.layers.GlobalAveragePooling2D()(cnn)
        conv_shape = (None, 28, 28, 128)
        
    elif encoder_backbone == 'ConvNeXtBase':
        if target_size[2] != 3:
            cnn = ConvNeXtBase(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ConvNeXtBase(input_shape=target_size, include_top=False, weights=None)(input_data)
        encoder = tf.keras.layers.GlobalAveragePooling2D()(cnn)
        conv_shape = (None, 28, 28, 128)
            
    elif encoder_backbone == 'ConvNeXtTiny':
        if target_size[2] != 3:
            cnn = ConvNeXtTiny(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ConvNeXtTiny(input_shape=target_size, include_top=False, weights=None)(input_data)
        encoder = tf.keras.layers.GlobalAveragePooling2D()(cnn)
        conv_shape = (None, 28, 28, 128)
    
    # Flatten:
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

