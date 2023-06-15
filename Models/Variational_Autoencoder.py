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
disable_eager_execution()


""" Constants: """
# Image size
#target_size = (224, 224, 12)
#latent_dim = 1024 # Number of latent dim parameters

""" Variational Autoencoder: """

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

    
def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = K.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random


def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch #*target_size[0]*target_size[1]
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        #return kl_loss_batch*(-.5)
        return kl_loss_batch*(-.5)#(-5e-4)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch
    
    return total_loss


def get_Variational_Autoencoder(model_path=None, backbone=None, target_size = (224, 224, 12), latent_dim = 1024, optimizer='adam', lr=0.001, model_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()], encoder_backbone=None):
    
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
        # ViT uses chanel first, but our images has chanel last
        chanel_first_inputs = tf.keras.layers.Permute((3,1,2))(input_data)
        vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")(chanel_first_inputs)
        encoder = vit['pooler_output']#['last_hidden_state']
        conv_shape = (None, 28, 28, 128)
        
    elif encoder_backbone == 'ResNet50V2':
        if target_size[2] != 3:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights=None)(input_data)
        else:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights='imagenet')(input_data)
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
    
    encoder = Flatten()(encoder)
        
    """ Latent Distribution and Sampling """
    initializer = tf.keras.initializers.Zeros()

    distribution_mean = Dense(latent_dim, name='mean', kernel_initializer=initializer)(encoder)
    distribution_variance = Dense(latent_dim, name='log_variance', kernel_initializer=initializer)(encoder)

    latent_encoding = Lambda(sample_latent_features)([distribution_mean, distribution_variance])
    
    encoder_model = Model(input_data, latent_encoding)

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
    
    if optimizer == 'adam':
        if int(tf.__version__.split('.')[1]) >= 11:
            optimizer = tf.keras.optimizers.legacy.Adam(lr=lr)
        else:
            optimizer = tf.keras.optimizers.Adam(lr=lr)
            
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
    
    # Compile VAE
    if metrics:
        autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer=optimizer)
        
    else:
        autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), 
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
