""" Setup Environment """
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU

from tensorflow.keras.models import Model


import warnings
warnings.filterwarnings('ignore')


""" Constants: """

# Image size
target_size = (224, 224, 12)
latent_dim = 1024

learning_rate = 0.001
projection_units = 1024
temperature = 0.05


""" Contrastive Learning Model: """

def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=target_size, pooling="avg"
    )

    inputs = keras.Input(shape=target_size)
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="dengue_sat-encoder")
    return model

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(encoder):
    inputs = keras.Input(shape=target_size)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="dengue_sat-encoder_with_projection-head"
    )
    return model



def get_ContrastiveLearning(model_path, backbone=None):
    """ Encoder Part: """
    encoder = create_encoder()

    """ Latent Dimension """
    encoder_with_projection_head = add_projection_head(encoder)

    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupervisedContrastiveLoss(temperature),
    )

    if model_path:
        # Load weights
        encoder_with_projection_head.load_weights(model_path, by_name=True)
    
    if backbone:        
        encoder_with_projection_head.trainable = False

    return encoder_with_projection_head