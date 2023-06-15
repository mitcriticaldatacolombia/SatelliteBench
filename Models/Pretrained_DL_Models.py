import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50V2
from tensorflow.keras.applications.convnext import ConvNeXtBase, ConvNeXtSmall, ConvNeXtTiny
from keras_cv.models import ViTTiny16

from huggingface_hub import from_pretrained_keras
from transformers import TFViTModel
import os

import warnings
warnings.filterwarnings('ignore')

def freeze_model(cnn):
    for idx, layer in enumerate(cnn.layers):
        layer.trainable = False # idx > len(cnn.layers) - 2 
    
    return cnn

def get_backbone(target_size, backbone, freeze=True, weights='imagenet'):
    
    if backbone == 'MobileNetV2':
        cnn = MobileNetV2(input_shape=target_size, include_top=False, weights=weights)

    elif backbone == 'VGG16':
        cnn = VGG16(input_shape=target_size, include_top=False, weights=weights)
        
    elif backbone == 'ResNet50V2':
        if weights == 'sentinel_vae':
            # Import the model from Hugging face
            model = from_pretrained_keras("MITCriticalData/Sentinel-2_Resnet50V2_VariationalAutoencoder_RGB_full_Colombia_Dataset")
            # From the model select just the Resnet50 Layer
            cnn = model.layers[1].layers[1]
            
        elif weights == 'sentinel_ae':
            model = from_pretrained_keras("MITCriticalData/Sentinel-2_Resnet50V2_Autoencoder_RGB_full_Colombia_Dataset")
            # From the model select just the Resnet50 Layer
            cnn = model.layers[1].layers[1]

        else:
            cnn = ResNet50V2(input_shape=target_size, include_top=False, weights=weights)

    elif backbone == 'ConvNeXtBase':
        cnn = ConvNeXtBase(input_shape=target_size, include_top=False, weights=weights)
    
    elif backbone == 'ConvNeXtSmall':
        cnn = ConvNeXtSmall(input_shape=target_size, include_top=False, weights=weights)
    
    elif backbone == 'ConvNeXtTiny':
        #if weights == 'sentinel_vae':

        if weights == 'sentinel_ae':
            # Import the model from Hugging face
            model = from_pretrained_keras("MITCriticalData/Sentinel-2_ConvNeXtTiny_Autoencoder_RGB_full_Colombia_Dataset")
            # From the model select just the Resnet50 Layer
            cnn = model.layers[1].layers[1]
        else:
            cnn = ConvNeXtTiny(input_shape=target_size, include_top=False, weights=weights)  
    
    elif backbone == 'ViT':
        if weights == 'sentinel_ae':
            os.system("git clone https://huggingface.co/MITCriticalData/Sentinel-2_ViT_Autoencoder_RGB_full_Colombia_Dataset")
            model = tf.keras.models.load_model('Sentinel-2_ViT_Autoencoder_RGB_full_Colombia_Dataset', custom_objects={"TFViTModel": TFViTModel})
            
            cnn = tf.keras.Sequential()
            for layer in model.layers[:-1]: # just exclude last layer from copying
                cnn.add(layer)
        else:
            inputs = tf.keras.layers.Input(shape=target_size)
            cnn = ViTTiny16(
                    include_rescaling=False,
                    include_top=False,
                    name="ViTTiny32",
                    weights=weights,
                    input_tensor=inputs,
                    pooling="token_pooling",
                    activation=tf.keras.activations.gelu,
                )
            
            cnn.trainable = True
            
    
    
    if not weights:
        freeze = False
    
    model = Sequential()
    model.add(cnn)
    if backbone != 'ViT':
        model.add(tf.keras.layers.GlobalAveragePooling2D())

    if freeze:
        #cnn.trainable = False
        model = freeze_model(model)
    
    return model
