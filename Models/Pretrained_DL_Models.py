import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50V2

def freeze_model(cnn):
    for idx, layer in enumerate(cnn.layers):
        layer.trainable = False # idx > len(cnn.layers) - 2 
    
    return model

def get_backbone(target_size, backbone, freeze=True):
    if backbone == 'MobileNetV2': # minimum size & parameters
        model = Sequential()
        cnn = MobileNetV2(input_shape=target_size, include_top=False, weights='imagenet')
        model.add(cnn)
        model.add(tf.keras.layers.GlobalAveragePooling2D())

    elif backbone == 'VGG16': # min depth
        model = Sequential()
        cnn = VGG16(input_shape=target_size, include_top=False, weights='imagenet')

        
    elif backbone == 'ResNet50V2':

        cnn = ResNet50V2(input_shape=target_size, include_top=False, weights='imagenet')
        model.add(cnn)
        model.add(tf.keras.layers.GlobalAveragePooling2D())  

    if backbone in ['MobileNetV2', 'VGG16', 'ResNet50V2']:
        model = Sequential()
        model.add(cnn)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        
        if freeze:
            model = freeze_model(model)

