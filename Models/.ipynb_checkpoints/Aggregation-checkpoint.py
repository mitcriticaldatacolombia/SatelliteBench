# Deep learning
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, concatenate

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

""" Create the Model """
def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape


def create_model(model, model_2, dense_acivation='relu'):

    # Create a Sequential Model
    input1 = model.inputs
    input2 = model_2.input
    
    out_1 = model(input1)
    out_2 = model_2(input2)
    
    concat_x = concatenate([out_1, out_2])

    #Final Layer
    x = Dense(2, activation=dense_acivation)(concat_x)
    output_layer = Dense(1)(x)

    #Model Definition 
    final_model = Model(inputs=[(input1,input2)],outputs=[output_layer])


    # Compile the model:
    opt = keras.optimizers.Adam()
    
    # Metrics
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        smape
    ]
    
    final_model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return final_model

def classification_aggregation(model, model_2, dense_acivation='relu'):

    # Create a Sequential Model
    input1 = model.inputs
    input2 = model_2.input
    
    out_1 = model(input1)
    out_2 = model_2(input2)
    
    concat_x = concatenate([out_1, out_2])

    #Final Layer
    x = Dense(6, activation=dense_acivation)(concat_x)
    output_layer = Dense(3, activation='softmax')(x)

    # Model Definition 
    final_model = Model(inputs=[(input1,input2)],outputs=[output_layer])

    # Compile the model:
    opt = keras.optimizers.Adam()
    
    # Metrics
    metrics = [
        tf.keras.metrics.AUC(name='auc'), #, multi_label=True, num_labels=3),
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tfa.metrics.F1Score(num_classes=3, threshold=0.5)
        ]
    
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
    

    return final_model