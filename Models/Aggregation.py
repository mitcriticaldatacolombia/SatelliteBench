# Deep learning
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, Model, layers
from tensorflow.keras.layers import Dense, concatenate, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import matplotlib.pyplot as plt

""" Create the Model """
def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape

class Attn_Net_Gated(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()

        self.attention_a = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_b = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_a = tf.keras.Sequential(self.attention_a)
        self.attention_b = tf.keras.Sequential(self.attention_b)

        self.attention_c = tf.keras.layers.Dense(n_classes)

    def call(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = tf.multiply(a, b)
        A = self.attention_c(A)
        return A

class Fusion_Net(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Fusion_Net, self).__init__()

        self.attention_a = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_b = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_a = tf.keras.Sequential(self.attention_a)
        self.attention_b = tf.keras.Sequential(self.attention_b)

        self.attention_c = tf.keras.layers.Dense(n_classes)

    def call(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = tf.multiply(a, b)
        A = self.attention_c(A)
        return A

# Defines the Kronecker product
from tensorflow.keras.layers import Layer
class OuterProductLayer(Layer):
    def __init__(self):
        super(OuterProductLayer, self).__init__()

    def call(self, inputs):
        # Assuming inputs is a tuple of two tensors
        output1, output2 = inputs
        ones_tensor = tf.ones((tf.shape(output1)[0], 1), dtype=output1.dtype)
        ones_tensor = tf.stop_gradient(ones_tensor)
        # Concatenate the ones tensor with the input vectors
        output1 = tf.concat([ones_tensor, output1], axis=1)
        ones_tensor2 = tf.ones((tf.shape(output1)[0], 1), dtype=output1.dtype)
        ones_tensor2 = tf.stop_gradient(ones_tensor2)
        # Concatenate the ones tensor with the input vectors
        output2 = tf.concat([ones_tensor, output2], axis=1)
        outer_product = tf.einsum('bi,bj->bij', output1, output2)
        return Flatten()(outer_product)

def create_aggregation_model(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):

    L=128
    D=64

    print(model.summary(), "model1 sum")

    if fusion == 'early' or 'late':
        modelc = Sequential(model.layers[:-1])
        modelc.trainable = False
        model_2c = Sequential(model_2.layers[:-1])
        model_2c.trainable = False
        if model_3:
            model_3 = Sequential(model_3.layers[:-1])
            model_3.trainable = False

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)
    # Create a Sequential Model
    input1 = modelc.inputs
    input2 = model_2c.input
    out_1 = modelc(input1)
    out_2 = model_2c(input2)
    out_1c = model(input1)
    out_2c = model_2(input2)

    fusion_features = OuterProductLayer()((out_1,out_2))
    print(fusion_features.shape)


    # Assuming self.fusion_fc is a Keras layer
    x = Dense(256, activation=dense_acivation)(fusion_features)

    x = Dense(128, activation=dense_acivation)(x)
    x = concatenate([x,out_1c,out_2c])
    x = Dense(64, activation=dense_acivation)(x)

    x = Dense(32, activation=dense_acivation)(x)

    x = Dense(16, activation= dense_acivation)(x)
    output_layer = Dense(1)(x)

    #Model Definition
    final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])


    # Compile the model:
    if int(tf.__version__.split('.')[1]) >= 11:
        opt = tf.keras.optimizers.legacy.Adam(lr=0.001)
    else:
        opt = tf.keras.optimizers.Adam(lr=0.001)

    # Metrics
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        smape
    ]

    final_model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return final_model

def create_aggregation_model_attention(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):

    L=128
    D=64


    if fusion == 'early' or 'late':

        modelc = Sequential(model.layers[:-1])
        modelc.trainable = False
        model_2c = Sequential(model_2.layers[:-1])
        model_2c.trainable = False
        if model_3:

            model_3 = Sequential(model_3.layers[:-1])
            model_3.trainable = False

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)

    input1 = modelc.inputs
    input2 = model_2c.input
    out_1 = modelc(input1)
    out_2 = model_2c(input2)
    out_1f = model(input1)
    out_2f = model_2(input2)

    h = tf.keras.layers.Dense(L)(out_1)
    A = Attn_Net_Gated(L, D, dropout=0.2, n_classes=1)(out_1)

    A = tf.nn.softmax(A, axis=1)
    atten_out = A*out_1
    fusion_features = OuterProductLayer()((atten_out,out_2))
    # Assuming self.fusion_fc is a Keras layer
    x = Dense(256, activation=dense_acivation)(fusion_features)
    x = Dense(128, activation=dense_acivation)(x)
    x = concatenate([x,out_1f,out_2f])
    x = Dense(64, activation=dense_acivation)(x)
    x = Dense(32, activation=dense_acivation)(x)
    x = Dense(16, activation=dense_acivation)(x)
    output_layer = Dense(1)(x)
    #Model Definition
    final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])
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

def classification_aggregation(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):


    # Create a Sequential Model
    input1 = model.inputs
    input2 = model_2.input

    if fusion == 'early' or 'joint':
        model.layers.pop()
        model_2.layers.pop()
        if model_3:
            model_3.layers.pop()

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)

    out_1 = model(input1)
    out_2 = model_2(input2)

    if model_3:
        concat_x = concatenate([out_1, out_2, out_3])
    else:
        concat_x = concatenate([out_1, out_2])

    #Final Layer
    x = Dense(6, activation=dense_acivation)(concat_x)
    output_layer = Dense(3, activation='softmax')(x)

    #Model Definition
    if model_3:
        final_model = Model(inputs=[(input1, input2, input3)], outputs=[output_layer])
    else:
        final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])

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