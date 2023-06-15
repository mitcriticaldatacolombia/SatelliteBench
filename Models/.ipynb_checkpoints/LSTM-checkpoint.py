# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, BatchNormalization, TimeDistributed
import tensorflow_addons as tfa

from Models.Pretrained_DL_Models import get_backbone

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

""" Create the Model for Regression """
def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape

""" Add the Learnable vector """
class CustomTimeDistributed(tf.keras.layers.TimeDistributed):
    def call(self, inputs):
        # Check if all values in the input are 0
        if tf.reduce_all(tf.equal(inputs, 0)):
            # Return the input without passing it through the cnn_base layer
            return inputs
        
        # Pass the input through the cnn_base layer and the rest of the layers
        else:
            return super().call(inputs)
        
        
# Custom Layer for zero input check
class ZeroInputCheckLayer(tf.keras.layers.Layer):
    def __init__(self, projection, cnn_base):
        super(ZeroInputCheckLayer, self).__init__()
        self.projection = projection
        self.cnn_base = cnn_base
        self.learnable_vector = self.add_weight(
            name='learnable_vector',
            shape=(projection,),
            trainable=True,
            initializer='zeros',
            constraint=tf.keras.constraints.NonNeg()
        )

    def call(self, inputs):
        print('tf.shape(inputs)',tf.shape(inputs))
        
        if tf.reduce_all(inputs == 0):
            outs = tf.expand_dims(self.learnable_vector, axis=0)
            print('tf.shape(outs)',tf.shape(outs))
            return outs
        else:
            outs = self.cnn_base(inputs)
            print('tf.shape(outs)',tf.shape(outs))
            return outs

        

def create_model(lstm_layers=[120, 240], nn_layers=[60, 1], sequence=3, features=1024, dense_acivation='relu', recurrent_cells='LSTM', bidirectional=False, backbone='ResNet50V2', weights='imagenet', freeze=True, projection=1024, learnable_vector=False):
    
    # Create a Sequential Model
    model = Sequential()

    
    if type(features) == list or type(features) == tuple and sequence != 0:
        # Get backbone:
        # Possible options to backbone: # 'ViT' # 'ConvNeXtTiny' # 'ConvNeXtSmall' # 'ConvNeXtBase' # 'ResNet50V2' # 'VGG16' # 'MobileNetV2'
        # Possible options to weights: # 'imagenet' # None # 'sentinel_vae' # 'sentinel_ae'
        #backbone = 'ConvNeXtTiny' # 'ViT' # 'ConvNeXtTiny' # 'ConvNeXtSmall' # 'ConvNeXtBase' # 'ResNet50V2' # 'VGG16' # 'MobileNetV2'
        #weights = 'imagenet' # 'imagenet' # None # 'sentinel_vae' # 'sentinel_ae'
        #freeze = False
        cnn_base = get_backbone(features, backbone, freeze=freeze, weights=weights)
        
        if learnable_vector:
            model.add(TimeDistributed(ZeroInputCheckLayer(projection, cnn_base), input_shape=((sequence,) + features)))
        else:
            model.add(TimeDistributed(cnn_base, input_shape = ((sequence,) + features)))
        # Projection layer
        model.add(tf.keras.layers.TimeDistributed(Dense(projection)))
        # model.add(tf.keras.layers.LSTM(120, dropout=0.1, return_sequences=True))
        features = projection
    
    
    if sequence != 0:
        # Add LSTM Layers
        for i, lstm_layer in enumerate(lstm_layers):
            if i < (len(lstm_layers) - 1):
                if bidirectional:
                    if recurrent_cells == 'LSTM':
                        model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True)))
                    elif recurrent_cells == 'GRU':
                        model.add(Bidirectional(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True)))
                    else:
                        model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True)))
                else:
                    if recurrent_cells == 'LSTM':
                        model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True))
                    elif recurrent_cells == 'GRU':
                        model.add(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True))
                    else:
                        model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True))
                features = lstm_layer
            else:
                if bidirectional:
                    if recurrent_cells == 'LSTM':
                        model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features))))
                    elif recurrent_cells == 'GRU':
                        model.add(Bidirectional(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features))))
                    else:
                        model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features))))
                else:
                    if recurrent_cells == 'LSTM':
                        model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features)))
                    elif recurrent_cells == 'GRU':
                        model.add(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features)))
                    else:
                        model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features)))
    else:
        for i, nn_layer in enumerate(lstm_layers):
            if i < (len(nn_layers) - 1):
                model.add(Dense(nn_layer, activation=dense_acivation))
                model.add(BatchNormalization())
            else:
                model.add(Dense(nn_layer))
            
    model.add(BatchNormalization())
    # Add Dense Layers
    for i, nn_layer in enumerate(nn_layers):
        if i < (len(nn_layers) - 1):
            model.add(Dense(nn_layer, activation=dense_acivation))
            model.add(BatchNormalization())
        else:
            model.add(Dense(nn_layer))
        
    
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
    
    model.compile(loss='mse', optimizer=opt, metrics=metrics)
    
    return model


""" Create the Model for Classification """
def create_model_classification(lstm_layers=[120, 240], nn_layers=[60, 3], sequence=3, features=1024, dense_acivation='relu', recurrent_cells='LSTM', bidirectional=False):
    
    # Create a Sequential Model
    model = Sequential()

    # Add LSTM Layers
    for i, lstm_layer in enumerate(lstm_layers):
        if i < (len(lstm_layers) - 1):
            if bidirectional:
                if recurrent_cells == 'LSTM':
                    model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True)))
                elif recurrent_cells == 'GRU':
                    model.add(Bidirectional(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True)))
                else:
                    model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True)))
            else:
                if recurrent_cells == 'LSTM':
                    model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True))
                elif recurrent_cells == 'GRU':
                    model.add(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True))
                else:
                    model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features), return_sequences=True))
            features = lstm_layer
        else:
            if bidirectional:
                if recurrent_cells == 'LSTM':
                    model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features))))
                elif recurrent_cells == 'GRU':
                    model.add(Bidirectional(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features))))
                else:
                    model.add(Bidirectional(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features))))
            else:
                if recurrent_cells == 'LSTM':
                    model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features)))
                elif recurrent_cells == 'GRU':
                    model.add(GRU(lstm_layer, dropout=0.1, input_shape=(sequence, features)))
                else:
                    model.add(LSTM(lstm_layer, dropout=0.1, input_shape=(sequence, features)))
    model.add(BatchNormalization())
    # Add Dense Layers
    for i, nn_layer in enumerate(nn_layers):
        if i < (len(nn_layers) - 1):
            model.add(Dense(nn_layer, activation=dense_acivation))
            model.add(BatchNormalization())
        else:
            model.add(Dense(nn_layer, activation='softmax'))
        
    # Compile the model:
    if int(tf.__version__.split('.')[1]) >= 11:
        opt = tf.keras.optimizers.legacy.Adam()
    else:
        opt = tf.keras.optimizers.Adam()
    
    # Metrics
    metrics = [
        tf.keras.metrics.AUC(name='auc'), #, multi_label=True, num_labels=3),
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tfa.metrics.F1Score(num_classes=3, threshold=0.5)
        ]
    
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)
    

    return model



""" Train the Model """
def create_monitor(monitor_var='val_loss', min_delta=1e-3, patience=8, verbose=1, mode='auto', restore_best_weights=True):
    # EarlyStopping:
    monitor = EarlyStopping(monitor=monitor_var, min_delta=min_delta, patience=patience, 
                            verbose=verbose, mode=mode, restore_best_weights=restore_best_weights)
    return monitor

# fit network
def train_model(model, train_X, train_y, val_X, val_y, monitor, plot=None, epochs=50, batch_size=16, verbose=0, weights=None):
    if monitor:
        monitor = create_monitor()
        if weights:
            history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=verbose, shuffle=False, callbacks=[monitor], class_weight=weights)
        else:
            history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=verbose, shuffle=False, callbacks=[monitor], class_weight=weights)
    else:
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=verbose, shuffle=False)
    
    if plot:
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.title('Train - Validation Loss Plot')
        plt.legend()
        plt.show()
        

        
