 # Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, BatchNormalization
from tensorflow.keras import layers
import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

""" Create the Model for Regression """
def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0,):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for i, dim in enumerate(mlp_units):
        if i < (len(mlp_units) - 1):
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        else:
            outputs = layers.Dense(dim)(x)
                
    return keras.Model(inputs, outputs)


def create_model(head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, nn_layers=[60, 1], sequence=3, features=1024, dropout=0.2):
    
    model = build_model(input_shape=(sequence, features), head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_transformer_blocks=num_transformer_blocks, mlp_units=nn_layers, dropout=dropout, mlp_dropout=dropout)
        
    
    # Compile the model:
    opt = keras.optimizers.Adam(lr=0.0003)
    
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
    opt = keras.optimizers.Adam()
    
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
def create_monitor(monitor_var='val_loss', min_delta=1e-3, patience=20, verbose=1, mode='auto', restore_best_weights=True):
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
        
