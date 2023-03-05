# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, BatchNormalization
import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

""" Create the Model for Regression """
def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape


def create_model(lstm_layers=[120, 240], nn_layers=[60, 1], sequence=3, features=1024, dense_acivation='relu', recurrent_cells='LSTM', bidirectional=False):
    
    # Create a Sequential Model
    model = Sequential()
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
    opt = keras.optimizers.Adam()
    
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
        

        
