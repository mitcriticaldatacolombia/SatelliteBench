""" Test the Model """
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from math import sqrt
from numpy import concatenate

from Preprocessing.time_series_preprocessing import get_dengue_dataset, get_municipality_name
from Preprocessing.time_series_preprocessing import preprocess_dataset_to_time_series

from Models.LSTM import create_model, train_model, create_model_classification
from Models.Transformer import create_model as create_transformer_model
from Models.Transformer import create_model as create_transformer_model_classification
from Models.Aggregation import create_aggregation_model as create_aggregation_model
from Models.Aggregation import classification_aggregation

from keras.layers import Input, Dense, concatenate
from keras.models import Model
from tensorflow.keras.utils import to_categorical


""" Generate predictions """
def test_model(model, test_X, test_y, scaler, rnn = True, classification=None):
    
    # If model is a classical machine learning model and test_X is a 3D tensor, then convert to 2D
    if not rnn and (len(test_X.shape) == 3):
        test_X = test_X.reshape((test_X.shape[0], -1))
    
    # do the prediction
    yhat = model.predict(test_X)
    
    # Invert scaling for forecast
    # Inverse Scaler
    
    # Predicted
    if not rnn:
        yhat = yhat.reshape(-1, 1)
        
    if not scaler:
        return yhat, test_y
    
    if classification:
        yhat = np.argmax(yhat, axis=1)
        yhat = yhat - 1
        yhat = yhat.reshape(-1, 1)
    
    inv_yhat = scaler.inverse_transform(yhat)
    
    # Real:
    inv_y = scaler.inverse_transform(test_y)
    
    return inv_yhat, inv_y

""" MAPE """
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Test MAPE: %.3f' % mape)
    return mape

""" sMAPE """
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    smape = 1/len(y_true) * np.sum(2 * np.abs(y_pred-y_true) / (np.abs(y_true) + np.abs(y_pred))*100)
    print('Test sMAPE: %.3f' % smape)
    return smape

""" MAE """
def Mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print('Test MAE: %.3f' % mae)
    return mae

""" R Squared """
def r_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r2 = r2_score(y_true, y_pred)
    print('Test R Squared: %.3f' % r2)
    return r2
    
""" RMSE """
def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print('Test RMSE: %.3f' % rmse)
    return rmse


""" Plot Time Series of Predictions"""
def plot_predictions(inv_y, inv_yhat, model_name = ''):

    dataY_plot = inv_y  ##  real test-target cases
    dataY_plot = dataY_plot.reshape(len(dataY_plot), 1)
    plt.plot(dataY_plot, label = 'actual')
    
    
    NoneType = type(None)
    if type(inv_yhat) != NoneType:
        data_predict = inv_yhat  ## predicted target cases
        data_predict = data_predict.reshape(len(data_predict), 1)
        plt.plot(data_predict, label = 'predicted')
    
    plt.legend(loc="upper left")
    
    if model_name == 'Train':
        plt.suptitle(f'Time-Series train Data')
    else:
        plt.suptitle(f'Time-Series Prediction with {model_name}')
    plt.show()
    
    
""" Evaluate """
def evaluate(model, test_X, test_y, scaler):
    stored_results = {}
    
    inv_yhat_lstm, inv_y_lstm = test_model(model=model, test_X=test_X, test_y=test_y, scaler=scaler)
    
    stored_results['mae'] = Mean_absolute_error(inv_y_lstm, inv_yhat_lstm)
    stored_results['rmse'] = root_mean_squared_error(inv_y_lstm, inv_yhat_lstm)
    stored_results['smape'] = symmetric_mean_absolute_percentage_error(inv_y_lstm, inv_yhat_lstm)
    stored_results['r2'] = r_squared(inv_y_lstm, inv_yhat_lstm)

    return stored_results, inv_yhat_lstm, inv_y_lstm

def evaluate_classification(model, test_X, test_y, scaler, verbose = True):
    
    
    inv_yhat_lstm, inv_y_lstm = test_model(model=model, test_X=test_X, test_y=test_y, scaler=scaler, classification=True)
    
    test_y = to_categorical(test_y+1, 3)
    
    if verbose:
        print('Evaluate: ')
    result = model.evaluate(test_X, test_y)
    stored_results = {}
    for i, metric in enumerate(model.metrics_names):
        stored_results[metric] = result[i]
        if verbose:
            print(f'{metric}: {result[i]}')
    return stored_results, inv_yhat_lstm, inv_y_lstm



"""  Clculate mean and std fot a Regression Model  """
def calculate_mean_std(lstm_layers, nn_layers, sequence, features, dense_acivation, recurrent_cells, bidirectional, train_X, train_y, test_X, test_y, scaler, monitor=True, plot=None, epochs=50, batch_size=16, n_tests=3, plot_samples=False, Municipality=None, fusion=None, x_ar=None, x_ar_test=None, x_ar_2=None, x_ar_test_2=None, head_size=256, num_heads=4, ff_dim=4, dropout=0.2, backbone='ResNet50V2', weights='imagenet', freeze=True, projection=1024, learnable_vector=False):
    
    metrics = {
        "rmse": [],
        "mae": [],
        "smape": [],
        "r2": []
    }
    
    plot_predictions(train_y, None, model_name = 'Train')
    
    for i in range(n_tests):
        """ Create the Model """
        if recurrent_cells != 'Transformer':
            model = create_model(lstm_layers, nn_layers, sequence, features, dense_acivation, recurrent_cells, bidirectional, backbone=backbone, weights=weights, freeze=freeze, projection=projection, learnable_vector=learnable_vector)
        else:
            model = create_transformer_model(head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_transformer_blocks=4, nn_layers=nn_layers, sequence=sequence, features=features, dropout=dropout)
        

        
        if fusion == 'late' or fusion == 'early':
            """ Train the Model 1 (Images) """
            train_model(model, train_X, train_y, test_X, test_y, monitor, plot, epochs, batch_size)
            
            """ Create the Model 2 """
            if recurrent_cells != 'Transformer':
                model_2 = create_model(lstm_layers, nn_layers, x_ar.shape[1], x_ar.shape[2], dense_acivation, recurrent_cells, bidirectional, backbone=backbone, weights=weights, freeze=freeze, projection=projection, learnable_vector=learnable_vector)
            else:
                model_2 = create_transformer_model(head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_transformer_blocks=4, nn_layers=nn_layers, sequence=x_ar.shape[1], features=x_ar.shape[2], dropout=dropout)
            
            """ Train the Model 2 """
            train_model(model_2, x_ar, train_y, x_ar_test, test_y, monitor, plot, epochs, batch_size)
            
            none_type = type(None)
            if (type(x_ar_2) != none_type) and (type(x_ar_test_2) != none_type):
                """ Create the Model 3 """
                if recurrent_cells != 'Transformer':
                    model_3 = create_model(lstm_layers, nn_layers, x_ar_2.shape[1], x_ar_2.shape[2], dense_acivation, recurrent_cells, bidirectional, backbone=backbone, weights=weights, freeze=freeze, projection=projection, learnable_vector=learnable_vector)
                else:
                    model_3 = create_transformer_model(head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_transformer_blocks=4, nn_layers=nn_layers, sequence=x_ar_2.shape[1], features=x_ar_2.shape[2], dropout=dropout)
                
                """ Train the Model 3 """
                train_model(model_3, x_ar_2, train_y, x_ar_test_2, test_y, monitor, plot, epochs, batch_size)
                
            # Freeze Models:
            model.trainable = False 
            model_2.trainable = False 
            
            if (type(x_ar_2) != none_type) and (type(x_ar_test_2) != none_type):
                model_3.trainable = False 
                # Merging models
                final_model = create_aggregation_model(model, model_2, model_3, fusion=fusion)
            else:
                # Merging models
                final_model = create_aggregation_model(model, model_2, fusion=fusion)
            
            train_model(final_model, [train_X, x_ar], train_y, [test_X, x_ar_test], test_y, monitor, plot, epochs, batch_size)
            
        if fusion == 'joint':
            """ Create the Model """
            if recurrent_cells != 'Transformer':
                model_2 = create_model(lstm_layers, nn_layers, x_ar.shape[1], x_ar.shape[2], dense_acivation, recurrent_cells, bidirectional, backbone=backbone, weights=weights, freeze=freeze, projection=projection, learnable_vector=learnable_vector)
            else:
                model_2 = create_transformer_model(head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_transformer_blocks=4, nn_layers=nn_layers, sequence=x_ar.shape[1], features=x_ar.shape[2], dropout=dropout)

            
            none_type = type(None)
            if (type(x_ar_2) != none_type) and (type(x_ar_test_2) != none_type):
                """ Create the Model 3 """
                if recurrent_cells != 'Transformer':
                    model_3 = create_model(lstm_layers, nn_layers, x_ar_2.shape[1], x_ar_2.shape[2], dense_acivation, recurrent_cells, bidirectional, backbone=backbone, weights=weights, freeze=freeze, projection=projection, learnable_vector=learnable_vector)
                else:
                    model_3 = create_transformer_model(head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, num_transformer_blocks=4, nn_layers=nn_layers, sequence=x_ar_2.shape[1], features=x_ar_2.shape[2], dropout=dropout)
                
                # Merging models
                final_model = create_aggregation_model(model, model_2, model_3, fusion=fusion)            
            else:
                # Merging models
                final_model = create_aggregation_model(model, model_2, fusion=fusion)
            
            train_model(final_model, [train_X, x_ar], train_y, [test_X, x_ar_test], test_y, monitor, plot, epochs, batch_size)
            
        else:
            """ Train the Model 1 (Images)"""
            train_model(model, train_X, train_y, test_X, test_y, monitor, plot, epochs, batch_size)
        
        """ Evaluate the Model """
        if fusion:
            final_model.summary()
            stored_results, inv_yhat_lstm, inv_y_lstm = evaluate(final_model, [test_X, x_ar_test], test_y, scaler)
            print(stored_results)
        else:
            stored_results, inv_yhat_lstm, inv_y_lstm = evaluate(model, test_X, test_y, scaler)
            print(stored_results)
            
        
        """ Get Metrics """
        for key in metrics.keys():
            metrics[key].append(stored_results[key])
        if plot_samples:
            plot_predictions(inv_y_lstm, inv_yhat_lstm, model_name = f'{Municipality} test {i}')
    
    
    """ Calculate Mean and Standard Deviation """    
    for key in metrics.keys():
        results = metrics[key]
        print(key, f": average={np.average(results):.3f}, std={np.std(results):.3f}")
        
    """ Time Series Plot """ 
    if not(plot_samples):
        plot_predictions(inv_y_lstm, inv_yhat_lstm, model_name = f'{Municipality} test {n_tests}')
        
    return [np.average(metrics["rmse"]), np.std(metrics["rmse"])], [np.average(metrics["mae"]), np.std(metrics["mae"])], [np.average(metrics["smape"]), np.std(metrics["smape"])], [np.average(metrics["r2"]), np.std(metrics["r2"])]




"""  Clculate mean and std fot a Classification Model  """
def calculate_mean_std_classification(lstm_layers, nn_layers, sequence, features, dense_acivation, recurrent_cells, bidirectional, train_X, train_y, test_X, test_y, scaler, monitor=True, plot=None, epochs=50, batch_size=16, n_tests=3, plot_samples=False, Municipality=None, fusion=None, x_ar=None, x_ar_test=None, weights=None):
    
    
    plot_predictions(train_y, None, model_name = 'Train')
    
    test_y_lab = test_y
    train_y = to_categorical(train_y+1, 3)
    test_y = to_categorical(test_y+1, 3)
    
    metrics = {
        "auc": [],
        "acc": [],
        "f1_score": []
    }
    
    for i in range(n_tests):
        """ Create the Model """
        model = create_model_classification(lstm_layers, nn_layers, sequence, features, dense_acivation, recurrent_cells, bidirectional)
        
        if fusion == 'late':
            """ Train the Model 1 (Images) """
            train_model(model, train_X, train_y, test_X, test_y, monitor, plot, epochs, batch_size, weights)
            
            """ Create the Model """
            model_2 = create_model_classification(lstm_layers, nn_layers, x_ar.shape[1], x_ar.shape[2], dense_acivation, recurrent_cells, bidirectional)
            """ Train the Model 2 """
            train_model(model_2, x_ar, train_y, x_ar_test, test_y, monitor, plot, epochs, batch_size, weights)
            
            # Freeze Models:
            model.trainable = False 
            model_2.trainable = False 
            
            # Merging models
            final_model = classification_aggregation(model, model_2, fusion=fusion)
            
            train_model(final_model, [train_X, x_ar], train_y, [test_X, x_ar_test], test_y, monitor, plot, epochs, batch_size, weights)
            
        if fusion == 'joint':
            """ Create the Model """
            model_2 = create_model_classification(lstm_layers, nn_layers, x_ar.shape[1], x_ar.shape[2], dense_acivation, recurrent_cells, bidirectional)
            
            # Merging models
            final_model = classification_aggregation(model, model_2, fusion=fusion)
            
            train_model(final_model, [train_X, x_ar], train_y, [test_X, x_ar_test], test_y, monitor, plot, epochs, batch_size, weights)
            
        else:
            """ Train the Model 1 (Images)"""
            train_model(model, train_X, train_y, test_X, test_y, monitor, plot, epochs, batch_size, weights)
        
        """ Evaluate the Model """
        if fusion:
            final_model.summary()
            stored_results, inv_yhat_lstm, inv_y_lstm = evaluate_classification(final_model, [test_X, x_ar_test], test_y_lab, scaler)
            print(stored_results)
        else:
            stored_results, inv_yhat_lstm, inv_y_lstm = evaluate_classification(model, test_X, test_y_lab, scaler)
            print(stored_results)
            
        
        """ Get Metrics """
        for key in metrics.keys():
            metrics[key].append(stored_results[key])
        if plot_samples:
            plot_predictions(inv_y_lstm-1, inv_yhat_lstm-1, model_name = f'{Municipality} test {i}')
    
    
    """ Calculate Mean and Standard Deviation """    
    for key in metrics.keys():
        results = metrics[key]
        print(key, f": average={np.average(results):.3f}, std={np.std(results):.3f}")
        
    """ Time Series Plot """ 
    if not(plot_samples):
        plot_predictions(inv_y_lstm, inv_yhat_lstm, model_name = f'{Municipality} test {n_tests}')
        
    return [np.average(metrics["auc"]), np.std(metrics["auc"])], [np.average(metrics["acc"]), np.std(metrics["acc"])], [np.average(metrics["f1_score"]), np.std(metrics["f1_score"])] 




"""  Evaluate the Model  """
def evaluate_lstm_for_city(labels, embeddings, Municipality, train_percentage, T, autoregressive, lstm_layers, nn_layers, dense_acivation, recurrent_cells, bidirectional, monitor=False, plot=True, epochs=100, batch_size=16, n_tests=3, plot_samples=True, temp_prec=False, fusion=None, classification=None, static=None):
    
    Municipality_name = get_municipality_name(Municipality)
    
    print('#'*100)
    if embeddings and (type(embeddings) == str):
        print('Embeddings: '.center(100, "-"))
        print(embeddings.center(100, "-"))
    elif type(embeddings) == list:
        print('Autoregressive Model: '.center(100, "-"))        
    else:
        print('Autoregressive Model: '.center(100, "-"))
    print('Municipality: '.center(100, "-"))
    print(Municipality_name.center(100, "-"))
    print('#'*100)
    
    """ Read Data """
    ### Read Data ###
    dengue_df = get_dengue_dataset(labels, embeddings, Municipality, temp_prec=temp_prec, static=static)
    
    """ Preprocess Dataset """
    ### Preprocessing for time series ###
    train_X, test_X, train_y, test_y, scalers = preprocess_dataset_to_time_series(dengue_df, train_percentage=train_percentage, T=T, autoregressive=autoregressive)
    
    """ Prepare Data Fusion """
    if fusion and (T != 0):
        if embeddings and static and temp_prec:
            # Modality 2
            x_ar = train_X[:, :, -28:]
            x_ar_test = test_X[:, :, -28:]
            # Modality 1
            train_X = train_X[:, :, :-28]
            test_X = test_X[:, :, :-28]
        
        if embeddings and static and not(temp_prec):
            # Modality 2
            x_ar = train_X[:, :, -26:]
            x_ar_test = test_X[:, :, -26:]
            # Modality 1
            train_X = train_X[:, :, :-26]
            test_X = test_X[:, :, :-26]
        
        if embeddings and not(static) and temp_prec:
            # Modality 2
            x_ar = train_X[:, :, -2:]
            x_ar_test = test_X[:, :, -2:]
            # Modality 1
            train_X = train_X[:, :, :-2]
            test_X = test_X[:, :, :-2]
        else:
            x_ar = None
            x_ar_test = None
    else:
        x_ar = None
        x_ar_test = None
    
    
    # Use the Function:
    if T != 0: 
        sequence=train_X.shape[1]
        features=train_X.shape[2]
    else:
        sequence=T
        features=train_X.shape[1]

    
    if not classification:
        rmse, mae, smape, r2 = calculate_mean_std(lstm_layers, nn_layers, sequence, features, dense_acivation, recurrent_cells, bidirectional, train_X, train_y, test_X, test_y, scalers['scaler_Labels'], monitor=monitor, plot=plot, epochs=epochs, batch_size=batch_size, n_tests=n_tests, plot_samples=plot_samples, Municipality='Global', fusion=fusion, x_ar=x_ar, x_ar_test=x_ar_test, backbone=backbone, weights=weights, freeze=freeze, projection=projection, learnable_vector=learnable_vector)
        
    else:
        auc, acc, f1 = calculate_mean_std_classification(lstm_layers, nn_layers, sequence, features, dense_acivation, recurrent_cells, bidirectional, train_X, train_y, test_X, test_y, scalers['scaler_Labels'], monitor, plot, epochs, batch_size, n_tests, plot_samples, Municipality=Municipality, fusion=fusion, x_ar=x_ar, x_ar_test=x_ar_test)
    
    print('#'*100)
    print(' End '.center(100, "-"))
    print('#'*100)
    
    if not classification:
        return rmse, mae, smape, r2
    else:
        return auc, acc, f1