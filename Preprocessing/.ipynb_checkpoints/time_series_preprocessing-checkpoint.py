# Data reading in Dataframe format and data preprocessing
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Linear algebra operations
import numpy as np 

# Machine learning models and preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Epiweek
from epiweeks import Week, Year

# Date
from datetime import date as convert_to_date

import warnings
warnings.filterwarnings('ignore')


cities =  {
  "76001": "Cali",
  "05001": "Medellín",
  "50001": "Villavicencio",
  "54001": "Cúcuta",
  "73001": "Ibagué",
  "68001": "Bucaramanga",
  "05360": "Itagüí",
  "08001": "Barranquilla",
  "41001": "Neiva",
  "23001": "Montería"
}
cities = {int(k):v for k,v in cities.items()}

codes =  {
  "Cali": "76001",
  "Medellín": "05001",
  "Villavicencio": "50001",
  "Cúcuta": "54001",
  "Ibagué": "73001",
  "Bucaramanga": "68001",
  "Itagüí": "05360",
  "Barranquilla": "08001",
  "Neiva": "41001",
  "Montería": "23001"
}


def get_code(city):
    str_code = codes[city]
    return str_code

def get_temperature_and_precipitation(city, features):
    
    if type(city) == int or (type(city) == np.int64):
        city = cities[city]
    elif type(city) == str and city.isdigit():
        city = cities[int(city)]
    
    code = get_code(city)

    # Precipitation
    for col in pd.read_csv(features[0]).columns:
        if code in col:
            column = col
            continue
    precipitation_df = pd.read_csv(features[0])[['LastDayWeek', column]]

    # Temperature
    for col in pd.read_csv(features[1]).columns:
        if code in col:
            column = col
            continue
    temperature_df = pd.read_csv(features[1])[['LastDayWeek', column]]

    # Merge:
    features_df = temperature_df.merge(precipitation_df, how='inner', on='LastDayWeek')

    features_df['LastDayWeek'] = features_df['LastDayWeek'].apply(epiweek_from_date)

    features_df = features_df.set_index('LastDayWeek')
    features_df.index.name = None

    return features_df



""" Read the file with Static Data  """
def read_static(path, Municipality = None):
    df = pd.read_csv(path)
    
    df = df.iloc[:,np.r_[:2, 10:14, 28:53]]
    
    pop_cols = ['Population2015', 'Population2016', 'Population2017', 'Population2018']
    df['population'] = df[pop_cols].mean(axis=1)
    df.drop(columns=pop_cols, inplace=True)
    
    df['Date'] = '2016-02-02'
    
    if 'Municipality code' in df.columns:
        df.rename(columns={'Municipality code':'Municipality Code'}, inplace=True)

    
    if df['Municipality Code'].dtype == 'int64':
        if type(Municipality) == str:
            if Municipality.isdigit():
                Municipality = int(Municipality)
            else:
                Municipality = int(codes[Municipality])
                
    if df['Municipality Code'].dtype == 'object':
        if type(Municipality) == int or (type(Municipality) == np.int64):
            Municipality = cities[Municipality]
        elif (type(Municipality) == str) and (Municipality.isdigit()):
            Municipality = cities[int(Municipality)]
    
    if Municipality:
        print(f'Obtaining dataframe for the city of {Municipality} - {cities[Municipality]} only...')
        df = df[df['Municipality Code'] == Municipality]
    
    df.Date = df.Date.apply(epiweek_from_date)
    
    df = df.sort_values(by=['Date'])
    
    df = df.set_index('Date')

    if Municipality:
        df.drop(columns=['Municipality Code','Municipality'], inplace=True)
        
    df.index.name = None
    
    return df
    

    
def get_municipality_name(municipality):
    if (type(municipality) == str) and (municipality.isdigit()):
        municipality = cities[int(municipality)]
    elif (type(municipality) == int) or (type(municipality) == np.int64):
        municipality = cities[municipality]
    return municipality

""" Get Epiweek as int """
def epiweek_from_date(image_date):
    date = image_date.split('-')
    
    # Get year as int
    year = ''.join(filter(str.isdigit, date[0]))
    year = int(year)
    
    # Get month as int
    month = ''.join(filter(str.isdigit, date[1]))
    month = int(month)
    
    # Get day as int
    day = ''.join(filter(str.isdigit, date[2]))
    day = int(day)
    
    # Get epiweek:
    date = convert_to_date(year, month, day)
    epiweek = str(Week.fromdate(date))
    epiweek = int(epiweek)
    
    return epiweek


""" Read the file with Embeddings  """
def read_features(path, Municipality = None):
    df = pd.read_csv(path)
    #df.Date = pd.to_datetime(df.Date)
    
    if 'Municipality code' in df.columns:
        df.rename(columns={'Municipality code':'Municipality Code'}, inplace=True)

    
    if df['Municipality Code'].dtype == 'int64':
        if type(Municipality) == str:
            if Municipality.isdigit():
                Municipality = int(Municipality)
            else:
                Municipality = int(codes[Municipality])
                
    if df['Municipality Code'].dtype == 'object':
        if type(Municipality) == int or (type(Municipality) == np.int64):
            Municipality = cities[Municipality]
        elif (type(Municipality) == str) and (Municipality.isdigit()):
            Municipality = cities[int(Municipality)]
    
    if Municipality:
        print(f'Obtaining dataframe for the city of {Municipality} - {cities[Municipality]} only...')
        df = df[df['Municipality Code'] == Municipality]
        
    df.Date = df.Date.apply(epiweek_from_date)
    
    df = df.sort_values(by=['Date'])
    
    df = df.set_index('Date')
    
    if Municipality:
        df.drop(columns=['Municipality Code'], inplace=True)
        
    df.index.name = None
    return df

""" Get epiweek as column name """
def get_epiweek(name):
    
    # Get week
    week = name.split('/')[1]
    week = week.replace('w','')
    week = int(week)
    
    # Year
    year = name.split('/')[0]
    year = int(year)
    
    epiweek = Week(year, week)
    
    epiweek = str(epiweek)
    epiweek = int(epiweek)

    return epiweek


""" Get labels"""
def read_labels(path, Municipality = None):
    df = pd.read_csv(path)
    if df.shape[1] > 678:
        df = pd.concat([df[['Municipality code', 'Municipality']], df.iloc[:,-676:]], axis=1)
        cols = df.iloc[:, 2:].columns
        new_cols = df.iloc[:, 2:].columns.to_series().apply(get_epiweek)
        df = df.rename(columns=dict(zip(cols, new_cols))) 
        
    if 'Label_CSV_All_Municipality' in path:
        # Get Columns
        df = df[['epiweek', 'Municipality code', 'Municipality', 'final_cases_label']]
        
        # change epiweek format
        df.epiweek = df.epiweek.apply(get_epiweek)
        
        # Remove duplicates
        df = df[df.duplicated(['epiweek','Municipality code','Municipality']) == False]
        
        # Replace Increase, decrease, stable to numerical:
        """
        - Decreased = 0
        - Stable = 1
        - Increased = 2 
        """
        df.final_cases_label = df.final_cases_label.replace({'Decreased': 0, 'Stable': 1, 'Increased': 2})
        
        # Create table
        df = df.pivot(index=['Municipality code', 'Municipality'], columns='epiweek', values='final_cases_label')

        # Reset Index:
        df = df.reset_index()
    
    if Municipality:
        
        if type(Municipality) == str:
            if Municipality.isdigit():
                Municipality = int(Municipality)
            else:
                Municipality = int(codes[Municipality])

        df = df[df['Municipality code'] == Municipality]
        df.drop(columns=['Municipality'], inplace=True)
        #df.rename(columns={'Municipality': 'Municipality Code'}, inplace=True)
    
        df = df.set_index('Municipality code')
        df = df.T

        df.columns.name = None
        df.index.name = None
        
        df.columns = ['Labels']
        
        df.index = pd.to_numeric(df.index)
    
    return df


def get_dengue_dataset(labels_path, embeddings_path, municipality, temp_prec=False, cases=None, limit=True, static=None):
    
    labels_df = read_labels(path=labels_path, Municipality=municipality)

    if limit:
        labels_df = labels_df[(labels_df.index > 201545) & (labels_df.index < 201901)]
    
    if (not embeddings_path) and (not cases) and (not static):
        return labels_df
    
    elif (not embeddings_path)  and (not static) and cases:
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        return features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
    
    elif (not embeddings_path)  and (not cases) and static:
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        
        return labels_df
    
    elif (not embeddings_path)  and cases and static:
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        
        return features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
    
    elif embeddings_path and cases and not(static):
        
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        
    elif embeddings_path and not(cases) and static:
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])

    elif embeddings_path and cases and static:
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        
    
    if temp_prec:
        features_df = get_temperature_and_precipitation(municipality, embeddings_path)
    else:
        features_df = read_features(path=embeddings_path, Municipality=municipality)
    
     
    # Merge the two dataframes based on the date values
    if cases:
        dengue_df = features_df.merge(features_extra_df, how='inner', left_index=True, right_index=True)
        dengue_df = dengue_df.merge(labels_df, how='inner', left_index=True, right_index=True)
    else:
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        
    return dengue_df


def train_test_split(df, train_percentage = 80):
    # We need a sequence so we can't split randomly
    # To divide into Train and test we have to calculate the train percentage of the dataset:
    size = df.shape[0]
    split = int(size*(train_percentage/100))
    
    """ Train """
    # We will train with 1st percentage % of data and test with the rest
    train_df = df.iloc[:split,:] ## percentage % train
    
    """ Test """
    test_df = df.iloc[split:,:] # 100 - percentage % test
    
    print(f'The train shape is: {train_df.shape}')
    print(f'The test shape is: {test_df.shape}')
    
    return train_df, test_df


""" Train-Test Split"""
def train_test_split(df, train_percentage = 80):
    # We need a sequence so we can't split randomly
    # To divide into Train and test we have to calculate the train percentage of the dataset:
    size = df.shape[0]
    split = int(size*(train_percentage/100))
    
    """ Train """
    # We will train with 1st percentage % of data and test with the rest
    train_df = df.iloc[:split,:] ## percentage % train
    
    """ Test """
    test_df = df.iloc[split:,:] # 100 - percentage % test
    
    print(f'The train shape is: {train_df.shape}')
    print(f'The test shape is: {test_df.shape}')
    
    return train_df, test_df

""" Normalization"""
# Normalize train data and create the scaler
def normalize_train_features(df, feature_range=(-1, 1), scaler=True, describe=None):
    
    scalers = {}
    # For each column in the dataframe
    for i, column in enumerate(df.columns):
        if not scaler:
            if (i == len(df.columns) - 1):
                continue
        
        # Get values of the column
        values = df[column].values.reshape(-1,1)
        # Generate a new scaler
        scaler = MinMaxScaler(feature_range=feature_range)
        # Fit the scaler just for that column
        scaled_column = scaler.fit_transform(values)
        # Add the scaled column to the dataframe
        scaled_column = np.reshape(scaled_column, len(scaled_column))
        df[column] = scaled_column
        
        # Save the scaler of the column
        scalers['scaler_' + column] = scaler
    if describe:
        print(f' Min values are: ')
        print(df.min())
        print(f' Max values are: ')
        print(df.max())
        
    return df, scalers


""" If you want to use the same scaler used in train, you can use this function"""
def normalize_test_features(df, scalers=None, scaler=True, describe=None):
    
    if not scalers:
        raise TypeError("You should provide a list of scalers.")
        
    for i, column in enumerate(df.columns):
        if not scaler:
            if (i == len(df.columns) - 1):
                continue
        
        # Get values of the column
        values = df[column].values.reshape(-1,1)
        # Take the scaler of that column
        scaler = scalers['scaler_' + column]
        # Scale values
        scaled_column = scaler.transform(values)
        scaled_column = np.reshape(scaled_column,len(scaled_column))
        # Add the scaled values to the df
        df[column] = scaled_column
    if describe:
        print(f' Min values are: ')
        print(df.min())
        print(f' Max values are: ')
        print(df.max())
        
    return df 


# prepare data for time series
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, autoregressive=True):
    no_autoregressive = not(autoregressive)
    if no_autoregressive:
        n_in = n_in - 1
        
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        if no_autoregressive:
            cols.append(df.shift(i).iloc[:,:-1])
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
        else:
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

""" Features and Labels """
def features_labels_set(timeseries_data, original_df, autoregressive):
    
    """ Features """
    # We define the number of features as (Cases and media cloud)
    n_features = original_df.shape[1]

    # The features to train the model will be all except the values of the actual week 
    # We can't use other variables in week t because whe need to resample a a 3D Array
    if autoregressive:
        features_set = DataFrame(timeseries_data.values[:,:-n_features])
    else:
        features_set = DataFrame(timeseries_data.values[:,:-1])
    # Convert pandas data frame to np.array to reshape as 3D Array
    features_set = features_set.to_numpy()
    print(f'The shape of the features is {features_set.shape}')
    
    """ Labels """
    # We will use Covid cases in last week 
    labels_set = DataFrame(timeseries_data.values[:,-1])
    # Convert pandas data frame to np.array
    labels_set = labels_set.to_numpy()
    print(f'The shape of the labels is {labels_set.shape}')
    
    return features_set, labels_set, n_features


def reshape_tensor(train_X, test_X, n_features, days, autoregressive=True):
    print('The initial shapes are:')
    print(f'The train shape is {train_X.shape}')
    print(f'The test shape is {test_X.shape}')
    
    # reshape input to be 3D [samples, timesteps, features]
    if not(autoregressive):
        train_X = train_X.reshape((train_X.shape[0], days, n_features-1))
        test_X = test_X.reshape((test_X.shape[0], days, n_features-1))
    
    else:
        train_X = train_X.reshape((train_X.shape[0], days, n_features))
        test_X = test_X.reshape((test_X.shape[0], days, n_features))
    
    print('-----------------------')
    print('The Final shapes are:')
    print(f'The train shape is {train_X.shape}')
    print(f'The test shape is {test_X.shape}')
    
    return train_X, test_X

def preprocess_dataset_to_time_series(df, train_percentage = 80, feature_range = (-1, 1), T=3, autoregressive = False, normalize=True, reshape=True):
    """ Train-Test Split"""
    train_df, test_df = train_test_split(df, train_percentage = train_percentage)
    """ Normalization """
    if normalize:
        # Train:
        train_df, scalers = normalize_train_features(train_df, feature_range=feature_range)
        # Test:
        test_df = normalize_test_features(test_df, scalers=scalers)
    """ Generate Time Frame"""
    # Train:
    train = series_to_supervised(train_df, n_in=T, autoregressive=autoregressive)
    # Test:
    test = series_to_supervised(test_df, n_in=T, autoregressive=autoregressive)
    """ Features and Labels set"""
    # Train:
    train_X, train_y, n_features = features_labels_set(timeseries_data=train, original_df=df, autoregressive=autoregressive)
    # Test:
    test_X, test_y, n_features = features_labels_set(timeseries_data=test, original_df=df, autoregressive=autoregressive)
    """ Reshape """
    if reshape:
        # reshape input to be 3D [samples, timesteps, features]
        train_X, test_X = reshape_tensor(train_X, test_X, n_features, T, autoregressive)
    if normalize:
        return train_X, test_X, train_y, test_y, scalers
    else:
        return train_X, test_X, train_y, test_y