# Data reading in Dataframe format and data preprocessing
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Read the images
from skimage import io
from skimage.transform import resize
from skimage.util import crop

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

# OS
import os

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
    
    dictionary = {}
    
    for cols in features_df.columns:
        if 'temperature' in cols:
            dictionary[cols] = 'temperature'
        if 'precipitation' in cols:
            dictionary[cols] = 'precipitation'
    features_df.rename(columns=dictionary, inplace=True)

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


""" Generate a CSV with the images: """
def read_image(path, crop=True, target_size=(224,224,3), BANDS='RGB', BAND=0):
    if os.path.isdir(path):
        return np.nan
    image_test = io.imread(path)
        
    if crop:
        x1 = target_size[0] // 2
        x2 = target_size[0] - x1
        y1 = target_size[1] // 2
        y2 = target_size[1] - y1
        x_mid = image_test.shape[0] // 2
        y_mid = image_test.shape[0] // 2

        # selecting part of the image only 
        image_arr = image_test[x_mid - x1:x_mid + x2,y_mid - y1:y_mid + y2, :]
        image_arr = image_arr / 255.
    else:
        # Resize the image and normalize values
        image_arr = resize(image_test, (target_size[0], target_size[1]),
                           anti_aliasing=True)
        
    # If just 3 bands get RGB
    if target_size[2] == 3:
        # RGB - 2, 3, 4
        # CI - 3, 4, 8
        # SWIR- 4, 8, 12
        if BANDS == 'RGB':
            image_arr = image_arr[:, :, [1,2,3]]
        elif BANDS == 'SWIR':
            image_arr = image_arr[:, :, [3,7,11]] 
        elif BANDS == 'CI':
            image_arr = image_arr[:, :, [3,4,7]] 
    # One band:
    elif target_size[2] == 1:
        image_arr = image_arr[:, :, BAND]
        image_arr = np.expand_dims(image_arr, axis=2)
        image_arr = np.concatenate((image_arr, image_arr, image_arr), axis=2)
    else:
        image_arr = image_arr[:, :, :target_size[2]]

    image_test = np.expand_dims(image_arr, axis=0)        

    return image_test

def convert_code(code):
    if code.isdigit():
        return int(code)
    else:
        return code

def create_df(images_dir, MUNICIPALITY, target_size=(224, 224, 3), return_paths=None):
    sub_dirs = os.listdir(images_dir)
    sub_dirs = list(map(convert_code, sub_dirs))
    
    if MUNICIPALITY in sub_dirs:
        MUNICIPALITY = MUNICIPALITY
    else:
        if type(MUNICIPALITY) == int or (type(MUNICIPALITY) == np.int64):
            MUNICIPALITY = cities[MUNICIPALITY]
        elif type(MUNICIPALITY) == str and MUNICIPALITY.isdigit():
            MUNICIPALITY = cities[int(MUNICIPALITY)]
        else:
            MUNICIPALITY = int(codes[MUNICIPALITY])
    
    images_dir = os.path.join(images_dir, str(MUNICIPALITY))
    
    out_df = {
        'epiweek':[],
        'image':[]
    }
    
    for image_path in os.listdir(images_dir):
        if image_path.endswith('.tiff'):
            epiweek = epiweek_from_date(image_path)
            full_path = os.path.join(images_dir, image_path)
            
            out_df['epiweek'].append(epiweek)
            out_df['image'].append(full_path)

    df = pd.DataFrame(out_df)
    
    df = df.set_index('epiweek')
    df.index.name = None
    
    if return_paths:
        return df
    
    df.image = df.image.apply(read_image, target_size=target_size)
    df = df.dropna()
    
    return df


def get_dengue_dataset(labels_path, embeddings_path, municipality, temp_prec=False, cases=None, limit=True, static=None, target_size=(224, 224, 3)):
    
    labels_df = read_labels(path=labels_path, Municipality=municipality)

    if limit:
        labels_df = labels_df[(labels_df.index > 201545) & (labels_df.index < 201901)]
        
    
    """ Test All Possible Combinations: """
    if not embeddings_path and not cases and not static and not temp_prec:
        # All variables are False
        return labels_df
    
    elif not embeddings_path and not cases and not static and temp_prec:
        # Only temp_prec is True
        features_df = get_temperature_and_precipitation(municipality, temp_prec)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
    elif not embeddings_path and not cases and static and not temp_prec:
        # Only static is True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])    
        return labels_df
    
    elif not embeddings_path and not cases and static and temp_prec:
        # Both static and temp_prec are True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        # Temperature and Precipitation
        features_df = get_temperature_and_precipitation(municipality, temp_prec)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
    elif not embeddings_path and cases and not static and not temp_prec:
        # Only cases is True
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return labels_df
    
    elif not embeddings_path and cases and not static and temp_prec:
        # Cases and temp_prec are True
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)    
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Temperature and Precipitation
        features_df = get_temperature_and_precipitation(municipality, temp_prec)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
        
    elif not embeddings_path and cases and static and not temp_prec:
        # Cases and static are True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        # Cases
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True) 
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return labels_df
        
    elif not embeddings_path and cases and static and temp_prec:
        # Cases, static, and temp_prec are True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        # Cases
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True) 
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Temperature and Precipitation
        features_df = get_temperature_and_precipitation(municipality, temp_prec)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
        
    elif embeddings_path and not cases and not static and not temp_prec:
        # Only embeddings_path is True
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)

        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
    elif embeddings_path and not cases and not static and temp_prec:
        # Embeddings_path and temp_prec are True
        # Temperature and Precipitation
        features_extra_df = get_temperature_and_precipitation(municipality, temp_prec)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
        
    elif embeddings_path and not cases and static and not temp_prec:
        # Embeddings_path and static are True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])  
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
    elif embeddings_path and not cases and static and temp_prec:
        # Embeddings_path, static, and temp_prec are True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0]) 
        # Temperature and Precipitation
        features_extra_df = get_temperature_and_precipitation(municipality, temp_prec)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
        
    elif embeddings_path and cases and not static and not temp_prec:
        # Embeddings_path and cases are True
        # Cases
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True) 
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)        
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
    elif embeddings_path and cases and not static and temp_prec:
        # Embeddings_path, cases, and temp_prec are True
        # Cases
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True) 
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Temperature and Precipitation
        features_extra_df = get_temperature_and_precipitation(municipality, temp_prec)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)        
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
        
    elif embeddings_path and cases and static and not temp_prec:
        # Embeddings_path, cases, and static are True
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        # Cases
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
        dengue_df = features_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        return dengue_df
    
    else:
        # All variables are True
        # Static
        static = read_static(path=static, Municipality=municipality)
        labels_df = static.merge(labels_df, how='right', left_index=True, right_index=True)
        labels_df = labels_df.fillna(labels_df.mode().iloc[0])
        # Cases
        features_extra_df = read_labels(path=cases, Municipality=municipality)
        features_extra_df.rename(columns={'Labels':'cases'}, inplace=True)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)
        # Temperature and Precipitation
        features_extra_df = get_temperature_and_precipitation(municipality, temp_prec)
        labels_df = features_extra_df.merge(labels_df, how='inner', left_index=True, right_index=True)  
        # Images:
        if embeddings_path.endswith('.csv'):
            # Embeddings
            features_df = read_features(path=embeddings_path, Municipality=municipality)
        else:
            # Pixels
            features_df = create_df(images_dir=embeddings_path, MUNICIPALITY=municipality, target_size=target_size)
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
def normalize_train_features(df, feature_range=(-1, 1), scaler=True, describe=None, min_max=None):
    
    scalers = {}
    # For each column in the dataframe
    for i, column in enumerate(df.columns):
        if not scaler:
            if (i == len(df.columns) - 1):
                continue
        
        # Get values of the column
        values = df[column].values.reshape(-1,1)
        # Generate a new scaler
        if min_max:
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            scaler = StandardScaler()
        try:
            # Fit the scaler just for that column
            scaled_column = scaler.fit_transform(values)
        except:
            continue
            
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
        try:
            # Take the scaler of that column
            scaler = scalers['scaler_' + column]
        except:
            continue
        # Get values of the column
        values = df[column].values.reshape(-1,1)
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
def convert_df_to_np(train):
    for i, column in enumerate(train.columns):
        if i == 0:
            train_arr = np.array(train[column].to_list())
            train_arr = np.expand_dims(train_arr, axis=1)

        else:
            #print(f'original: {train_arr.shape}')

            train_arr_aux = np.array(train[column].to_list())
            train_arr_aux = np.expand_dims(train_arr_aux, axis=1)

            #print(f'aux: {train_arr_aux.shape}')

            train_arr = np.concatenate((train_arr, train_arr_aux), axis=1)
            
    train_arr = np.squeeze(train_arr)

    return train_arr

def reshape_image_array(arr):
    train_arr = pd.DataFrame(arr)
    train_arr = convert_df_to_np(train_arr)
    
    return train_arr


def features_labels_set(timeseries_data, original_df, autoregressive, embeddings=True):
    
    """ Features """
    # We define the number of features as (Cases and media cloud)
    n_features = original_df.shape[1]

    # The features to train the model will be all except the values of the actual week 
    # We can't use other variables in week t because whe need to resample a a 3D Array
    if autoregressive:
        features_set = DataFrame(timeseries_data.values[:,:-n_features])
    else:
        features_set = DataFrame(timeseries_data.values[:,:-1])
    if embeddings:
        # Convert pandas data frame to np.array to reshape as 3D Array
        features_set = features_set.to_numpy()
    else:
        # Convert pandas data frame to np.array to reshape as 3D Array
        features_set = convert_df_to_np(features_set)

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


""" Multimodal Split with Images"""
def multimodal_split(train_X, train_y, test_X, test_y, embeddings, static, temp_prec, autoregressive, T):
    if embeddings and (static or temp_prec or autoregressive):
        if (T != 0):
            # Modality 2
            x_ar = train_X[:, :, 1:]
            x_ar_test = test_X[:, :, 1:]
            # Modality 1
            train_X = train_X[:, :, 0]
            test_X = test_X[:, :, 0]
        else:
            # Modality 2
            x_ar = train_X[:, 1:]
            x_ar_test = test_X[:, 1:]
            # Modality 1
            train_X = train_X[:, 0]
            test_X = test_X[:, 0]
    elif not(embeddings) and static and temp_prec and autoregressive:
        if (T != 0):
            # Modality 2
            x_ar = train_X[:, :, -3:]
            x_ar_test = test_X[:, :, -3:]
            # Modality 1
            train_X = train_X[:, :, :-3]
            test_X = test_X[:, :, :-3]
        else:
            # Modality 2
            x_ar = train_X[:, -3:]
            x_ar_test = test_X[:, -3:]
            # Modality 1
            train_X = train_X[:, :-3]
            test_X = test_X[:, :-3]

    elif not(embeddings) and static and temp_prec and not(autoregressive):
        if (T != 0):
            # Modality 2
            x_ar = train_X[:, :, -2:]
            x_ar_test = test_X[:, :, -2:]
            # Modality 1
            train_X = train_X[:, :, :-2]
            test_X = test_X[:, :, :-2]
        else:
            # Modality 2
            x_ar = train_X[:, -2:]
            x_ar_test = test_X[:, -2:]  
            # Modality 1
            train_X = train_X[:, :-2]
            test_X = test_X[:, :-2]
 
    else:
        x_ar = None
        x_ar_test = None
        
    if embeddings:
        train_X = reshape_image_array(train_X)
        test_X = reshape_image_array(test_X)

        
    None_type = type(None)
    if type(x_ar) != None_type:
        x_ar = np.asarray(x_ar).astype(np.float32)
        x_ar_test = np.asarray(x_ar_test).astype(np.float32)
        

    train_X = np.asarray(train_X).astype(np.float32)
    train_y = np.asarray(train_y).astype(np.float32)

    test_X = np.asarray(test_X).astype(np.float32)
    test_y = np.asarray(test_y).astype(np.float32)
    
    if type(x_ar) != None_type:
        if embeddings:
            print('*'*50)
            print(f'Modality 1 are images of shape:')
            print(f'Train shape: {train_X.shape}')
            print(f'Test shape: {test_X.shape}')
            print('*'*50)
            print(f'Modality 2 is tabular data of shape:')
            print(f'Train shape: {x_ar.shape}')
            print(f'Test shape: {x_ar_test.shape}')
            print('*'*50)
        else:
            print('*'*50)
            print(f'Modality 1 is socodemographic data of shape:')
            print(f'Train shape: {train_X.shape}')
            print(f'Test shape: {test_X.shape}')
            print('*'*50)
            print(f'Modality 2 is temperature and precipitation data of shape:')
            print(f'Train shape: {x_ar.shape}')
            print(f'Test shape: {x_ar_test.shape}')
            print('*'*50)
    else:
        print('*'*50)
        print(f'The output data has shape:')
        print(f'Train shape: {train_X.shape}')
        print(f'Test shape: {test_X.shape}')
        print('*'*50) 
    
    print(f'The labels has shape:')
    print(f'Train shape: {train_y.shape}')
    print(f'Test shape: {test_y.shape}')
    
    return train_X, train_y, test_X, test_y, x_ar, x_ar_test


""" Preprocess the entire dataset for 1 single municipality """
def preprocess_dataset_to_time_series(df, train_percentage = 80, feature_range=(-1, 1), T=3, autoregressive=False, normalize=True, reshape=True, min_max=None):
    """ Train-Test Split"""
    train_df, test_df = train_test_split(df, train_percentage = train_percentage)
    """ Normalization """
    if normalize:
        # Train:
        train_df, scalers = normalize_train_features(train_df, feature_range=feature_range, min_max=min_max)
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