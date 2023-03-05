from skimage import io
from skimage.transform import resize
from skimage.util import crop

import os

import numpy as np

import pandas as pd


""" Read a single Image """
def read_image(path, crop=True, target_size=(224,224,3), BANDS='RGB', BAND=0):
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
        print(f'rgb shape: {image_arr.shape}')
    # One band:
    elif target_size[2] == 1:
        image_arr = image_arr[:, :, BAND]
        image_arr = np.expand_dims(image_arr, axis=2)
        image_arr = np.concatenate((image_arr, image_arr, image_arr), axis=2)
    else:
        image_arr = image_arr[:, :, :target_size[2]]

    image_test = np.expand_dims(image_arr, axis=0)        

    return image_test

""" Generate the Embedding for an Image """
def generate_embedding(image, model):
    embaedding = model.predict(image)
    embaedding = np.squeeze(embaedding, axis=0)
    return embaedding

""" Image Date from Path """
def get_image_name(path):
    image_name = path[path.index('/image')+7:path.index('.tiff')]
    return image_name

""" Image Name from Path """
def get_municipality_name(path):
    image_name = path[path.index('_cities/')+8:path.index('/image')]
    return image_name

def generate_embeddings_df(image_list, model, crop=True, target_size=(224,224,3), BANDS='RGB', BAND=0):

    embeddings = pd.DataFrame(columns=['Municipality Code', 'Date', 'Embedding'])

    for path in image_list:
        image = read_image(path=path, crop=crop, target_size=target_size, BANDS=BANDS, BAND=BAND)
        embedding = generate_embedding(image, model)
        date = get_image_name(path=path)
        name = get_municipality_name(path)
        embeddings = embeddings.append({'Date': date, 'Embedding': embedding, 'Municipality Code': name}, ignore_index=True )

    return embeddings

def split_columns(df, column='Embedding'):
    df_aux = pd.DataFrame(df[column].tolist())
    df_aux = pd.concat([df[['Municipality Code', 'Date']], df_aux], axis=1)
    return df_aux

def save_embeddings_as_csv(df, path):
    # new df from the column of lists
    embeddings_df = split_columns(df)

    list_paths = path.split('/')
    
    final_path = ''
    for item in list_paths:
        final_path = os.path.join(final_path, item)
        if not os.path.exists(final_path) and (not final_path.endswith('.csv')):
            os.mkdir(final_path)
        else:
            continue
    
    embeddings_df.to_csv(path,index=False)
    # display the resulting df
    embeddings_df