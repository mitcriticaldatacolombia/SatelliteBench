import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize
from skimage.util import crop

import tensorflow as tf

import pandas as pd
import numpy as np

from epiweeks import Week
from datetime import date as convert_to_date


    
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, image_paths,
                 batch_size,
                 input_size = (224, 224, 12),
                 shuffle = False, 
                 df = None,
                 augmentation=None,
                 crop = True):
        
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.df = df
        
        self.augmentation = augmentation
        self.crop = crop
        # Length of dataset
        self.n = len(self.image_paths)
        
        if self.crop:
            self.x1 = self.input_size[0] // 2
            self.x2 = self.input_size[0] - self.x1
            self.y1 = self.input_size[1] // 2
            self.y2 = self.input_size[1] - self.y1

        
    def __get_epiweek(self, image_name):
        date = image_name.split('-')
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
    
    def __get_label(self, path):
        # Get city:
        city = path.split('/')[-2]
        # Get epiweek:
        date = path.split('/')[-1]
        epiweek = self.__get_epiweek(date)
        # Get cases:
        cases = int(self.df[self.df['Municipality'] == city].loc[:,epiweek])
        return cases
        
        
    # Helper function to read the image
    def __get_image(self, path, target_size):
        # Read the image and convert to numpy array
        image = io.imread(path)
        
        if self.crop:
            x_mid = image.shape[0] // 2
            y_mid = image.shape[0] // 2
            
            # selecting part of the image only 
            image_arr = image[x_mid - self.x1:x_mid + self.x2,y_mid - self.y1:y_mid + self.y2, :]
            image_arr = image_arr / 255.
        else:
            # Resize the image and normalize values
            image_arr = resize(image,(target_size[0], target_size[1]))

        # If just 3 bands get RGB
        if target_size[2] == 3:
            image_arr = image_arr[:, :, [1,2,3]]
        else:
            image_arr = image_arr[:, :, :target_size[2]]
        
        #print(f'The shape of the image before reshape: {image_arr.shape}, of type{type(image_arr)}')
        return image_arr
        
    
    def __getitem__(self, index):     
        batches = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = []
        y_batch = []
        # for each image in batch
        for image_name in batches:
            # Get the image
            image = self.__get_image(image_name, self.input_size)
            X_batch.append(image)
            # Get the label
            label = self.__get_label(image_name)
            y_batch.append(label)
            
        y_batch = np.array(y_batch)
        X_batch = np.array(X_batch)
        
        # Augmentation
        if self.augmentation:
            # prepare iterator
            X_batch = self.augmentation.flow(X_batch, batch_size=self.batch_size, shuffle=True).next()
            #print('Augmentation done!')
        
        #print(f'The shape of the batch is : {X_batch.shape} of type: {type(X_batch)}')
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)
            
    def __len__(self):
        return self.n // self.batch_size
    
    

def show_random_samples(list_data, df, augmentation=None, target_size=(224,224,3)):

    example_gen = CustomDataGen(list_data,
                                batch_size=9, 
                                input_size=target_size, 
                                df=df, 
                                augmentation=augmentation, 
                                shuffle = True)

    for x,y in example_gen:
        x, y = x, y
        break

    ax = plt. subplots(3, 3, figsize=(10,10))
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)

        # plot raw pixel data
        plt.imshow(x[i, :, :, :])
        plt.title(f'outbreak: {y[i]}')

    # show the figure
    plt.show()
    