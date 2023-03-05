import os
import shutil

import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize
from skimage.util import crop
from sklearn.model_selection import train_test_split

import numpy as np

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')



""" Read the list with the path og the images"""
def show_image(image, bands=[1,2,3], size=6):
    print(f'The shape of the image is: {image.shape}')
    plt.figure(figsize=(size, size))
    plt.imshow(image[:, :, bands])
    plt.show()
    
def plot_samples(path):
    for city in os.listdir(path):
        images_path = os.path.join(path, city)
        print(f'---------- {city} ----------')
        print(f'The city has {len(os.listdir(images_path))} images')
        # Get the fist image
        image_path = os.path.join(images_path, os.listdir(images_path)[0])
        # read the image
        image_tiff = io.imread(image_path)
        show_image(image_tiff)
        
def get_dataset_list(path, head=0, show_dirs=False):
    image_list = []
    image_dir = {}
    
    if show_dirs:
        print('Images in directories: ')
    cities = os.listdir(path)
    for city in cities:
        if show_dirs:
            print(city)
        # Get path to city:
        images_path = os.path.join(path, city)
        image_dir[city] = os.listdir(images_path)
        for image in os.listdir(images_path):
            # make sure file is an image
            if image.endswith(('.jpg', '.png', 'jpeg', 'tiff')):
                # Get path to image:
                image_path = os.path.join(images_path, image)
                # Skip directory
                if os.path.isdir(image_path):
                    #print(f'Directory: {image_path}')
                    continue
                #img_path = path + file
                image_list.append(image_path)
    if head:
        print(f'Image list top 5 examples of length {len(image_list)}:')
        #image_dir
        print(image_list[:head])
        
    return image_list

#image_list = get_dataset_list(path)

""" Train-Test Split"""
def split_list(image_list, val=False, test_size=0.2, val_size=0.2):
    
    # Train-Test split
    image_list = np.array(image_list)
    x_train, x_test = train_test_split(image_list, test_size=test_size)
    if val:
        x_train, x_val = train_test_split(x_train, test_size=val_size)

    # To numpy
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    print(f'Train shape is: {x_train.shape}')
    print(f'Test shape is: {x_test.shape}')
    if val:
        x_val = np.array(x_val)
        print(f'Validation shape is: {x_val.shape}')
        return x_train, x_test, x_val
    return x_train, x_test

#x_train, x_test = split_list(image_list)


""" Create Custom Dataloader: """
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, image_paths,
                 batch_size,
                 input_size = (224, 224, 12),
                 shuffle = True, 
                 crop = True,
                 augmentation = None):
        
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.crop = crop
        # Length of dataset
        self.n = len(self.image_paths)
        
        if self.crop:
            self.x1 = self.input_size[0] // 2
            self.x2 = self.input_size[0] - self.x1
            self.y1 = self.input_size[1] // 2
            self.y2 = self.input_size[1] - self.y1
            
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
        # for each image in batch
        for image_name in batches:
            # Get the image
            image = self.__get_image(image_name, self.input_size) 
            X_batch.append(image)
        
        X_batch = np.array(X_batch)
        
        # Augmentation
        if self.augmentation:
            # prepare iterator
            X_batch = self.augmentation.flow(X_batch, batch_size=self.batch_size, shuffle=True).next()
            #print('Augmentation done!')

        #print(f'The shape of the batch is : {X_batch.shape} of type: {type(X_batch)}')
        return X_batch, X_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)
            
    def __len__(self):
        return self.n // self.batch_size
    

def show_random_images(list_data, augmentation=None, target_size=(224,224,3)):
    example_gen = CustomDataGen(list_data,
                                 batch_size=9, 
                                 input_size=target_size, 
                                 augmentation=augmentation, 
                                 shuffle = True)
    
    for x,y in example_gen:
        x, _ = x, y
        break

    ax = plt. subplots(3, 3, figsize=(10,10))
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)

        # plot raw pixel data
        plt.imshow(x[i, :, :, :])
        #plt.title(f'Image {[i]}')

    # show the figure
    plt.show()
    
    
""" Predictions"""
def plot_autoencoder_predictions(autoencoder, testgen, samples=3):
    fig, ax = plt. subplots(samples, 2, figsize=(10,10))

    for i in range(samples):
        # generator
        # generate batch of images
        example_batch = np.array(next(zip(testgen)))
        print(example_batch.shape)
        # Take RGB bands from first image in batch
        example_image = example_batch[0, 0, i, :, :, 0:3]
        # plot raw pixel data
        ax[i, 0].imshow(example_image)

        # Predicted Images
        example_image_test = example_batch[0, 0, i, :, :, :]
        example_image_test = np.expand_dims(example_image_test, axis=0)
        output = autoencoder.predict(example_image_test)
        op_image = output[0, :, :, 0:3]
        ax[i, 1].imshow(op_image)

    plt.show()