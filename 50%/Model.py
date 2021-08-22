# System libraries
import os
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import openpyxl
import cv2
import numpy as np
from numba import jit
from numba import cuda
from tempfile import TemporaryFile
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

# User libraries
from modules.FileIO import *


path = os.path.join(os.path.dirname(__file__))

class myModel:
    def __init__(self, model_path, train, validate, test):
        self.model = load_model(model_path)
        self.model.summary()
        self.x_train = train[0]
        self.y_train = tf.keras.utils.to_categorical(train[1], 4)
        self.x_validate = validate[0]
        self.y_validate = tf.keras.utils.to_categorical(validate[1], 4)
        self.x_test = test[0]
        self.y_test = tf.keras.utils.to_categorical(test[1], 4)
        self.datagen = {}

    def data_generate(self, driver_path, group = 'train'):
        data_config_path = os.path.join(driver_path, group + '.json')
        driver = json2dict(data_config_path)
        driver_name = driver.get('name', None)
        properties = driver.get("properties", None)
        
        rotation_range = properties.get("rotation_range", 0)
        width_shift_range = properties.get("width_shift_range", 0)
        height_shift_range = properties.get("height_shift_range", 0)
        brightness_range = properties.get("brightness_range", None)
        shear_range = properties.get("shear_range", 0)
        zoom_range = properties.get("zoom_range", 0)
        channel_shift_range = properties.get("channel_shift_range", 0)
        fill_mode = properties.get("fill_mode", 'nearest')
        cval = properties.get("cval", 0)
        horizontal_flip = properties.get("horizontal_flip", False)
        vertical_flip = properties.get("vertical_flip", False)
        rescale = properties.get("rescale", None)
        
        batch_size = properties.get("batch_size", 32)
        shuffle = properties.get("shuffle", True)
        seed = properties.get("seed", None)
        save_to_dir = properties.get("save_to_dir", None)
        save_prefix = properties.get("save_prefix", '')
        save_format = properties.get("save_format", 'png')
        subset = properties.get("subset", None)
        
        if rescale is not None:
            rescale = rescale/255
        preprocessing_function = properties.get("preprocessing_function", None)
        data_format = properties.get("data_format", None)
        validation_split = properties.get("validation_split", 0)
        dtype = properties.get("dtype", None)
        
        datagen = ImageDataGenerator(
            rotation_range = rotation_range,
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            brightness_range = brightness_range,
            shear_range = shear_range,
            zoom_range = zoom_range,
            channel_shift_range = channel_shift_range,
            fill_mode = fill_mode,
            cval = cval,
            horizontal_flip = horizontal_flip,
            vertical_flip = vertical_flip,
            rescale = rescale,
            preprocessing_function = preprocessing_function,
            data_format = data_format,
            validation_split = validation_split,
            dtype = dtype
        )
        
        if group == 'train': 
            x_data = self.x_train
            y_data = self.y_train
        elif group == 'validate':
            x_data = self.x_validate
            y_data = self.y_validate
        elif group == 'test':
            x_data = self.x_test
            y_data = self.y_test   
        else: 
            raise Exception("Unknown parameter '{group}'".format(group = group))
        
        datagen_Iterator = datagen.flow(
            x = x_data,
            y = y_data,
            batch_size = batch_size,
            shuffle = shuffle,
            seed = seed,
            save_to_dir = save_to_dir,
            save_prefix = save_prefix,
            save_format = save_format,
            subset = subset
        )
        if group == 'train': 
            datagen.fit(self.x_train)
            self.train_datagen = datagen_Iterator
        elif group == 'validate':
            datagen.fit(self.x_validate)
            self.validate_datagen = datagen_Iterator
        elif group == 'test':
            datagen.fit(self.x_test) 
            self.test_datagen = datagen_Iterator      
        else: 
            raise Exception("Unknown parameter '{group}'".format(group = group))
        self.datagen[group] = datagen        
    
    def compile_model(self):
        self.model.compile(optimizer='adam',                                 # Cài đặt thuật toán tính toán sai số
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        
    @jit(forceobj  = True)
    def fit_model(self, epochs):
        self.history = self.model.fit(
            x = self.train_datagen,
            validation_data = self.validate_datagen,
            epochs = epochs,
            initial_epoch = 0,
            # workers = 20,
            # use_multiprocessing = True
        )  
    
    def plot_history(self, direct_path):
        history = self.history
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Accuracy and loss')
        
        ax1.plot(history.history['accuracy'], label='accuracy')
        ax1.plot(history.history['val_accuracy'], label = 'val_accuracy')
        ax1.set(xlabel='epochs', ylabel='Accuracy')
        ax1.legend(loc='lower right')
        
        ax2.plot(history.history['loss'], label='loss')
        ax2.plot(history.history['val_loss'], label = 'val_loss')
        ax2.set(xlabel='epochs', ylabel='Loss')
        ax2.legend(loc='highest right')
        
        fig.savefig(direct_path)
        
    def save_history(self, history_name):
        direct_path = os.path.join(path, 'history', history_name)
        with open(direct_path, 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)
        
    def save_model(self, target_path, model_name):
        direct_path = os.path.join(target_path, model_name)
        self.model.save(direct_path, overwrite=True)
                          