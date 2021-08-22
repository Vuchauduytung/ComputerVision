# System libraries
import os
import sys
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow import keras

# User libraries
from modules.FileIO import *

def main():
    path = os.path.join(os.path.dirname(__file__))
    # driver_name = 'model_40X.json'
    driver_name = sys.argv[1]
    driver_path = os.path.join(path, 'drivers/models', driver_name)
    model_dict = json2dict(driver_path)
    model_filename, model = create_model(model_dict)
    target_path = os.path.join(path, 'original-models', model_filename)
    model.save(target_path, overwrite=True)
    print("\nModel has been created, \ncheck path {path}\n" \
        .format(path = target_path))
    
def create_model(model_dict):
    input_shape = tuple(model_dict.get('input_shape', None))
    CNN = model_dict.get('CNN', None)
    ANN = model_dict.get('ANN', None)
    model_filename = model_dict.get('name', None) + model_dict.get('format', None)
    # Create model
    model = models.Sequential()    
    model.add(keras.Input(shape=input_shape))    
    # Create CNN layers
    for layer_content in CNN.values():
        Conv2D_dict = layer_content.get('Conv2D', None)
        MaxPooling2D_dict = layer_content.get('MaxPooling2D', None)
        BatchNormalization_dict = layer_content.get('BatchNormalization', None)
        if Conv2D_dict is not None:
            model.add(Conv2D(
                filters = Conv2D_dict.get("filters", None),
                kernel_size = tuple(Conv2D_dict.get("kernel_size", None)),
                padding = Conv2D_dict.get("padding", None),
                activation = Conv2D_dict.get("activation", None)
            ))
        if MaxPooling2D_dict is not None:
            model.add(MaxPooling2D(
                pool_size = tuple(MaxPooling2D_dict.get("pool_size", None)),
                padding = Conv2D_dict.get("padding", None),
            ))
        if BatchNormalization_dict is not None:
            model.add(BatchNormalization())            
    #Flatten layer
    model.add(Flatten())    
    # Create ANN layers
    for layer_content in ANN.values():
        Dense_dict = layer_content.get('Dense', None)
        BatchNormalization_dict = layer_content.get('BatchNormalization', None)
        if Dense_dict is not None:
            model.add(Dense(
                units = Dense_dict.get("units", None),
                activation = Dense_dict.get("activation", None)
            ))
        if BatchNormalization_dict is not None:
            model.add(BatchNormalization())   
    model.summary() # Show model summary
    return model_filename, model        
    
    
if __name__ == "__main__":
    main()
    