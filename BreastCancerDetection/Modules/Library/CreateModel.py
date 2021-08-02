from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow import keras

def create_new_model(input_shape):
    model = models.Sequential()                 # Khởi tạo model
    model.add(keras.Input(shape=input_shape))   # Định nghĩa kích dữ liệu đầu vào
    model.add(layers.Conv2D(256,                # Add 1 lớp Tích chập có 512 kernel, kích thước 3x3,
                            (3, 3),             # hàm kích hoạt gelu, kích thước ảnh đầu vào (32,32,3)
                            activation='gelu', 
                            input_shape=input_shape,
                            padding = "same"
                            ))
    # Kích thước ảnh (23,23,3)
    model.add(layers.MaxPooling2D((2, 2), 
                                  padding = "same"))
    # Kích thước ảnh (11,11,3)
    model.add(layers.BatchNormalization())
    # Chuẩn hóa mức xám
    model.add(layers.Conv2D(512, 
                            (3, 3), 
                            activation='gelu', 
                            paddingpadding = "same"))
    # Kích thước ảnh (9,9,3)
    model.add(layers.MaxPooling2D((2, 2), 
                                  padding = "same"))
    # Kích thước ảnh (4,4,3)
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, 
                            (3, 3), 
                            activation='gelu',
                            padding = "same"))
    # Kích thước ảnh (2,2,3)
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), 
                            activation='gelu', 
                            padding = 'same'))
    # Kích thước ảnh (2,2,3) vì có padding zero
    model.add(layers.MaxPooling2D((2, 2),
                                  padding='same'))
    # Kích thước ảnh (2,2,3) vì có padding zero
    model.add(layers.BatchNormalization())

    model.summary()

    model.add(layers.Flatten())
    # (2,2,3) -> (12,)
    
    model.add(layers.Dense(256, activation='gelu'))
    # Hidden layer 1 có 256 perceptron
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(512, activation='gelu'))
    # Hidden layer 1 có 512 perceptron
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256, activation='gelu'))
    # Hidden layer 1 có 256 perceptron
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(128, activation='gelu'))
    # Hidden layer 1 có 128 perceptron
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(2, activation='sigmoid'))
    # Layer cuối cùng của mạng ANN, hàm kích hoạt sigmoid do có 2 lớp ngõ ra

    model.summary()
    return model
    
def result_plot(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')