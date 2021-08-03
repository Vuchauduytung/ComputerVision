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
    # Kích thước ảnh (25,25,1)
    model.add(layers.MaxPooling2D((2, 2), 
                                  padding = "same"))
    # Kích thước ảnh (13,13,1)
    model.add(layers.BatchNormalization())
    # Chuẩn hóa mức xám
    model.add(layers.Conv2D(512, 
                            (3, 3), 
                            activation='gelu', 
                            padding = "same"))
    # Kích thước ảnh (13,13,1)
    model.add(layers.MaxPooling2D((2, 2), 
                                  padding = "same"))
    # Kích thước ảnh (7,7,1)
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, 
                            (3, 3), 
                            activation='gelu',
                            padding = "same"))
    # Kích thước ảnh (7,7,1)
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), 
                            activation='gelu', 
                            padding = 'same'))
    # Kích thước ảnh (7,7,1) vì có padding zero
    model.add(layers.MaxPooling2D((2, 2),
                                  padding='same'))
    # Kích thước ảnh (4,4,1) vì có padding zero
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(64, (3, 3), 
                            activation='gelu', 
                            padding = 'same'))
    # Kích thước ảnh (4,4,1) vì có padding zero
    model.add(layers.MaxPooling2D((2, 2),
                                  padding='same'))
    # Kích thước ảnh (2,2,1) vì có padding zero
    model.add(layers.BatchNormalization())

    # model.summary()

    model.add(layers.Flatten())
    # (2,2,1) -> (4,)
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Accuracy and loss')
    
    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label = 'val_accuracy')
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.legend(loc='lower right')
    
    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label = 'val_loss')
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.legend(loc='highest right')