# from __future__ import absolute_import, division, print_function, unicode_literals
from timeit import default_timer as timer 
import os
import sys
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# User library
from Modules.Library.LoadData import load_ds_data
from Modules.Library.EarlyStopping import CustomCallback as CC
from Modules.Library.CreateModel import create_new_model, result_plot

def main():
    # Khởi tạo thông số
    epochs = 200                 # Số lần train hết tập dữ liệu
    input_shape = (25,25,1)     # Kích thước ảnh + kênh màu
    image_size = (25,25)        # Kích thước ảnh
    batch_size = 32             

    path = os.path.dirname(os.path.abspath(__file__))
    weight_file_directory = os.path.join(path, 'Weight', sys.argv[1])
    # weight_file_directory = os.path.join(path, 'Weight', 'breast_cancer_wieght_25x25_v3.h5')
    train_ds, val_ds = load_ds_data(image_size = image_size, 
                                    batch_size = batch_size)

    # model = create_new_model(input_shape)
    
    # Model initialization
    if len(sys.argv) == 2:
        model = create_new_model(input_shape)
    else:
        pretrain_file_directory = os.path.join(path, 'Weight', sys.argv[2])
        print('Pre-train model: {}'.format(pretrain_file_directory))
        model = load_model(pretrain_file_directory)

    # Model backpropagation
    model.compile(optimizer='adam',                                 # Cài đặt thuật toán tính toán sai số
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    # Train data
    start = timer()
    history = model.fit(train_ds, 
                        epochs=epochs, 
                        validation_data=val_ds,
                        callbacks = CC())
    print("\nWith GPU kernel: %ds" % (timer()-start))
    model.save(weight_file_directory, overwrite=True)
    print('Model train completed')
    print('Weight file path: %s' % weight_file_directory)

    try:
        result_plot(history)
    except Exception as err:
        print(err)
    finally:
        history_path = os.path.join(path, 'train-history', sys.argv[1].replace('.h5', '.json'))
        with open(history_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        val_loss, val_acc = model.evaluate(val_ds, verbose=2)
        print('\n--------------------------------------------------------------')
        print('\tTest accuracy: %f \n\tTest loss: %f' % (val_loss, val_acc))
        print('--------------------------------------------------------------\n')

if __name__ == "__main__":
    main()