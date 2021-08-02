import os
import sys
import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow.keras.models import load_model


class classify(Enum):
    Benign = True
    Malignant = False

def main():
    path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(path, 'Weight', sys.argv[1])
    test_data_path = os.path.join(path, 'Data/Original/Test', sys.argv[2])
    # print(model_path)
    # print(test_data_path)
    # # model_path = os.path.join(path, 'Weight', 'breast_cancer_report_final.h5')
    # # test_data_path = os.path.join(path, 'Data/Original/Test', 'Malignant')

    model = load_model(model_path)

    validation_split=None
    label_mode = None
    seed = None
    image_size = (25,25)
    batch_size = 32
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_path,
        validation_split=validation_split,
        label_mode = label_mode,
        subset=validation_split,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        color_mode = "grayscale"
        )

    result = model.predict(x = test_ds, 
                            batch_size = batch_size, 
                            verbose = 1)
    class_true = result[:,1]<0.5
    
    # Đếm số phần tử có trong class_true
    # unique các giá trị xuất hiện trong class_true
    # counts số lần xuất hiện của từng giá trị trong class_true
    unique, counts = np.unique(class_true, return_counts=True)
    switcher = {
        classify.Benign.value : 'Benign',
        classify.Malignant.value : 'Malignant'
    }
    for value in unique:
        try:
            classify_result = np.append(classify_result, switcher.get(value, None))
        except:
            classify_result = np.array([switcher.get(value, None)])    
    print('\n--------------------------------------------------------------')  
    print('Result:')
    print(dict(zip(classify_result, counts)))
    print('--------------------------------------------------------------\n')
    
if __name__ == "__main__":
    main()
