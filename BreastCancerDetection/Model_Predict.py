import os
import sys
import shutil
import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow.keras.models import load_model


class classify(Enum):
    Benign = True
    Malignant = False

def main():
    model_name = sys.argv[1]
    test_folder = sys.argv[2]
    dimension_1 = int(sys.argv[3])
    dimension_2 = int(sys.argv[4])
    
    # model_name = 'breast_cancer_wieght_25x25_v3.h5'
    # test_folder = 'Benign'
    # dimension_1 = 25
    # dimension_2 = 25
    
    path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(path, 'Weight', model_name)
    test_data_path = os.path.join(path, 'Data/Original/Test', test_folder)

    model = load_model(model_path)

    validation_split=None
    label_mode = None
    seed = None
    image_size = (dimension_1, dimension_2)
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
    
    switcher = {
        classify.Benign.value : 'Benign',
        classify.Malignant.value : 'Malignant'
    }
    
    if test_folder == 'Benign':
        class_true = result[:,0]>0.5
    elif test_folder == 'Malignant':
        class_true = result[:,1]>0.5
    
    # Đếm số phần tử có trong class_true
    # unique các giá trị xuất hiện trong class_true
    # counts số lần xuất hiện của từng giá trị trong class_true
    unique, counts = np.unique(class_true, return_counts=True)

    for value in unique:
        try:
            classify_result = np.append(classify_result, switcher.get(value, None))
        except:
            classify_result = np.array([switcher.get(value, None)])    
    
    target_path = os.path.join(path, 'TargetTest')
    positive_target_path = os.path.join(target_path, 'positive')
    negative_target_path = os.path.join(target_path, 'negative')
    
    try:
        shutil.rmtree(positive_target_path)
        shutil.rmtree(negative_target_path)
    finally:
        os.mkdir(positive_target_path)
        os.mkdir(negative_target_path)
        
    for index, img in enumerate(os.listdir(test_data_path)):
        src = os.path.join(test_data_path, img)
        if class_true[index]:
            dst = os.path.join(negative_target_path, img)
        else:
            dst = os.path.join(positive_target_path, img)
        shutil.copyfile(src, dst)
        
    print('\n--------------------------------------------------------------')  
    print('Result:')
    print(dict(zip(classify_result, counts)))
    if test_folder == 'Benign':
        print('Negative accuracy: {:.2f}%'.format(100*counts[1]/(counts[0]+counts[1])))
    elif test_folder == 'Malignant':
        print('Positive accuracy: {:.2f}%'.format(100*counts[0]/(counts[0]+counts[1])))
    print('--------------------------------------------------------------\n')
    
if __name__ == "__main__":
    main()
