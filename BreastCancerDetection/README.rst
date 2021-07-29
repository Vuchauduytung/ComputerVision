-----------------------------------------------------BreastCancerDetection------------------------------------------------------

-- Purpose --
Train breast image processing model to predict whether patients have breast cancer or not base on their mammography image

-- User manual --
Go to this file directory
    To train a new model, type: <python/python3> Model.py <weight-file-name>
        Examples: python Model.py breast_cancer_wieght.h5
    To train a pre-train model, type: <python/python3> Model.py <new-weight-file-name> <old-weight-file-name> <new-weight-file-name>
        Examples: python Model.py new_breast_cancer_wieght.h5 breast_cancer_wieght.h5
    To test a wieght file, type: <python/python3> Model_Predict.py <weight-file-name> <test-data-folder>
        Examples: python Model_Predict.py breast_cancer_wieght.h5 Benign 