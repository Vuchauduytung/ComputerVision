-----------------------------------------------------BreastCancerDetection------------------------------------------------------

-- Purpose --
Train breast image processing model to predict whether patients have breast cancer or not base on their mammography X-ray image

-- User manual --
Go to this file directory
    To train a new model, type: <python/python3> Model.py <weight-file-name>
        Examples: python Model.py breast_cancer_wieght_25x25_v1.h5
    To train a pre-train model, type: <python/python3> Model.py <new-weight-file-name> <old-weight-file-name> <new-weight-file-name>
        Examples: python Model.py breast_cancer_wieght_25x25_v2.h5 breast_cancer_wieght_25x25_v1.h5
    To test a wieght file, type: <python/python3> Model_Predict.py <weight-file-name> <test-data-folder> <dimension-1> <dimension-2>
        Examples: python Model_Predict.py breast_cancer_wieght_25x25_v1.h5 Benign 25 25

    positive: Malignant
    negative: Benign