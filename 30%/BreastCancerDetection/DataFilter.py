import os
import shutil

path = os.path.dirname(os.path.realpath(__file__))
original  = 'E:\data\Computer Vision\Dataset of Mammography with Benign Malignant Breast Masses\INbreast Dataset\Malignant Masses'
target = os.path.join(path, 'Data/Original/Test/Malignant')

files = os.listdir(original)

for f in files:
    if '(' not in f and ')' not in f:
        shutil.copyfile(os.path.join(original, f), os.path.join(target, f))