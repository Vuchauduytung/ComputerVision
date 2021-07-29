import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
train_direct_path = os.path.join(parent_path, 'Data/Original/Train')
validation_direct_path = os.path.join(parent_path, 'Data/Original/Validate')

def load_ds_data(image_size, batch_size):
      train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(-1,1),
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            rescale=1./255,
            )

      val_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=(-1,1),
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            rescale=1./255,
            )

      train_generator = train_datagen.flow_from_directory(
            directory=train_direct_path,
            target_size=image_size,
            batch_size=batch_size,
            save_to_dir=None,
            save_prefix="train",
            save_format="png",
            )        

      val_generator = val_datagen.flow_from_directory(
            directory=validation_direct_path,
            target_size=image_size,
            batch_size=batch_size,
            save_to_dir=None,
            save_prefix="val",
            save_format="png",
            )
      return train_generator, val_generator

