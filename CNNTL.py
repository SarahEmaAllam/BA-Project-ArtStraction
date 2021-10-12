import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# %matplotlib inline


import glob
import numpy as np
import os
import cv2
from PIL import Image
import shutil

from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

np.random.seed(42)


def load_images(path):
    """
    Loads images from specified path.
    Args:
        path: Path to load images from.
    Returns:
        images (numpy.array): numpy array of loaded images.
    """
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return np.array(images)


def normalize_data(data, width, height, negative=False):
    """
    Normalizes images in interval [0 1] or [-1 1] and returns batches.
    Args:
        data (numpy.array): Array of images to process.
        width (int): Width dimension of image in pixels.
        height (int): Height dimension of image in pixels.
        negative (bool, optional): Flag that determines interval (True: [-1 1], False: [0 1]). Defaults to True.
    Returns:
        data (tf.data.Dataset): Kvasir-SEG dataset sliced w/ batch_size and normalized.
    """
    normalized_data = []
    for image in data:
        resized_image = cv2.resize(image, (width, height)).astype('float32')
        if negative:
            image = (resized_image / 127.5) - 1
        else:
            image = (resized_image / 255.0)
        normalized_data.append(image)
    return normalized_data



IMG_DIM = (300, 300)

train_files = glob.glob('training_data/*')
imgs = load_images(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Model 1\training_data\\')
train_imgs = normalize_data(imgs, 300, 300, negative=False)
# train_imgs = [img_to_array(img) for img in trains_imgs]
# train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
# train_imgs = np.array(train_imgs)
# train_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in train_files]
train_labels = [fn.split('\\')[1].split('-')[0] for fn in train_files]

print(train_labels)
imgs_val = load_images(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Model 1\validation_data\\')
validation_files = glob.glob('validation_data/*')
validation_imgs = normalize_data(imgs_val, 300, 300, negative=False)
# validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('\\')[1].split('-')[0] for fn in validation_files]
print(validation_labels)

print('Train dataset shape:', train_imgs.shape,
      '\tValidation dataset shape:', validation_imgs.shape)

train_imgs_scaled = train_imgs.astype('float32')
print(train_imgs_scaled)
print(train_imgs_scaled.shape)
# cv2.imshow('', train_imgs_scaled[0])
# cv2.waitKey(0)
validation_imgs_scaled  = validation_imgs.astype('float32')
# train_imgs_scaled /= 255
# validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])

batch_size = 32
num_classes = 6
epochs = 50
input_shape = (300, 300, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)
#
# print(train_labels)
# print(train_labels_enc)


# Augmentation for training images

# 1. Zooming the image randomly by a factor of 0.3 using the zoom_range parameter.
# 2. Rotating the image randomly by 50 degrees using the rotation_range parameter.
# 3. Translating the image randomly horizontally or vertically by a 0.2 factor of the image’s width or height using the width_shift_range
# and the height_shift_range parameters.
# 5. Randomly flipping half of the images horizontally using the horizontal_flip parameter.
# 6. Leveraging the fill_mode parameter to fill in new pixels for images after we apply any of the preceding operations
# (especially rotation or translation). In this case, we just fill in the new pixels with their nearest surrounding pixel values.

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)


#
# #  CNN

# generate new augmented images
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=10)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=5)
input_shape = (300, 300, 3)

from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                  input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

import pandas as pd

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])