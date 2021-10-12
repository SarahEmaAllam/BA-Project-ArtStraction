# # example of loading the vgg16 model
# from keras.applications.vgg16 import VGG16
# # load model
# model = VGG16()
# # summarize the model
# model.summary()


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



# load images per class
Gracesfiles = glob.glob(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Image Extractor\Databases\train\Graces\*')
Recliningfiles = glob.glob(
    r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Image Extractor\Databases\train\Reclining\*')
Pietafiles = glob.glob(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Image Extractor\Databases\train\Pieta\*')
Sebastianfiles = glob.glob(
    r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Image Extractor\Databases\train\Sebastian\*')
Antinousfiles = glob.glob(
    r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Image Extractor\Databases\train\Antinous\*')
Tanagrafiles = glob.glob(
    r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Image Extractor\Databases\train\Tanagra\*')

# for fn in Tanagrafiles:
#     im1 = Image.open(fn)
#     im1.save(fn + '.png')
#
#  return image from files path if string is in image name
graces_files = [fn for fn in Gracesfiles]
reclining_files = [fn for fn in Recliningfiles]
pieta_files = [fn for fn in Pietafiles]
sebastian_files = [fn for fn in Sebastianfiles]
antinous_files = [fn for fn in Antinousfiles]
tanagra_files = [fn for fn in Tanagrafiles]

sets = [graces_files, reclining_files, pieta_files, sebastian_files, antinous_files, tanagra_files]

print("Number of classes:", len(sets))

print(len(graces_files), len(reclining_files), len(pieta_files),
      len(sebastian_files), len(antinous_files), len(tanagra_files))

# training set
datasets_train =[]
datasets_val = []
datasets_test = []
for dataset in sets:
    current_set = dataset
    # training set
    size_train = len(current_set) * 70 / 100
    dataset_train = np.random.choice(current_set, size=int(size_train), replace=False)
    datasets_train.append(dataset_train)
    current_set = list(set(dataset) - set(dataset_train))
    print(len(dataset), len(dataset_train))

#     validation set
    size_val = len(dataset) * 15 / 100
    dataset_val = np.random.choice(current_set, size=int(size_val), replace=False)
    datasets_val.append(dataset_val)
    current_set = list(set(current_set) - set(dataset_val))
# # test set
    dataset_test = np.random.choice(current_set, size=int(size_val), replace=False)
    datasets_test.append(dataset_test)

# declare directories
train_dir = 'training_data'
val_dir = 'validation_data'
test_dir = 'test_data'

for index, train in enumerate(datasets_train):
    print(' training dataset ', index,  train.shape)

for index, val in enumerate(datasets_val):
        print(' validation dataset ', index, val.shape)

for index, test in enumerate(datasets_test):
        print(' testing dataset ', index, test.shape)



# concatenate all classes
train_files = datasets_train
validate_files = datasets_val
test_files = datasets_test

#  create declared folders
os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None

# copy images into folders
for fi in train_files:
    for fn in fi:
        shutil.copy(fn, train_dir)

for fi in validate_files:
    for fn in fi:
        shutil.copy(fn, val_dir)

for fi in test_files:
    for fn in fi:
        shutil.copy(fn, test_dir)

        # Preparing Datasets

import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# %matplotlib inline

IMG_DIM = (150, 150)

train_files = glob.glob('training_data/*')
# imgs = load_images(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Model 1\training_data\\')
# train_imgs = normalize_data(imgs, 150, 150, negative=False)
# train_imgs = [img_to_array(img) for img in trains_imgs]
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
# train_imgs = np.array(train_imgs)
# train_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in train_files]
train_labels = [fn.split('\\')[1].split('-')[0] for fn in train_files]

print(train_labels)
# imgs_val = load_images(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Model 1\validation_data\\')
validation_files = glob.glob('validation_data/*')
# validation_imgs = normalize_data(imgs_val, 150, 150, negative=False)
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
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
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])

batch_size = 5
num_classes = 6
epochs = 20
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)
#
print(train_labels)
print(train_labels_enc)


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
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=5)
input_shape = (150, 150, 3)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(6))

model.compile(loss='sparse_categorical_crossentropy',
              # optimizer=optimizers.Adam(learning_rate=0.0001),
              optimizer=optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
model.summary()
model.save('model4SGDExtradata')
# graphs 1
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# graphs

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,31))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

