import glob

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from pasta.augment import inline
# %matplotlib inline

import glob
import numpy as np
import os
import cv2
from PIL import Image
import shutil


# from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

# np.random.seed(42)
class CategoricalTruePositives(tensorflow.keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tensorflow.reshape(tensorflow.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tensorflow.cast(y_true, "int32") == tensorflow.cast(y_pred, "int32")
        values = tensorflow.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tensorflow.cast(sample_weight, "float32")
            values = tensorflow.multiply(values, sample_weight)
        self.true_positives.assign_add(tensorflow.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features


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
        #if negative:
        #    image = (resized_image / 127.5) - 1
        #else:
        #    image = (resized_image / 255.0)
        # normalized_data.append(image)
        normalized_data.append(resized_image)
    return normalized_data



files = glob.glob('DATA/*')
labels = [fn.split('/')[1].split('-')[0] for fn in files]
print(labels)
np_labels = np.array(labels)
np.save('labels.npy', np_labels)

data_path = os.path.join(os.getcwd(), 'DATA')
imgs = load_images(data_path)
normalized_imgs = normalize_data(imgs, 224, 224, negative=False)
np_imgs = np.array(normalized_imgs)
np.save('np_imgs.npy', np_imgs)


labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')

batch_size = 60
num_classes = 6
epochs = 3
input_shape = (224, 224, 3)

idx = np.random.permutation(len(np_imgs))  # get suffeled indices
imgs, labels = np_imgs[idx], labels[idx]  # uniform suffle of data and label

imgs_train, imgs_val, imgs_test = np.split(imgs, [int(len(imgs) * 0.75), int(len(imgs) * 0.9)])  # split of 75:15:10
labels_train, labels_val, labels_test = np.split(labels, [int(len(labels) * 0.75), int(len(labels) * 0.9)])

print(len(imgs_train), len(imgs_val), len(imgs_test))
print(imgs_train[:3])
print(labels_train[:3])

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(labels_train)
train_labels_enc = le.transform(labels_train)
validation_labels_enc = le.transform(labels_val)
test_labels_enc = le.transform(labels_test)

#
# print("TRAINING LEBELS - TRIAN IMG")
# print(list(set(train_labels_enc) - set(train_imgs)))


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

train_datagen = ImageDataGenerator(zoom_range=0.3, rotation_range=50,
                                       width_shift_range=0.5, height_shift_range=0.5,
                                       horizontal_flip=True, fill_mode='nearest')
#
# val_datagen = ImageDataGenerator(rescale=1./255)


#
# #  CNN

# generate new augmented images
train_generator = train_datagen.flow(imgs_train, train_labels_enc, batch_size=10)
# train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=10)
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=5)
# input_shape = (300, 300, 3)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import layers
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

disable_eager_execution()

statistics = []
x_accuracy = []
x_loss = []

for i in range(40, 43, 1):
    np.random.seed(i)
    # Define VGG16 base model
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    # Freeze layers
    base_model.trainable = True
    # print(input_shape)
    # print(train_imgs_scaled.shape)
    # print(validation_imgs_scaled.shape)
    # print(type(train_labels_enc))
    # print(type(validation_labels_enc))
    # Create new model on top of it
    train_ds = preprocess_input(imgs_train)
    validation_ds = preprocess_input(imgs_val)
    test_ds = preprocess_input(imgs_test)
    

    inputs = tensorflow.keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = tensorflow.keras.layers.Dense(6, activation='softmax')(x)
    model = tensorflow.keras.Model(inputs, outputs)

    # inputs = tensorflow.keras.Input(shape=input_shape)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    # x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    # x = layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    # outputs = tensorflow.keras.layers.Dense(6, activation='sigmoid')(x)
    # model = tensorflow.keras.Model(inputs, outputs)
    # flatten = layers.GlobalAveragePooling2D()

    # prediction_layer = layers.Dense(6, activation='sigmoid')
    # model = models.Sequential([
    #     base_model,
    #     flatten,
    #     prediction_layer
    # ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-5),
                  # optimizer=optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=15, restore_best_weights=True)
    # history = model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

    history = model.fit(x=train_ds, y=train_labels_enc,
                        validation_data=(validation_ds, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=25,
                        callbacks=[es],
                        verbose=1)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_ds, test_labels_enc, batch_size=32)
    y_labels = model.metrics_names
    x_accuracy.append(results[1])
    x_loss.append(results[0])
    # statistic = zip(model.metrics_names, results)
    # statistics.append(statistic)
    # print(statistics)
    # c = list.append(statistics)
    # print(list(statistics))
    # print("statistics:" , statistics)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print("Generate predictions")
    # predictions = model.predict(test_imgs_scaled)
    # print("predictions shape:", predictions.shape)

    model.summary()
    # model.save('modelVGG16Frozen'+str(i))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    plt.savefig('VGG16FrozenTrainValGraph' + str(i))
    runs = i

# print(statistics)
# print("unzipped")
# # print(zip(*statistics))
#
#
# plt.style.use('seaborn-whitegrid')
#
# plt.errorbar(runs, x_accuracy, fmt='--o');
# plt.ylabel("Test accuracy")
# plt.xlabel("Trial number")
# plt.title("Standard deviation over ", str(runs), " runs")
# plt.savefig('ErrorbarsTestGraph')

