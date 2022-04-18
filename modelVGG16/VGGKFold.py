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


from imgaug import augmenters as iaa


def data_augmentation_imgaug():  # Graces
    """
    Data augmentation using imgaug library which provides more flexibility and diversity in augmentation techniques.
    Returns:
        Sequential image augmenter.
    """
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.7, aug)


    seq = iaa.Sequential([
        iaa.Fliplr(0.7),  # horizontal flips
        sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.1))),  # sharpen images)
        iaa.ContrastNormalization((1.2, 1.6)),
        sometimes(iaa.Crop(percent=(0.2, 0.2, 0.2, 0.2), keep_size=True)),
    ], random_order=True)  # apply augmenters in random order
    return seq

from sklearn.metrics import f1_score
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import os
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        # self.var_y_true = tf.Variable(0., validate_shape=False)
        # self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_test_end(self, epoch, logs={}):
        print('======================================inside visualization====================================')
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]
        y_pred_class = np.argmax(y_pred, axis=1)

        self.targets.append(y_true)
        self.outputs.append(y_pred_class)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, 'confusion_matrix_epoch_testKFold'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, 'roc_curve_epoch_testKFold'))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()


def scheduler_cycle1(epoch, lr):
    # if epoch <= 15:  # 15 is the epoch when val loss < train loss
    #     return lr
    # if epoch > 15 and epoch <= 25:
    #     return lr / 10
    # if epoch > 25:
    #     return lr / 100
    print(lr)
    return lr + 0.0002

def scheduler_cycle2(epoch, lr):
    print(lr)
    if epoch > 20:
        return lr
    else:
        return lr - 0.0002


def compute_sds(list_accuracy):
    sd_per_epoch = []

    for index, acc in enumerate(list_accuracy):
        if index != 0:
            sd = list_accuracy[0:index + 1]

            sd_epoch = np.std(sd)
            sd_per_epoch.append(sd_epoch)
        else:
            sd_per_epoch.append(0)
    return sd_per_epoch


# Custom callback for model.fit to save the standard deviation
class callback_std(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('HEREEEEEEEEEEEEEEEEEE')
        print(self.model.history)
        print(self.model.history.history)
        print(self.model.history.history['acc'])
        print(self.model.history.history['accuracy'])


# callback for scheduled learning rate: decrease lr by /10 after N epochs
def scheduler(epoch, lr):
    if epoch <= 15:  # 15 is the epoch when val loss < train loss
        return lr
    if epoch > 15 and epoch <= 25:
        return lr / 10
    if epoch > 25:
        return lr / 100
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
        if negative:
            image = (resized_image / 127.5) - 1
        else:
            image = (resized_image / 255.0)
        normalized_data.append(image)
    return normalized_data



import seaborn as sns
import pandas as pd

labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')
print("length labels", len(labels))
new_labels = np.load('new_labels.npy')
print("new length labels", len(new_labels))
new_np_imgs = np.load('new_np_imgs.npy')
np_imgs = np.concatenate((np_imgs, new_np_imgs), axis=0)
labels = np.concatenate((labels, new_labels), axis=0)
print("more length labels", len(labels))

idx = np.random.permutation(len(np_imgs))  # get suffeled indices
imgs, labels = np_imgs[idx], labels[idx]  # uniform suffle of data and label


imgs_train, imgs_test = np.split(imgs, [int(len(imgs) * 0.85)])  # split of 75:15:10
print('len train, len test', len(imgs_train), len(imgs_test))
labels_train, labels_test = np.split(labels, [int(len(labels) * 0.85)])
print('len train, len test', len(labels_train), len(labels_test))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(labels)
labels_train = le.transform(labels_train)
labels_test = le.transform(labels_test)

# im = imgs_train[len(imgs_train)-1].astype(np.uint8)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# cv2.imshow('', im)
# cv2.waitKey(0)


batch_size = 32
num_classes = 6
epochs = 80
input_shape = (224, 224, 3)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((imgs_train, imgs_test), axis=0)
targets = np.concatenate((labels_train, labels_test), axis=0)

num_folds = 10

from sklearn.model_selection import StratifiedKFold
# Define the K-fold Cross Validator
kfold = StratifiedKFold(n_splits=num_folds)

#
# print("TRAINING LEBELS - TRIAN IMG")

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



np.random.seed(40)

# Augmentation for training images

# 1. Zooming the image randomly by a factor of 0.3 using the zoom_range parameter.
# 2. Rotating the image randomly by 50 degrees using the rotation_range parameter.
# 3. Translating the image randomly horizontally or vertically by a 0.2 factor of the image’s width or height using the width_shift_range
# and the height_shift_range parameters.
# 5. Randomly flipping half of the images horizontally using the horizontal_flip parameter.
# 6. Leveraging the fill_mode parameter to fill in new pixels for images after we apply any of the preceding operations
# (especially rotation or translation). In this case, we just fill in the new pixels with their nearest surrounding pixel values.

# train_datagen = ImageDataGenerator( zoom_range=0.2, brightness_range=[0.4,1.2],
#                                     height_shift_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')

# val_datagen = ImageDataGenerator(rescale=1./255)


#
# #  CNN

# generate new augmented images
# train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=10)
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=5)
# input_shape = (300, 300, 3)

# Define VGG16 base model
# K-fold Cross Validation model evaluation
fold_no = 1


for train, test in kfold.split(inputs, targets):
    print("inputs", len(inputs[train]), inputs[train].shape)
    print("targets", len(targets[train]), targets[train].shape)

    # data_aug = tensorflow.keras.Sequential([
    #     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #     layers.experimental.preprocessing.RandomRotation(0.2),
    # ])

    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # Freeze layers
    base_model.trainable = False

    train_ds = preprocess_input(inputs[train])
    test_ds = preprocess_input(inputs[test])

    input = tensorflow.keras.Input(shape=input_shape)
    x = base_model(input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = tensorflow.keras.layers.Dense(6, activation='softmax')(x)
    model = tensorflow.keras.Model(input, outputs)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
    # history = model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
    callback_scheduler_cycle1 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle1)

    history = model.fit(x=train_ds, y=targets[train],
                        batch_size=batch_size,
                        epochs=25,
                        callbacks=[es, callback_scheduler_cycle1],
                        verbose=1)

    list_accuracy = history.history['accuracy']
    print("printing list of frozen acc")
    print(list_accuracy)
    list_loss = history.history['loss']

    acc = history.history['accuracy']

    loss = history.history['loss']


    # Unfreeze the base model
    base_model.trainable = True

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-5),
                  # optimizer=optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    callback_scheduler_cycle2 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle2)
    performance_cbk = PerformanceVisualizationCallback(
        model=model,
        validation_data=inputs[train],
        image_dir='performance_vizualizations')
    es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
    history = model.fit(x=train_ds, y=targets[train],
                        batch_size=batch_size,
                        epochs=30,
                        callbacks=[es, callback_scheduler_cycle2, performance_cbk],
                        verbose=1)
    # Train end-to-end. Be careful to stop before you overfit!
    print("printing list of tuned acc")

    list_accuracy += history.history['accuracy']
    print(list_accuracy)
    list_loss += history.history['loss']


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    scores = model.evaluate(test_ds, targets[test], batch_size=32)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fold_no = fold_no + 1

    print("model evaluate accuracy")
    print(scores[1])
    print("list of accuracy evaluated ")
    accs = scores[1].tolist()
    los = scores[0].tolist()
    print(accs)

    # list_accuracy.append(accs)
    # list_loss.append(los)
    print(list_accuracy)
    print(list_loss)

    y_labels = model.metrics_names
    x_accuracy = list_accuracy
    x_loss = list_loss

    std_per_epoch = compute_sds(list_accuracy)

    # statistic = zip(model.metrics_names, results)
    # statistics.append(statistic)
    # print(statistics)
    # c = list.append(statistics)
    # print(list(statistics))
    # print("statistics:" , statistics)
    print("test loss, test acc:", scores)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print("Generate predictions")
    # predictions = model.predict(test_imgs_scaled)
    # print("predictions shape:", predictions.shape)

    model.summary()



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

