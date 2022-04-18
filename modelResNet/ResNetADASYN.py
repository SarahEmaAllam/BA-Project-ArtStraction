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
def scheduler_cycle1(epoch, lr):
    # if epoch <= 15:  # 15 is the epoch when val loss < train loss
    #     return lr
    # if epoch > 15 and epoch <= 25:
    #     return lr / 10
    # if epoch > 25:
    #     return lr / 100
    print(lr)
    return lr * 2


def scheduler_cycle2(epoch, lr):
    print(lr)
    if epoch > 20:
        return lr
    else:
        return lr / 5


# from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

def plot_metrics(history, i):
    metrics = ['precision0', 'recall0', 'precision1', 'recall1', 'precision2', 'recall2', 'precision3', 'recall3',
               'precision4', 'recall4', 'precision5', 'recall5']
    for n, metric in enumerate(metrics):
        print(metric)
        # name = metric.replace("_"," ").capitalize()
        name = metric
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
        plt.savefig(os.path.join('performance_vizualizations', 'MetricsADASYN' + str(i)))


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


from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


from sklearn.metrics import f1_score


def f1(y_true, y_pred):
    # precision = precision_m(y_true, y_pred)
    # recall = recall_m(y_true, y_pred)
    # return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    return f1_score(y_true, y_pred, average='weighted')


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

    def on_epoch_end(self, epoch, logs={}):
        print('inside visualization')
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}ADASYN'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}ADASYN'))
        plt.close()


class PerformanceVisualizationCallbackTest(Callback):
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
        fig.savefig(os.path.join(self.image_dir, 'confusion_matrix_epoch_testADASYN'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, 'roc_curve_epoch_testADASYN'))
        plt.close()


import seaborn as sns
import pandas as pd

labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')

# batch_size = 36
batch_size = 100
num_classes = 6
epochs = 100
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
names = le.inverse_transform([0, 1, 2, 3, 4, 5])
print("ont-hot encoded classes for [0,1,2,3,4,5] : ", names)

#
# print("TRAINING LEBELS - TRIAN IMG")
# print(list(set(train_labels_enc) - set(train_imgs)))


#
# print(train_labels)
# print(train_labels_enc)


# Augmentation for training images


from collections import Counter
import imblearn
from imblearn import over_sampling
from imblearn.over_sampling import ADASYN
from numpy import where


counter = Counter(train_labels_enc)
print(counter)
image_dir = 'performance_vizualizations'
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(train_labels_enc == label)[0]
    print("X axis ", imgs_train[row_ix, 0])
    print("Y axis ", imgs_train[row_ix, 1])
    print("label", str(label))
    plt.scatter(imgs_train[row_ix, 0], imgs_train[row_ix, 1], label=str(label))

plt.legend()
plt.xlabel('X pixel value')
plt.ylabel('Y pixel value')
imgs_train = imgs_train.reshape(len(imgs_train), input_shape[0] * input_shape[1] * input_shape[2])
plt.savefig(os.path.join('DataSamplesDistributionPriorADASYN'))
plt.close()

# transform the dataset
oversample = ADASYN()
imgs_train, train_labels_enc = oversample.fit_resample(imgs_train, train_labels_enc)
# summarize the new class distribution
counter = Counter(train_labels_enc)
print(counter)

imgs_train = imgs_train.reshape(-1, input_shape[0], input_shape[1], input_shape[2])

for label, _ in counter.items():
    row_ix = where(train_labels_enc == label)[0]
    plt.scatter(imgs_train[row_ix, 0], imgs_train[row_ix, 1], label=str(label))

plt.legend()
print("===============PLOT DATA SAMPLES ====================================")
plt.savefig(os.path.join('DataSamplesDistributionADASYN'))
plt.close()


n = 0
for im, lab in zip(imgs_train, train_labels_enc):
    print("pair ", lab, im)
    if (lab == 4):
        print("inside ", lab)
        n = n + 1
        img = im.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = str(n) + '.png'
        cv2.imwrite(os.path.join('adasyn', name), img)

# print('Original dataset shape %s' % Counter(train_labels_enc))
# sm = SMOTE(sampling_strategy='auto', random_state=42)
# imgs_train, train_labels_enc = sm.fit_resample(imgs_train, train_labels_enc)
# print('Resampled dataset shape %s' % Counter(train_labels_enc))

# 1. Zooming the image randomly by a factor of 0.3 using the zoom_range parameter.
# 2. Rotating the image randomly by 50 degrees using the rotation_range parameter.
# 3. Translating the image randomly horizontally or vertically by a 0.2 factor of the image’s width or height using the width_shift_range
# and the height_shift_range parameters.
# 5. Randomly flipping half of the images horizontally using the horizontal_flip parameter.
# 6. Leveraging the fill_mode parameter to fill in new pixels for images after we apply any of the preceding operations
# (especially rotation or translation). In this case, we just fill in the new pixels with their nearest surrounding pixel values.

# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                    width_shift_range=0.2, height_shift_range=0.2,
#                                    horizontal_flip=True, fill_mode='nearest')
#
# val_datagen = ImageDataGenerator(rescale=1./255)


#
# #  CNN

# generate new augmented images
# train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=10)
# val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=5)
# input_shape = (300, 300, 3)
# import tensorflow_addons as tfa
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import layers
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

disable_eager_execution()

METRICS = [
    # tensorflow.keras.metrics.TruePositives(name='tp'),
    # tensorflow.keras.metrics.FalsePositives(name='fp'),
    # tensorflow.keras.metrics.TrueNegatives(name='tn'),
    # tensorflow.keras.metrics.FalseNegatives(name='fn'),
    tensorflow.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    # tfa.metrics.F1Score(num_classes=6, average='macro'),
    # tfa.metrics.F1Score(num_classes=6, average='weighted'),
    # tensorflow.keras.metrics.Precision(name='precision0', class_id=0),
    # tensorflow.keras.metrics.Recall(name='recall0', class_id=0),
    # tensorflow.keras.metrics.Precision(name='precision1', class_id=1),
    # tensorflow.keras.metrics.Recall(name='recall1', class_id=1),
    # tensorflow.keras.metrics.Precision(name='precision2', class_id=2),
    # tensorflow.keras.metrics.Recall(name='recall2', class_id=2),
    # tensorflow.keras.metrics.Precision(name='precision3', class_id=3),
    # tensorflow.keras.metrics.Recall(name='recall3', class_id=3),
    #  tensorflow.keras.metrics.Precision(name='precision4', class_id=4),
    # tensorflow.keras.metrics.Recall(name='recall4', class_id=4),
    # tensorflow.keras.metrics.Precision(name='precision5', class_id=5),
    # tensorflow.keras.metrics.Recall(name='recall5', class_id=5),
]

statistics = []
x_accuracy = []
x_loss = []

seedStart = 40
seedEnd = 43

for i in range(seedStart, seedEnd, 1):
    np.random.seed(i)
    # Define VGG16 base model
    base_model = ResNet50(
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
    # train_ds = preprocess_input(imgs_train)
    # validation_ds = preprocess_input(imgs_val)
    # test_ds = preprocess_input(imgs_test)

    train_ds = imgs_train

    validation_ds = imgs_val
    test_ds = imgs_test

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
    initial_LR = 1e-6  # should be a factor of 10 or 20 less than MAX_LR if one cycle is used
    patience = 5

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=initial_LR),
                  # optimizer=optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True)
    # history = model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
    callback_scheduler_cycle1 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle1)

    history = model.fit(x=train_ds, y=train_labels_enc,
                        validation_data=(validation_ds, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es, callback_scheduler_cycle1],
                        verbose=1)

    print("METRIC NAMES =====================", history.params['metrics'])
    copy_history = history

    MAX_LR = model.optimizer.lr
    print("MAX_LR", MAX_LR)
    MAX_EPOCHS = len(history.history['accuracy'])

    list_accuracy = history.history['accuracy']
    print("printing list of frozen acc")
    print(list_accuracy)
    list_loss = history.history['loss']

    acc = history.history['accuracy']
    list_val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    list_val_loss = history.history['val_loss']

    runs = seedEnd - seedStart

    # Unfreeze the base model
    # base_model.trainable = True

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    # LR = LR * 2 + 1e-5
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.000255),
                  # optimizer=optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True)
    callback_scheduler_cycle2 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle2)
    validation_data = (imgs_val, validation_labels_enc)
    performance_cbk = PerformanceVisualizationCallback(
        model=model,
        validation_data=validation_data,
        image_dir='performance_vizualizations')

    history = model.fit(x=imgs_train, y=train_labels_enc,
                        validation_data=(imgs_val, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=epochs - MAX_EPOCHS,
                        callbacks=[es, callback_scheduler_cycle2, performance_cbk],
                        verbose=1)
    # Train end-to-end. Be careful to stop before you overfit!
    print("printing list of tuned acc")

    copy_history_2 = history

    list_accuracy += history.history['accuracy']
    print(list_accuracy)
    list_loss += history.history['loss']

    list_val_acc += history.history['val_accuracy']
    list_val_loss += history.history['val_loss']

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    test_data = (imgs_test, test_labels_enc)
    performance_cbkt = PerformanceVisualizationCallbackTest(
        model=model,
        validation_data=test_data,
        image_dir='performance_vizualizations')


    print("Number of TEST SAMPLES:", len(labels_test), len(test_labels_enc))
    results = model.evaluate(x=imgs_test, y=test_labels_enc, batch_size=90, callbacks=[performance_cbkt])

    # plot and save F1 score
    y_pred = performance_cbkt.outputs[0]
    y_true = performance_cbkt.targets[0]
    print("Y PRED", np.round(y_pred))
    print("Y TRUE", np.round(y_true))
    from sklearn.metrics import classification_report

    scores = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], target_names=names, output_dict=True)
    print("Scores ", scores)

    # plot and save classification report

    classfic_report = sns.heatmap(pd.DataFrame(scores).iloc[:-1, :].T, annot=True)
    classfic_report = classfic_report.figure
    classfic_report.savefig(os.path.join(image_dir, 'test_plot_classif_reportADASYN.png'))

    plt.figure()
    print("model evaluate accuracy")
    print(results[1])
    print("list of accuracy evaluated ")
    accs = results[1].tolist()
    los = results[0].tolist()
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
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print("Generate predictions")
    # predictions = model.predict(test_imgs_scaled)
    # print("predictions shape:", predictions.shape)

    model.summary()
    model.save('modelVGGFineTune' + str(i))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    time = range(1, len(list_accuracy) + 1)
    print("standard deviation", std_per_epoch)
    print("list_accuracy", list_accuracy)
    std_val_acc = compute_sds(list_val_acc)
    print("list_val_acc", list_val_acc)
    print("=================================================================")
    print("time", time)
    print("x accuracy", x_accuracy)
    print("st per epoch", std_per_epoch)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    # plt.plot(time,history.history['accuracy'], label='Training Accuracy')

    plt.errorbar(time, x_accuracy, yerr=std_per_epoch, label='Training Accuracy')
    plt.errorbar(time, list_val_acc, yerr=std_per_epoch, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(time, x_loss, label='Training Loss')
    plt.plot(time, list_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('TrainValGraphFineTune' + str(i))
    runs = seedEnd - seedStart
    names_classes = le.inverse_transform([0, 1, 2, 3, 4, 5])
    print("ont-hot encoded classes for [0,1,2,3,4,5] : ", names_classes)
    plt.figure()
    # plotted = plot_metrics(copy_history_2, i)

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

