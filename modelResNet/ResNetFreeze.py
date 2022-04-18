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

# given a list of values
# we can calculate the mean by dividing the sum of the numbers over the length of the list
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


# we can then use the mean to calculate the variance
def calculate_variance(numbers):
    print("calcult variance", numbers)
    mean = calculate_mean(numbers)

    variance = 0
    for number in numbers:
        variance += (mean - number) ** 2

    return variance / len(numbers)


def calculate_standard_deviation(numbers):
    print("calculate stand dev", numbers)
    variance = calculate_variance(numbers)
    return np.sqrt(variance)


def scale_values(values):
    print("scale valueS", values)
    std = calculate_standard_deviation(values)
    mean = calculate_mean(values)
    transformed_values = list()
    for value in values:
        transformed_values.append((value - mean) / std)
    return transformed_values


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


from tensorflow.keras.callbacks import Callback


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
        fig.savefig(os.path.join(self.image_dir, 'confusion_matrix_epoch_testfreeze'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, 'roc_curve_epoch_testfreeze'))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

import seaborn as sns
import pandas as pd

from scikitplot.metrics import plot_confusion_matrix, plot_roc
import os

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

seedStart = 40
seedEnd = 45


list_trials_test_acc = []
list_trials_test_loss = []
list_trials_acc = []
list_trials_val_acc = []
list_trials_loss = []
list_trials_val_loss = []
list_trials_val_acc_np = np.array(list_trials_val_acc)
list_trials_val_loss_np = np.array(list_trials_val_loss)
labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')


batch_size = 36
num_classes = 6
epochs = 20
input_shape = (224, 224, 3)

idx = np.random.permutation(len(np_imgs)) # get suffeled indices
imgs, labels = np_imgs[idx], labels[idx] # uniform suffle of data and label

imgs_train, imgs_val, imgs_test = np.split(imgs, [int(len(imgs)*0.8), int(len(imgs)*0.9)]) # split of 75:15:10
labels_train, labels_val, labels_test = np.split(labels, [int(len(labels)*0.8), int(len(labels)*0.9)])

# print(len(imgs_train),len(imgs_val),len(imgs_test))
# print(imgs_train[:3])
# print(labels_train[:3])

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(labels_train)
train_labels_enc = le.transform(labels_train)
validation_labels_enc = le.transform(labels_val)
test_labels_enc = le.transform(labels_test)
names = le.inverse_transform([0, 1, 2, 3, 4, 5])
print("ont-hot encoded classes for [0,1,2,3,4,5] : ", names)

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

statistics = []
x_accuracy = []
x_loss =[]

for i in range(40, 45, 1):
    np.random.seed(i)
    # Define VGG16 base model
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    # Freeze layers
    base_model.trainable = False
    # print(input_shape)
    # print(train_imgs_scaled.shape)
    # print(validation_imgs_scaled.shape)
    # print(type(train_labels_enc))
    # print(type(validation_labels_enc))
    # Create new model on top of it
    
    train_ds = preprocess_input(imgs_train)

    validation_ds = preprocess_input(imgs_val)
    test_ds = preprocess_input(imgs_test)
    
    # train_ds = imgs_train

    # validation_ds = imgs_val
    # test_ds = imgs_test
    
    inputs = tensorflow.keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = tensorflow.keras.layers.Dense(6, activation='softmax')(x)
    model = tensorflow.keras.Model(inputs, outputs)


    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  #optimizer=optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=25, restore_best_weights=True)
    # history = model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])

    history = model.fit(x=train_ds, y=train_labels_enc,
                        validation_data=(validation_ds, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=100,
                        callbacks=[es],
                        verbose=1)

    acc1 = history.history['val_accuracy']
    list_val_acc = history.history['val_accuracy']

    print("printing list of tuned acc")

    acc = history.history['val_accuracy']
    loss = history.history['val_loss']
    list_trials_val_acc_np = np.append(list_trials_val_acc_np, acc, axis=0)
    list_trials_val_loss_np = np.append(list_trials_val_loss_np, loss, axis=0)
    list_trials_val_acc.append(acc)
    list_trials_val_loss.append(loss)

    np.save('freeze_list_trials_val_acc_np.npy', list_trials_val_acc_np)
    np.save('freeze_list_trials_val_loss_np.npy', list_trials_val_loss_np)
    np.save('freeze_list_trials_val_acc.npy', np.array(list_trials_val_acc))
    np.save('freeze_list_trials_val_loss.npy', np.array(list_trials_val_loss))
    np.save('freeze_list_trials_val_acc_nonnp.npy', list_trials_val_acc)
    np.save('freeze_list_trials_val_loss_nonnp.npy', list_trials_val_loss)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    test_data = (imgs_test, test_labels_enc)
    performance_cbkt = PerformanceVisualizationCallbackTest(
        model=model,
        validation_data=test_data,
        image_dir='performance_vizualizations')

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_data, test_labels_enc, batch_size=32, callbacks=[performance_cbkt])
    list_trials_test_acc.append(results[1])
    list_trials_test_loss.append(results[0])
    np.save('freeze_list_trials_test_loss.npy', list_trials_test_loss)
    np.save('freeze_list_trials_test_acc.npy', list_trials_test_acc)

    # plot and save F1 score
    y_pred = performance_cbkt.outputs[0]
    y_true = performance_cbkt.targets[0]
    print("Y PRED", np.round(y_pred))
    print("Y TRUE", np.round(y_true))
    from sklearn.metrics import classification_report

    scores = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], target_names=names, output_dict=True)
    print("Scores ", scores)

    # plot and save classification report
    image_dir = 'performance_vizualizations'
    classfic_report = sns.heatmap(pd.DataFrame(scores).iloc[:-1, :].T, annot=True)
    classfic_report = classfic_report.figure
    classfic_report.savefig(os.path.join(image_dir, 'test_plot_classif_report_freeze' + str(i) + '.png'))

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # print(list(statistics))
    # print("statistics:" , statistics)
    print("test loss, test acc:", results)