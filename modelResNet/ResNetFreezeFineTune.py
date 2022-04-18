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
    if epoch > 30:
        return lr
    else:
        return lr / 2


class CustomMomentumScheduler(tensorflow.keras.callbacks.Callback):
    """Momentum scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new momentum as output (float).
  """

    def __init__(self, schedule):
        super(CustomMomentumScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tensorflow.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduler_momentum = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tensorflow.keras.backend.set_value(self.model.optimizer.momentum, scheduler_momentum)
        print("\nEpoch %05d: Momentum is %6.4f." % (epoch, scheduler_momentum))


def scheduler_momentum(epoch, lr):
    if lr > 0.00025:
        return 0.80
    else:
        return (0.95 - 500 * lr)
    # print("momentum", momentum)
    # tensorflow.keras.backend.set_value(model.optimizer.lr, scheduled_lr)


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


labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')


batch_size = 36
num_classes = 6
epochs = 80
input_shape = (224, 224, 3)

idx = np.random.permutation(len(np_imgs))  # get suffeled indices
imgs, labels = np_imgs[idx], labels[idx]  # uniform suffle of data and label

imgs_train, imgs_val, imgs_test = np.split(imgs, [int(len(imgs) * 0.80), int(len(imgs) * 0.9)])  # split of 75:15:10
labels_train, labels_val, labels_test = np.split(labels, [int(len(labels) * 0.80), int(len(labels) * 0.9)])

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
x_loss = []

seedStart = 40
seedEnd = 45

for i in range(seedStart, seedEnd, 1):
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
    # train_ds = preprocess_input(imgs_train)
    # validation_ds = preprocess_input(imgs_val)
    # test_ds = preprocess_input(imgs_test)

    train_ds = imgs_train

    validation_ds = imgs_val
    test_ds = imgs_test

    inputs = tensorflow.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
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
    initial_momentum = 0.95
    MAX_momentum = 0.85

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=initial_LR),
                  # optimizer=optimizers.SGD(learning_rate=initial_LR, momentum=initial_momentum, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=15, restore_best_weights=True)
    # history = model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
    callback_scheduler_cycle1 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle1)

    history = model.fit(x=imgs_train, y=train_labels_enc,
                        validation_data=(imgs_val, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es, callback_scheduler_cycle1, CustomMomentumScheduler(scheduler_momentum)],
                        # callbacks=[es, callback_scheduler_cycle1],
                        verbose=1)

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
    base_model.trainable = True

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    # LR = LR * 2 + 1e-5
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.000255),
                  # optimizer=optimizers.SGD(learning_rate=0.000255, momentum=MAX_momentum, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=25, restore_best_weights=True)
    callback_scheduler_cycle2 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle2)
    history = model.fit(x=imgs_train, y=train_labels_enc,
                        validation_data=(imgs_val, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=epochs - MAX_EPOCHS,
                        callbacks=[es, callback_scheduler_cycle2, CustomMomentumScheduler(scheduler_momentum)],
                        # callbacks=[es, callback_scheduler_cycle2],
                        verbose=1)
    # Train end-to-end. Be careful to stop before you overfit!
    print("printing list of tuned acc")

    list_accuracy += history.history['accuracy']
    print(list_accuracy)
    list_loss += history.history['loss']

    list_val_acc += history.history['val_accuracy']
    list_val_loss += history.history['val_loss']

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(imgs_test, test_labels_enc, batch_size=32)
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

