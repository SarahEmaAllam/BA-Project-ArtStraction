import tensorflow

import numpy as np

import cv2


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
    return lr * 1.5


def scheduler_cycle2(epoch, lr):
    print(lr)
    if epoch > 30:
        return lr
    else:
        return lr / 1.3


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
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}ImgAug'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}ImgAug'))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()


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
        fig.savefig(os.path.join(self.image_dir, 'confusion_matrix_epoch_testImgAug'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        fig.savefig(os.path.join(self.image_dir, 'roc_curve_epoch_testImgAug'))
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()


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
        sometimes(iaa.Sharpen(lightness=(0.85, 1.1))),  # sharpen images)
        sometimes(iaa.ContrastNormalization((0.9, 1.1))),
        sometimes(iaa.Crop(percent=(0.1, 0.1, 0.1, 0.1), keep_size=True)),
    ], random_order=True)  # apply augmenters in random order
    return seq



from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

seedStart = 40
seedEnd = 45

list_trials_acc = []
list_trials_val_acc = []
list_trials_loss = []
list_trials_val_loss = []
list_trials_val_acc_np = np.array(list_trials_val_acc)
list_trials_val_loss_np = np.array(list_trials_val_loss)

for trial in range(seedStart, seedEnd, 1):

    # Initiate data augmenter
    data_augmenter_model0 = data_augmentation_imgaug()
    data_augmenter_model1 = data_augmentation_imgaug()
    data_augmenter_model2 = data_augmentation_imgaug()
    data_augmenter_model3 = data_augmentation_imgaug()
    data_augmenter_model4 = data_augmentation_imgaug()
    data_augmenter_model5 = data_augmentation_imgaug()

    import seaborn as sns
    import pandas as pd

    labels = np.load('labels_xception.npy')
    np_imgs = np.load('np_imgs_xception.npy')
    print(np_imgs[0])
    print("length labels", len(labels))

    new_labels = np.load('new_labels_xception.npy')
    print("new length labels", len(new_labels))
    new_np_imgs = np.load('new_np_imgs_xception.npy')
    np_imgs = np.concatenate((np_imgs, new_np_imgs), axis=0)
    labels = np.concatenate((labels, new_labels), axis=0)
    print(new_np_imgs[0])
    print("more length labels", len(labels))

    # batch_size = 36
    # batch_size = 100
    # num_classes = 6
    # epochs = 100
    # input_shape = (299, 299, 3)

    batch_size = 60
    num_classes = 6
    epochs = 60
    input_shape = (299, 299, 3)

    idx = np.random.permutation(len(np_imgs))  # get suffeled indices
    imgs, labels = np_imgs[idx], labels[idx]  # uniform suffle of data and label

    from sklearn.model_selection import StratifiedShuffleSplit

    stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    for train_index, test_index in stratSplit.split(imgs, labels):
        print("inside first split")
        imgs_train, imgs_valtest = imgs[train_index], imgs[test_index]
        labels_train, labels_valtest = labels[train_index], labels[test_index]

    valTestSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    for train_index, test_index in valTestSplit.split(imgs_valtest, labels_valtest):
        print("inside first split 2")

        imgs_val, imgs_test = imgs_valtest[train_index], imgs_valtest[test_index]
        labels_val, labels_test = labels_valtest[train_index], labels_valtest[test_index]

    # encode text category labels
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(labels_train)
    train_labels_enc = le.transform(labels_train)
    validation_labels_enc = le.transform(labels_val)
    test_labels_enc = le.transform(labels_test)
    names = le.inverse_transform([0, 1, 2, 3, 4, 5])
    print("ont-hot encoded classes for [0,1,2,3,4,5] : ", names)

    from collections import Counter
    import imblearn
    from imblearn import over_sampling
    from imblearn.over_sampling import SMOTE
    from numpy import where

    # transform the dataset
    imgs_train = imgs_train.reshape(len(imgs_train), input_shape[0] * input_shape[1] * input_shape[2])
    print("pre shape res", imgs_train.shape)

    # transform the dataset
    oversample = SMOTE()
    imgs_train, train_labels_enc = oversample.fit_resample(imgs_train, train_labels_enc)
    # summarize the new class distribution
    counter = Counter(train_labels_enc)
    print(counter)

    imgs_train = imgs_train.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    print("POOOOOOOOOOOOST shape res", imgs_train.shape)

    counter = Counter(train_labels_enc)
    print(counter)
    image_dir = 'performance_vizualizations'


    # im = imgs_train[0].astype(np.uint8)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # cv2.imshow('', im)
    # cv2.waitKey(0)

    # Iterate through batches
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []

    for i, img in enumerate(imgs_train):
        if train_labels_enc[i] == 0:
            class0.append(img)
        if train_labels_enc[i] == 1:
            class1.append(img)
        if train_labels_enc[i] == 2:
            class2.append(img)
        if train_labels_enc[i] == 3:
            class3.append(img)
        if train_labels_enc[i] == 4:
            class4.append(img)
        if train_labels_enc[i] == 5:
            class5.append(img)

    class0 = np.array(class0)
    class1 = np.array(class1)
    class2 = np.array(class2)
    class3 = np.array(class3)
    class4 = np.array(class4)
    class5 = np.array(class5)
    # Iterate through batches



    for augmentation in range(2):
        class0idx = np.random.choice(class0.shape[0], size=int(len(class0) * 60 / 100), replace=False)

        class0c = data_augmenter_model0(images=class0[class0idx])
        class0 = np.concatenate((class0c, class0), axis=0)

        class1idx = np.random.choice(class1.shape[0], size=int(len(class1) * 60 / 100), replace=False)
        class1c = data_augmenter_model0(images=class1[class1idx])
        class1 = np.concatenate((class1c, class1), axis=0)

        class2idx = np.random.choice(class2.shape[0], size=int(len(class2) * 60 / 100), replace=False)
        class2c = data_augmenter_model0(images=class2[class2idx])
        class2 = np.concatenate((class2c, class2), axis=0)

        class3idx = np.random.choice(class3.shape[0], size=int(len(class3) * 60 / 100), replace=False)
        class3c = data_augmenter_model0(images=class3[class3idx])
        class3 = np.concatenate((class3c, class3), axis=0)

        class4idx = np.random.choice(class4.shape[0], size=int(len(class4) * 60 / 100), replace=False)
        class4c = data_augmenter_model0(images=class4[class4idx])
        class4 = np.concatenate((class4c, class4), axis=0)

        class5idx = np.random.choice(class5.shape[0], size=int(len(class5) * 60 / 100), replace=False)
        class5c = data_augmenter_model0(images=class5[class5idx])
        class5 = np.concatenate((class5c, class5), axis=0)

    # Augmentation for training images

    imgs_train = np.concatenate((class0, class1, class2, class3, class4, class5), axis=0)
    print("after concatenate =====================================")
    print(imgs_train[0])
    train_labels = []
    for index in class0:
        train_labels.append('Antinous')
    for index in class1:
        train_labels.append('Graces')
    for index in class2:
        train_labels.append('Pieta')
    for index in class3:
        train_labels.append('Reclining')
    for index in class4:
        train_labels.append('Sebastian')
    for index in class5:
        train_labels.append('Tanagra')
    labels_train = np.array(train_labels)

    idx = np.random.permutation(len(imgs_train))  # get suffeled indices
    imgs_train, labels_train = imgs_train[idx], labels_train[idx]  # uniform suffle of data and label

    le = LabelEncoder()
    le.fit(labels_train)
    train_labels_enc = le.transform(labels_train)
    validation_labels_enc = le.transform(labels_val)
    test_labels_enc = le.transform(labels_test)
    names = le.inverse_transform([0, 1, 2, 3, 4, 5])
    print("ont-hot encoded classes for [0,1,2,3,4,5] : ", set(test_labels_enc))

    counter = Counter(train_labels_enc)
    print(counter)

    # batch_size = 60
    # num_classes = 6
    # epochs = 60
    # input_shape = (299, 299, 3)

    from tensorflow.keras.applications.xception import Xception
    from tensorflow.keras.models import Model
    from tensorflow.keras import optimizers
    from tensorflow.python.framework.ops import disable_eager_execution
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping

    disable_eager_execution()

    statistics = []
    x_accuracy = []
    x_loss = []

    # Define VGG16 base model
    base_model = Xception(
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
    inputs = tensorflow.keras.Input(shape=input_shape)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=True)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = tensorflow.keras.layers.Dense(6, activation='softmax')(x)
    model = tensorflow.keras.Model(inputs, outputs)

    initial_LR = 1e-6  # should be a factor of 10 or 20 less than MAX_LR if one cycle is used
    patience = 5
    initial_momentum = 0.95
    MAX_momentum = 0.85

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=initial_LR),
                  # optimizer=optimizers.SGD(learning_rate=initial_LR, momentum=initial_momentum, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)
    # history = model.fit(train_ds, train_labels, epochs=50, validation_split=0.2, batch_size=32, callbacks=[es])
    callback_scheduler_cycle1 = tensorflow.keras.callbacks.LearningRateScheduler(scheduler_cycle1)

    history = model.fit(x=imgs_train, y=train_labels_enc,
                        validation_data=(imgs_val, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=epochs,
                        # callbacks=[es, callback_scheduler_cycle1, CustomMomentumScheduler(scheduler_momentum)],
                        callbacks=[es, callback_scheduler_cycle1],
                        verbose=1)

    MAX_LR = model.optimizer.lr
    print("MAX_LR", MAX_LR)
    MAX_EPOCHS = len(history.history['accuracy'])

    list_accuracy = history.history['accuracy']
    print("printing list of frozen acc")
    print(list_accuracy)
    list_loss = history.history['loss']

    acc1 = history.history['val_accuracy']
    list_val_acc = history.history['val_accuracy']

    loss1 = history.history['val_loss']
    list_val_loss = history.history['val_loss']

    runs = seedEnd - seedStart

    # Unfreeze the base model
    base_model.trainable = True

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.000255),
                  # optimizer=optimizers.SGD(learning_rate=0.000255, momentum=MAX_momentum, nesterov=True),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)
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
    acc2 = history.history['val_accuracy']
    print("acc2", acc2)
    loss2 = history.history['val_loss']

    acc = np.concatenate((acc1, acc2), axis=0)
    loss = np.concatenate((loss1, loss2), axis=0)

    list_trials_val_acc_np = np.append(list_trials_val_acc_np, acc, axis=0)
    list_trials_val_loss_np = np.append(list_trials_val_loss_np, loss, axis=0)
    list_trials_val_acc.append(acc)
    list_trials_val_loss.append(loss)

    np.save('list_trials_val_acc_np.npy', list_trials_val_acc_np)
    np.save('list_trials_val_loss_np.npy', list_trials_val_loss_np)
    np.save('list_trials_val_acc.npy', np.array(list_trials_val_acc))
    np.save('list_trials_val_loss.npy', np.array(list_trials_val_loss))
    np.save('list_trials_val_acc_nonnp.npy', list_trials_val_acc)
    np.save('list_trials_val_loss_nonnp.npy', list_trials_val_loss)

    list_accuracy += history.history['accuracy']
    print(list_accuracy)
    list_loss += history.history['loss']

    # list_val_acc += history.history['val_accuracy']
    # list_val_loss += history.history['val_loss']

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    test_data = (imgs_test, test_labels_enc)
    performance_cbkt = PerformanceVisualizationCallbackTest(
        model=model,
        validation_data=test_data,
        image_dir='performance_vizualizations')

    print("Number of TEST SAMPLES:", len(labels_test), len(test_labels_enc))
    results = model.evaluate(x=imgs_test, y=test_labels_enc, batch_size=32, callbacks=[performance_cbkt])

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
    classfic_report.savefig(os.path.join(image_dir, 'test_plot_classif_report_imgaug' + str(trial) + '.png'))

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
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
    model.save('modelXceptionImgAug' + str(trial))
