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
    mean = calculate_mean(numbers)

    variance = 0
    for number in numbers:
        variance += (mean - number) ** 2

    return variance / len(numbers)


def calculate_standard_deviation(numbers):
    variance = calculate_variance(numbers)
    return np.sqrt(variance)

def scale_values(values):
    std = calculate_standard_deviation(values)
    mean = calculate_mean(values)
    transformed_values = list()
    for value in values:
        transformed_values.append((value-mean)/std)
    return transformed_values

train_files = glob.glob('training_data/*')
train_path = os.path.join(os.getcwd(), 'training_data')
print(train_path)
imgs = load_images(train_path)
train_imgs = normalize_data(imgs, 299, 299, negative=False)
# train_imgs = [img_to_array(img) for img in train_path]
# train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
np.save('train_imgs.npy', train_imgs)
# train_imgs = np.array(train_imgs)
train_labels = [fn.split('/')[1].split('-')[0] for fn in train_files]



# print(train_labels)
# imgs_val = load_images(r'C:\Users\Sarah Allam\Desktop\Bachelor Project\Code\Model 1\validation_data\\')
val_path = os.path.join(os.getcwd(), 'validation_data')
imgs_val = load_images(val_path)
validation_files = glob.glob('validation_data/*')
validation_imgs = normalize_data(imgs_val, 299, 299, negative=False)
# validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
np.save('validation_imgs.npy', validation_imgs)
validation_labels = [fn.split('/')[1].split('-')[0] for fn in validation_files]
# print(validation_labels)

test_path = os.path.join(os.getcwd(), 'test_data')
imgs_test = load_images(test_path)
test_files = glob.glob('test_data/*')
test_imgs = normalize_data(imgs_test, 299, 299, negative=False)
# validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
test_imgs = np.array(test_imgs)
np.save('test_imgs.npy', test_imgs)
test_labels = [fn.split('/')[1].split('-')[0] for fn in test_files]

print('Train dataset shape:', train_imgs.shape,
      '\tValidation dataset shape:', validation_imgs.shape)

train_imgs_scaled = train_imgs.astype('float32')
print(train_imgs_scaled)
print(train_imgs_scaled.shape)
# cv2.imshow('', train_imgs_scaled[0])
# cv2.waitKey(0)
validation_imgs_scaled = validation_imgs.astype('float32')
test_imgs_scaled = test_imgs.astype('float32')
# train_imgs_scaled /= 255
# validation_imgs_scaled /= 255

print(test_imgs[0].shape)
array_to_img(test_imgs[0])

batch_size = 32
num_classes = 6
epochs = 30
input_shape = (299, 299, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)
test_labels_enc = le.transform(test_labels)

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
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import layers
disable_eager_execution()

statistics = []
x_accuracy = []
x_loss =[]

seedStart = 40
seedEnd = 45

for i in range(seedStart, seedEnd, 1):
    np.random.seed(i)
    # Define VGG16 base model
    base_model = Xception(
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
    inputs = tensorflow.keras.Input(shape=input_shape)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = tensorflow.keras.layers.Dense(6, activation='sigmoid')(x)
    model = tensorflow.keras.Model(inputs, outputs)


    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  # optimizer=optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                        validation_data=(validation_imgs_scaled, validation_labels_enc),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_imgs_scaled, test_labels_enc, batch_size=32)
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
    model.save('modelVGGFrozen'+str(i))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    time = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, acc, label='Training Accuracy')
    plt.plot(time, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(time, loss, label='Training Loss')
    plt.plot(time, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('TrainValGraph'+str(i))
    runs = seedEnd - seedStart

plt.figure()
# print(statistics)
print("unzipped")
# print(zip(*statistics))


# Return the standard deviation values of the accuracy and loss points
sd_accuracy = scale_values(x_accuracy)
print(x_accuracy)
print(x_loss)
sd_loss = scale_values(x_loss)

runs = range(1, len(x_accuracy) + 1)
print(runs)


plt.plot(runs, x_accuracy, 'o')
plt.errorbar(runs, x_accuracy, yerr=sd_accuracy, fmt='o', ecolor='black', capsize=5);
plt.ylabel("Test accuracy")
plt.xlabel("Trial number")
plt.title("Standard deviation over runs")
plt.savefig('ErrorbarsTestFrozenAcc')

plt.figure()

plt.plot(runs, x_loss, 'o')
plt.errorbar(runs, x_loss, yerr=sd_loss, fmt='o', ecolor='black', capsize=5);
plt.ylabel("Test loss")
plt.xlabel("Trial number")
plt.title("Standard deviation over runs")
plt.savefig('ErrorbarsTestFrozen')

