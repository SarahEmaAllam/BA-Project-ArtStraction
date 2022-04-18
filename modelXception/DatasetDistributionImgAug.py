import numpy as np
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

data_augmenter_model = data_augmentation_imgaug()
labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')
print("np_imgs=========================")
print(np_imgs[0])
print("length labels", len(labels))
new_labels = np.load('new_labels.npy')
print("new length labels", len(new_labels))
new_np_imgs = np.load('new_np_imgs.npy')
print("np_imgs=========================")
print(new_np_imgs[0])
exit()
imgs = np.concatenate((np_imgs, new_np_imgs))
labels = np.concatenate((labels, new_labels))
print("more length labels", len(labels))
# files = glob.glob('DATA/*')
# labels = [fn.split('/')[1].split('-')[0] for fn in files]
# labels = np.array(labels)

import matplotlib.pyplot as plt

# plt.hist(labels)
# plt.savefig('ClassDistribution')
# plt.show()
# plt.figure()

from sklearn.preprocessing import LabelEncoder

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
from numpy import where

counter = Counter(train_labels_enc)
print(counter)

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

for augmentation in range(4):
    class0idx = np.random.choice(class0.shape[0], size=int(len(class0) * 15 / 100), replace=False)

    class0c = data_augmenter_model(images=class0[class0idx])
    class0 = np.concatenate((class0c, class0), axis=0)

    class1idx = np.random.choice(class1.shape[0], size=int(len(class1) * 15 / 100), replace=False)
    class1c = data_augmenter_model(images=class1[class1idx])
    class1 = np.concatenate((class1c, class1), axis=0)

    class2idx = np.random.choice(class2.shape[0], size=int(len(class2) * 15 / 100), replace=False)
    class2c = data_augmenter_model(images=class2[class2idx])
    class2 = np.concatenate((class2c, class2), axis=0)

    class3idx = np.random.choice(class3.shape[0], size=int(len(class3) * 15 / 100), replace=False)
    class3c = data_augmenter_model(images=class3[class3idx])
    class3 = np.concatenate((class3c, class3), axis=0)

    class4idx = np.random.choice(class4.shape[0], size=int(len(class4) * 15 / 100), replace=False)
    class4c = data_augmenter_model(images=class4[class4idx])
    class4 = np.concatenate((class4c, class4), axis=0)

    class5idx = np.random.choice(class5.shape[0], size=int(len(class5) * 15 / 100), replace=False)
    class5c = data_augmenter_model(images=class5[class5idx])
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

fig, axs = plt.subplots(3)
fig.suptitle('Class Distribution After Stratified Shuffle')
axs[0].hist(labels_train)
axs[0].set_title('Training Data 70%')
axs[1].hist(labels_val)
axs[1].set_title('Validation Data 15%')
axs[2].hist(labels_test)
axs[2].set_title('Testing Data 15%')
plt.savefig('ClassDistributionStratifiedShuffleSplitImgAug')