
import numpy as np


labels = np.load('labels.npy')
np_imgs = np.load('np_imgs.npy')
print("length labels", len(labels))
exit()
new_labels = np.load('new_labels.npy')
print("new length labels", len(new_labels))
new_np_imgs = np.load('new_np_imgs.npy')
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


idx = np.random.permutation(len(labels))  # get suffeled indices
imgs, labels = np_imgs[idx], labels[idx]  # uniform suffle of data and label

idx = np.random.permutation(len(np_imgs))  # get suffeled indices
imgs, labels = imgs[idx], labels[idx]  # uniform suffle of data and label


fig, axs = plt.subplots(3)
fig.suptitle('Class Distribution After Shuffle')
axs[0].hist(labels_train)
axs[0].set_title('Training Data 70%')
axs[1].hist(labels_val)
axs[1].set_title('Validation Data 15%')
axs[2].hist(labels_test)
axs[2].set_title('Testing Data 15%')
plt.savefig('ClassDistributionShuffledExtraSamples')
plt.close()
plt.cla()
plt.clf()

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
plt.savefig('ClassDistributionStratifiedShuffleSplitExtraSamples')