from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np
import cv2

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
        # if negative:
        #     image = (resized_image / 127.5) - 1
        # else:
        #     image = (resized_image / 255.0)
        normalized_data.append(resized_image)
    return normalized_data



files = glob.glob('DATA\\*')
labels = [fn.split('\\')[1].split('-')[0] for fn in files]
print(labels)
print("set labels", set(labels))


data_path = os.path.join(os.getcwd(), 'DATA')
imgs = load_images(data_path)
normalized_imgs = normalize_data(imgs, 299, 299, negative=False)
np_imgs = np.array(normalized_imgs)
np.save('np_imgs.npy', np_imgs)

labels = np.array(labels)
np.save('labels.npy', labels)

print(np_imgs[40])
print(labels[40])
