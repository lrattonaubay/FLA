import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pandas as pd
import tensorflow as tf
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(cls, cutout_length=0):
    MEAN = [0.49139968]
    STD = [0.24703233]
    transf = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid


def read_data(batch_size, post_KDARTS=False):

    df = pd.read_csv('./data/fer2013.csv')
    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
    img_array = np.stack(img_array, axis=0)
    le = LabelEncoder()
    img_labels = le.fit_transform(df.emotion)
    img_labels = tf.keras.utils.to_categorical(img_labels)

    X_TRAIN, X_test, y_TRAIN, y_test = train_test_split(
		img_array,
		img_labels,
		shuffle=True, 
		stratify=img_labels,
		test_size=0.1,
        random_state=42
    )

    if len(X_TRAIN)%2 == 1:
        X_TRAIN = np.delete(X_TRAIN, -1, 0)
        y_TRAIN = np.delete(y_TRAIN, -1, 0)

    X_train, X_valid, y_train, y_valid = train_test_split(
		X_TRAIN,
		y_TRAIN,
		stratify=y_TRAIN,
		test_size=0.5,
        random_state=42
    )

    X_train = X_train / 255.
    X_valid = X_valid / 255.

    # For TensorFlow => teacher predictions

    train_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    valid_tf = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size)
    test_tf = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    if post_KDARTS == True :
        tf_train = tf.data.Dataset.from_tensor_slices((X_TRAIN / 255., y_TRAIN)).batch(batch_size)
        tf_test = tf.data.Dataset.from_tensor_slices((X_test / 255., y_test)).batch(batch_size)
        
        return tf_train, tf_test

    # for PyTorch => KDARTS compile

    X_train = X_train.reshape(-1,1,48,48)
    X_valid = X_valid.reshape(-1,1,48,48)
    X_test = X_test.reshape(-1,1,48,48)

    train_pt = TensorDataset(torch.Tensor(X_train),torch.Tensor(y_train))
    valid_pt = TensorDataset(torch.Tensor(X_valid),torch.Tensor(y_valid))
    test_pt = TensorDataset(torch.Tensor(X_test),torch.Tensor(y_test))

    return (train_tf, valid_tf, test_tf), ((train_pt, valid_pt), test_pt)
