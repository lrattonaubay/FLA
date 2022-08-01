import pandas as pd

from functions import *
from MLPNAS import MLPNAS
import tensorflow as tf
import time
from params import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train_dumm = pd.get_dummies(y_train).values
x_train_list = []
for matrix in x_train:
    x_train_list.append(matrix.flatten())
x_train_flat =np.array(x_train_list)


# split it into X and y values
# x = data.drop('quality_label', axis=1, inplace=False).values
# y = pd.get_dummies(data['quality_label']).values
time_start = time.time()
# let the search begin
nas_object = MLPNAS(x_train_flat, y_train_dumm)
data = nas_object.search()

# get top n architectures (the n is defined in constants)
get_top_n_architectures(TOP_N)

# Time taken for the whole MLPNAS
time_taken = time.time() - time_start
print("time taken : ", time_taken)
