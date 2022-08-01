import tensorflow as tf

def get_dataset(cls):

    if cls == "cifar10":
        layer = tf.keras.layers.Normalization(mean=0.49139968, variance=0.24703233)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return 3, (layer(x_train), y_train), (layer(x_test), y_test)

    elif cls == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return 1, (x_train/255.0, y_train), (x_test/255.0, y_test)

    else:
        print("Dataset Error ------- " + cls + " is unknown !")
        return 0