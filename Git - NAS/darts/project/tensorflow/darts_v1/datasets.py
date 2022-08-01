import tensorflow as tf




def get_dataset(cls):

    if cls == "cifar10":
        (x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
        x_train , x_test = x_train/255.0 , x_test/255.0
        y_train , x_test = y_train/255.0 , y_test/255.0
        dataset_train, dataset_valid = (x_train, y_train), (x_test, y_test) 

        return 3, dataset_train, dataset_valid

    elif cls == "mnist":
        (x_train,y_train) , (x_test,y_test) = tf.keras.datasets.mnist.load_data()
        x_train , x_test = x_train/255.0 , x_test/255.0
        y_train , x_test = y_train/255.0 , y_test/255.0
        dataset_train, dataset_valid = (x_train, y_train), (x_test, y_test) 
        return 1, dataset_train, dataset_valid

    else:
        print("Dataset Error ------- " + cls + " is unknown !")
        return 0
