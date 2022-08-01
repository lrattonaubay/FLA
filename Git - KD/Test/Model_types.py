import tensorflow as tf

image_shape = (32,32,3)

def functional_model(nb_classes: int):
  input = tf.keras.Input(shape=image_shape)
  # First Branch
  b1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='valid')(input)
  b1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(b1)
  # Second Branch
  b2 = tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='valid')(input)
  b2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid')(b2)
  # Common neuron network
  x = tf.keras.layers.Add()([b1,b2])
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(units=256)(x)
  x = tf.keras.layers.Dense(units=128)(x)
  x = tf.keras.layers.Dense(units=nb_classes)(x)
  # Returning model
  return tf.keras.Model(inputs=input, outputs=x)

def sequential_model(nb_classes: int):
  # Defining model
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=image_shape))
  model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='valid'))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(units=256))
  model.add(tf.keras.layers.Dense(units=128))
  model.add(tf.keras.layers.Dense(units=nb_classes))
  # Returning model
  return model

if __name__ == "__main__":
    print("oui")