from cgi import test
import tensorflow as tf
from CKD import CKD
from SPKD import SPKD
from CCKD import CCKD

def test_ckd():
	imShape = (32, 32, 3)
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
	y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	train_cat_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).batch(32)
	test_cat_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(32)
	# Create the teacher
	teacher = tf.keras.Sequential(
		[
			tf.keras.Input(shape=imShape),
			tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.LeakyReLU(alpha=0.2),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10)
		],
		name="teacher",
	)
	# Train the teacher
	teacher.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.CategoricalAccuracy()]
	)
	teacher.fit(train_cat_batch, epochs=3)
	# Create the student
	student = tf.keras.Sequential(
		[
			tf.keras.Input(shape=imShape),
			tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.LeakyReLU(alpha=0.2),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10)
		],
		name="student",
	)
	myCKD = CKD(
		teacher,
		student,
		alreadySoftmax=False,
		optimizer=tf.keras.optimizers.Adam(),
		studentLoss=tf.keras.losses.CategoricalCrossentropy()
	)
	distilled, history = myCKD.distil(
		trainData=train_cat_batch,
		valData=test_cat_batch,
		epochs=3,
		trainBatchSize=None,
		valBatchSize=None,
		history=True
	)
	print(history)
	print("\nTest")
	teacher.evaluate(test_cat_batch)
	distilled.evaluate(test_cat_batch)

def test_spkd():
	imShape = (32, 32, 3)
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
	y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	train_cat_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).batch(32)
	test_cat_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(32)
	# Create the teacher
	teacher = tf.keras.Sequential(
		[
			tf.keras.Input(shape=imShape),
			tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.LeakyReLU(alpha=0.2),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10)
		],
		name="teacher",
	)
	# Train the teacher
	teacher.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.CategoricalAccuracy(), 'accuracy'],
	)
	teacher.fit(train_cat_batch, epochs=3)
	# Create the student
	student = tf.keras.Sequential(
		[
			tf.keras.Input(shape=imShape),
			tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.LeakyReLU(alpha=0.2),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10)
		],
		name="student",
	)
	mySPKD = SPKD(
		teacher=teacher,
		student=student,
		teacherLayers=[2],
		studentLayers=[2],
		alreadySoftmax=False
	)
	distilled, history = mySPKD.distil(
		trainData=train_cat_batch,
		valData=test_cat_batch,
		epochs=3,
		trainBatchSize=None,
		valBatchSize=None,
		gamma=500,
		history=True
	)


def test_cckd():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
	y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	train_cat_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).batch(64)
	test_cat_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(64)
	# Create the teacher
	teacher = tf.keras.Sequential(
		[
			tf.keras.Input(shape=(28, 28, 1)),
			tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.LeakyReLU(alpha=0.2),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10)
		],
		name="teacher",
	)
	# Train the teacher
	teacher.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.CategoricalAccuracy()]
	)
	teacher.fit(train_cat_batch, epochs=1)
	# Create the student
	student = tf.keras.Sequential(
		[
			tf.keras.Input(shape=(28, 28, 1)),
			tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.LeakyReLU(alpha=0.2),
			tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
			tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(10)
		],
		name="student",
	)
	myCCKD = CCKD(
		teacher,
		student,
		alreadySoftmax=False,
		optimizer=tf.keras.optimizers.Adam(),
		studentLoss=tf.keras.losses.CategoricalCrossentropy(),
	)
	distilled, history = myCCKD.distil(
		trainData=train_cat_batch,
		valData=test_cat_batch,
        alpha= 0.1,
        temperature= 5,
        beta=1,
		epochs=3,
		trainBatchSize=None,
		valBatchSize=None,
		history=True
	)
	print(history)
	print("\nTest")
	teacher.evaluate(test_cat_batch)
	distilled.evaluate(test_cat_batch)


if __name__ == "__main__":
	test_spkd()