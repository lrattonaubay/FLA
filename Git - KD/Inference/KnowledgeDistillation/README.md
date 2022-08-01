# Knowledge Distillation

In this section, the following methods would be available:
- Classic Knowledge Distillation (CKD)
- Correlation Congruence for Knowledge Distillation (CCKD)
- Similarity-Preserving Knowledge Distillation (SPKD)

# Classic Knowledge Distillation (CKD)

Reference: G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv.org, 09-Mar-2015. Available: https://arxiv.org/abs/1503.02531.

|Arguments||
|---|---|
|teacher|A trained Keras Sequential or Functional model (Sub-class models are not supported).|
|student|An untrained Keras Sequential or Functional model (Sub-class models are not supported).|
|alreadySoftmax|If the last layer is softmax it must be true, else it must be false (for teacher and student). By default true.|
|optimizer|Optimizer instance. By default Adam.|
|studentLoss|Loss instance. By default CategoricalCrossentropy.|
|distilLoss|Loss instance. By default KLDivergence.|
|metrics|List of metrics to be evaluated by the model during training and testing. By default CategoricalAccuracy.|

## Method(s)

### **distil**

Distil the knowledge of the teacher to the student.

|Arguments||
|---|---|
|trainData|TensorFlow Dataset with training images.|
|valData|TensorFlow Dataset with validation images.|
|epochs|Number of epochs to distil the model. By default 1.|
|trainBatchSize|Number of samples per gradient update. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.|
|valBatchSize|Number of samples per validation batch. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.|
|alpha|Loss balancing factor. dictates regular/distillation loss ratio. By default 0.1 (10% student's loss, 90% distillation's loss).|
|temperature|Temperature for softening probability distributions. Larger temperature gives softer distributions. By default 3.|
|history|Boolean. If True, returns the losses and metrics history. If False, does not return history. By default False.|

|Returns|
|---|
|Distilled Keras Sequential or Functional student model.|
|If "history" equals True, it returns the losses and metrics history|

# Correlation Congruence for Knowledge Distillation (CCKD)

Reference: B. Peng, X. Jin, J. Liu, S. Zhou, Y. Wu, Y. Liu, D. Li and Z. Zhang, “Correlation Congruence for Knowledge Distillation”, 2019. Available at: https://arxiv.org/pdf/1904.01802.

|Arguments||
|---|---|
|teacher|A trained Keras Sequential or Functional model (Sub-class models are not supported).|
|student|An untrained Keras Sequential or Functional model (Sub-class models are not supported).|
|alreadySoftmax|If the last layer is softmax it must be true, else it must be false (for teacher and student). By default true.|
|optimizer|Optimizer instance. By default Adam.|
|studentLoss|Loss instance. By default CategoricalCrossentropy.|
|distilLoss|Loss instance. By default KLDivergence.|
|kernel|Sklearn pairwise kernel instance, used to compute correlations. By default Radial Basis Function kernel.|
|metrics|List of metrics to be evaluated by the model during training and testing. By default CategoricalAccuracy.|

## Method(s)

### **distil**

Distil the knowledge of the teacher to the student.

|Arguments||
|---|---|
|trainData|TensorFlow Dataset with training images.|
|valData|TensorFlow Dataset with validation images.|
|epochs|Number of epochs to distil the model. By default 1.|
|trainBatchSize|Number of samples per gradient update. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.|
|valBatchSize|Number of samples per validation batch. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.|
|alpha|Loss balancing factor. dictates regular/distillation loss ratio. By default 0.1 (10% student's loss, 90% distillation's loss).|
|beta|Loss weighting factor. dictates correlation congruence importance. By default 0.5.|
|order|Order of the Taylor's series to approximate correlation matrices with. By default 2.|
|temperature|Temperature for softening probability distributions. Larger temperature gives softer distributions. By default 3.|
|history|Boolean. If True, returns the losses and metrics history. If False, does not return history. By default False.|

|Returns|
|---|
|Distilled Keras Sequential or Functional student model.|
|If "history" equals True, it returns the losses and metrics history|


# Similarity-Preserving Knowledge Distillation (SPKD)

Reference: F. Tung and G. Mori, “Similarity-preserving knowledge distillation” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019. Available at: https://arxiv.org/pdf/1907.09682.pdf.

|Arguments||
|---|---|
|teacher|A trained Keras Sequential or Functional model (Sub-class models are not supported).|
|student|An untrained Keras Sequential or Functional model (Sub-class models are not supported).|
|teacherLayers|List of teacher model layer's indexes where the user want to preserve the pairwise activation similarities of input samples.|
|studentLayers|List of student model layer's indexes where the user want to preserve the pairwise activation similarities of input samples.|
|alreadySoftmax|If the last layer is softmax it must be true, else it must be false (for teacher and student). By default true.|
|optimizer|Optimizer instance. By default Adam.|
|loss|Loss intance. By default CategoricalCrossentropy.|
|metrics|List of metrics to be evaluated by the model during training and testing. By default CategoricalAccuracy.|

## Method(s)

### **distil**

Distil the knowledge of the teacher to the student.

|Arguments||
|---|---|
|trainData|TensorFlow Dataset with training images.|
|valData|TensorFlow Dataset with validation images.|
|epochs|Number of epochs to distil the model. By default 1.|
|trainBatchSize|Number of samples per gradient update. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.|
|valBatchSize|Number of samples per validation batch. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.|
|gamma|Balancing parameter. By default 2000.|
|history|Boolean. If True, returns the losses and metrics history. If False, does not return history. By default False.|

|Returns|
|---|
|Distilled Keras Sequential or Functional student model.|
|If "history" equals True, it returns the losses and metrics history|

# Examples
## Common path

First, you have to get data:

```python
# Get data from Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Labels to categories
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)
# Cast to Float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0
# Transforming into tf.data.Dataset
train_cat_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).batch(32)
test_cat_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(32)
```
NB: It is mandatory to put the data into a tf.data.Dataset

Then, you have to define your teacher, compile it and train it:

```python
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
# Compile
teacher.compile(
	optimizer=tf.keras.optimizers.Adam(),
	loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
	metrics=[tf.keras.metrics.CategoricalAccuracy()]
)
# Train
teacher.fit(train_cat_batch, epochs=1)
```

Then, you have to define your student:

```python
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
```

## Classic Knowledge Distillation (CKD)

```python
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
```

## Correlation Congruence for Knowledge Distillation (CCKD)

```python
myCCKD = CCKD(
	teacher,
	student,
	optimizer=tf.keras.optimizers.Adam(),
	studentLoss=tf.keras.losses.CategoricalCrossentropy(),
	kernel=sklearn.metrics.pairwise.rbf_kernel,
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
	history=true
)
```

## Similarity-Preserving Knowledge Distillation (SPKD)

```python
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
```
