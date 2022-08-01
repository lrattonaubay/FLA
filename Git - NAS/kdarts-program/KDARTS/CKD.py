"""
Reference: G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv.org, 09-Mar-2015. Available: https://arxiv.org/abs/1503.02531.
"""

import tensorflow as tf

class CKD():
	def __init__(
		self,
		teacher: tf.keras.Model,
		student: tf.keras.Model,
		alreadySoftmax: bool = True,
		optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(),
		studentLoss: tf.keras.losses = tf.keras.losses.CategoricalCrossentropy(),
		distilLoss: tf.keras.losses = tf.keras.losses.KLDivergence(),
		metrics = [tf.keras.metrics.CategoricalAccuracy()],
		#callbacks = tf.keras.callbacks.CallbackList()
	):
		"""
		Description
		---------------
		Initialize the teacher model, the student model and their last layer index.

		Input(s)
		---------------
		teacher: A trained Keras Sequential or Functional model (Sub-class models are not supported).
		student: An untrained Keras Sequential or Functional model (Sub-class models are not supported).
		alreadySoftmax : If the last layer is softmax it must be true, else it must be false (for teacher and student). By default true.
		optimizer: Optimizer instance. By default Adam.
		studentLoss: Loss instance. By default CategoricalCrossentropy.
		distilLoss: Loss instance. By default KLDivergence.
		metrics: List of metrics to be evaluated by the model during training and testing. By default CategoricalAccuracy.
		"""
		self.teacher = teacher
		self.student = student
		self.alreadySoftmax = alreadySoftmax
		self.metrics = metrics
		self.optimizer = optimizer
		self.distilLoss = distilLoss
		self.studentLoss = studentLoss
		#self.callbacks = callbacks
	
	def __metrics_after_batch(
		self,
		metricsHandlerTruth,
		metricsHandlerTeacher,
		y_truth,
		teacher_preds,
		student_preds,
		step,
		training
	):
		"""
		Description
		---------------
		Compute metrics after a mini-batch.

		Input(s)
		---------------
		metricsHandlerTruth: List with metrics instance for the ground truth labels
		metricsHandlerTeacher: List with metrics instance for the teacher labels.
		y_truth: Ground truth categories.
		teacher_preds: Teacher's predictions.
		student_preds: Student's predictions
		step: Current minbatch index.
		training: Boolean, if true training mode, if false validation mode.

		Output(s)
		---------------
		metricsTuples: List of tuples with metrics inside.
		"""
		metricsTuples = []
		# Comparing to ground truth values
		for handler in metricsHandlerTruth:
			if step == 0:
				handler.reset_state()
			handler.update_state(y_truth.numpy(), student_preds.numpy())
			if training == True:
				metricsTuples.append((handler.name, handler.result().numpy()))
			else:
				metricsTuples.append(("val_" + handler.name, handler.result().numpy()))
		# Comparing to teacher
		for handler in metricsHandlerTeacher:
			if step == 0:
				handler.reset_state()
			handler.update_state(teacher_preds.numpy(), student_preds.numpy())
			if training == True:
				metricsTuples.append((handler.name, handler.result().numpy()))
			else:
				metricsTuples.append(("val_distillation_" + handler.name, handler.result().numpy()))
		return metricsTuples
	
	def __compute_loss(
		self,
		x,
		y,
		training,
		alpha,
		temperature
	):
		"""
		Description
		---------------
		Compute predictions and losses.

		Input(s)
		---------------
		x: Image mini-batch.
		y: Categories mini-batch.
		training: Boolean, if true prediction mode of student is Training=True, else Training=False.
		alpha: Loss weighting factor.
		temperature: Temperature for softening probability distributions.

		Output(s)
		---------------
		teacherPreds: Predictions of the teacher.
		studentPreds: Predictions of the student.
		loss: Value of the loss.
		studentLoss: Value of the studentLoss.
		distilLoss: Value of the distilLoss.
		"""
		# Make predictions
		teacherPreds = self.teacher(x, training=False)
		studentPreds = self.student(x, training=training)
		if self.alreadySoftmax == False:
			distilLoss = self.distilLoss(
				tf.keras.activations.softmax(teacherPreds) / temperature,
				tf.keras.activations.softmax(studentPreds) / temperature
			)
			studentLoss = self.studentLoss(
				y,
				tf.keras.activations.softmax(studentPreds)
			)
		else:
			distilLoss = self.distilLoss(
				teacherPreds / temperature,
				studentPreds / temperature
			)
			studentLoss = self.studentLoss(
				y,
				tf.keras.activations.softmax(studentPreds)
			)
		# Computing loss
		loss = (alpha * studentLoss) + ((1 - alpha) * distilLoss)
		return teacherPreds, studentPreds, loss, studentLoss, distilLoss

	def __packing_history(
		self,
		loss,
		studentLoss,
		distilLoss,
		metricsHandlerTruth,
		metricsHandlerTeacher,
		training,
		history
	):
		"""
		Description
		---------------
		Add losses and metrics values to the history.

		Input(s)
		---------------
		loss: Value of the loss.
		studentLoss: Value of the studentLoss.
		distilLoss: Value of the distilLoss.
		metricsHandlerTruth: List with metrics instance for the ground truth labels
		metricsHandlerTeacher: List with metrics instance for the teacher labels.
		training: Boolean, if true training mode, if false validation mode.
		history: Dictionnary to fill.
		"""
		# Adding losses to history
		if training == True:
			losses = ['loss', 'distilLoss', 'studentLoss']
		else:
			losses = ['val_loss', 'val_distilLoss', 'val_studentLoss']
		for i in losses:
			if i not in history:
				history[i] = []
		history[losses[0]].append(loss.numpy())
		history[losses[1]].append(distilLoss.numpy())
		history[losses[2]].append(studentLoss.numpy())
		# Adding metrics to history
		# Comparing to ground truth
		for handler in metricsHandlerTruth:
			if training == True:
				if handler.name not in history:
					history[handler.name] = []
				history[handler.name].append(handler.result().numpy())
			else:
				if "val_" + handler.name not in history:
					history["val_" + handler.name] = []
				history["val_" + handler.name].append(handler.result().numpy())
		# Comparing to teacher
		for handler in metricsHandlerTeacher:
			if training == True:
				if "distillation_" + handler.name not in history:
					history["distillation_" + handler.name] = []
				history["distillation_" + handler.name].append(handler.result().numpy())
			else:
				if "val_distillation_" + handler.name not in history:
					history["val_distillation_" + handler.name] = []
				history["val_distillation_" + handler.name].append(handler.result().numpy())

	def distil(
		self,
		trainData: tf.data.Dataset,
		valData: tf.data.Dataset,
		epochs: int = 1,
		trainBatchSize: int = None,
		valBatchSize: int = None,
		alpha: float = 0.1,
		temperature: int = 3,
		history: bool = False
	):
		"""
		Description
		---------------
		Distil the knowledge of the teacher to the student.

		Input(s)
		---------------
		trainData: TensorFlow Dataset with training images.
		valData: TensorFlow Dataset with validation images.
		epochs: Number of epochs to distil the model. By default 1.
		trainBatchSize: Number of samples per gradient update. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.
		valBatchSize: Number of samples per validation batch. If None, we assume that the user provides a batched TensorFlow Dataset. By default None.
		alpha: Loss weighting factor. By default 0.1 (10% student's loss, 90% distillation's loss).
		temperature: Temperature for softening probability distributions. Larger temperature gives softer distributions. By default 3.
		history: Boolean. If True, returns the losses and metrics history. If False, does not return history. By default False.

		Output(s)
		---------------
		distilled_model: Distilled Keras Sequential or Functional student model.
		lossesAndMetrics : If "history" equals True, it returns the losses and metrics history.
		"""
		# Compiling student model
		self.student.compile(
			optimizer=self.optimizer,
			loss=self.studentLoss,
			metrics=self.metrics
		)
		# Prepare the training dataset
		if trainBatchSize != None:
			trainData = trainData.shuffle(1024).batch(batch_size=trainBatchSize)
		batchNbTrain = trainData.cardinality().numpy()
		# Prepare the validation dataset
		if valBatchSize != None:
			valData = valData.batch(batch_size=valBatchSize)
		batchNbVal = valData.cardinality().numpy()
		# Getting metrics
		lossesAndMetrics = {}
		metricsHandlerTruth = []
		metricsHandlerTeacher = []
		for metric in self.metrics:
			metricsHandlerTruth.append(tf.keras.metrics.get(metric))
			metricsHandlerTeacher.append(tf.keras.metrics.get(metric))

		#logs = {}
		#self.callbacks.on_train_begin(logs=logs)

		# Training
		for epoch in range(epochs):

			#self.callbacks.on_epoch_begin(epoch, logs=logs)

			print("Distillation Epoch {}/{}".format(epoch+1, epochs))
			pb = tf.keras.utils.Progbar(batchNbTrain + 1)
			for step, (x_batch_train, y_batch_train) in enumerate(trainData):

				#self.callbacks.on_batch_begin(step, logs=logs)
				#self.callbacks.on_train_batch_begin(step, logs=logs)

				with tf.GradientTape() as tape:
					# Computing distillation and student losses
					teacherPredsTrain, studentPredsTrain, lossTrain, studentLossTrain, distilLossTrain = self.__compute_loss(
						x_batch_train,
						y_batch_train,
						True,
						alpha,
						temperature
					)
				# Computing metrics
				metricsTuplesTrain = self.__metrics_after_batch(
					metricsHandlerTruth,
					metricsHandlerTeacher,
					y_batch_train,
					teacherPredsTrain,
					studentPredsTrain,
					step,
					True
				)
				# Updating progress bar losses and metrics
				lossesTuplesTrain = [
					('loss', lossTrain),
					('distilLoss', distilLossTrain),
					('studentLoss', studentLossTrain)
				]
				globalTuplesTrain = lossesTuplesTrain + metricsTuplesTrain
				pb.add(1, values=globalTuplesTrain)
				# Computing gradient
				gradients = tape.gradient(lossTrain, self.student.trainable_variables)
				# Update weights
				self.student.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

				#logs = dict(metricsTuplesTrain)

				#self.callbacks.on_train_batch_end(step, logs=logs)
				#self.callbacks.on_batch_end(step, logs=logs)

			# Adding training losses and metrics values to the model
			self.__packing_history(
				lossTrain,
				studentLossTrain,
				distilLossTrain,
				metricsHandlerTruth,
				metricsHandlerTeacher,
				True,
				lossesAndMetrics
			)

			# Validation
			for step, (x_batch_val, y_batch_val) in enumerate(valData):

				#self.callbacks.on_batch_begin(step, logs=logs)
				#self.callbacks.on_test_batch_begin(step, logs=logs)

				teacherPredsVal, studentPredsVal, lossVal, studentLossVal, distilLossVal = self.__compute_loss(
						x_batch_val,
						y_batch_val,
						False,
						alpha,
						temperature
					)
				# Updating validation losses and metrics
				lossesTuplesVal = [
					('val_loss', lossVal),
					('val_distilLoss', distilLossVal),
					('val_studentLoss', studentLossVal)
				]
				# Computing metrics
				metricsTuplesVal = self.__metrics_after_batch(
					metricsHandlerTruth,
					metricsHandlerTeacher,
					y_batch_val,
					teacherPredsVal,
					studentPredsVal,
					step,
					False
				)
				globalTuplesVal = lossesTuplesVal + metricsTuplesVal

				#logs = dict(metricsTuplesVal)

				#self.callbacks.on_test_batch_end(step, logs=logs)
				#self.callbacks.on_batch_end(step, logs=logs)

			globalTuples = globalTuplesTrain + globalTuplesVal
			pb.add(1, values=globalTuplesVal)
			# Adding validation losses and metrics values to the model
			self.__packing_history(
				lossVal,
				studentLossVal,
				distilLossVal,
				metricsHandlerTruth,
				metricsHandlerTeacher,
				False,
				lossesAndMetrics
			)

			#self.callbacks.on_epoch_end(epoch, logs=logs)

		#self.callbacks.on_train_end(logs=logs)
		
		# Returning distilled student
		if history == True:
			return self.student, lossesAndMetrics
		else:
			return self.student