"""
Reference: F. Tung and G. Mori, “Similarity-preserving knowledge distillation” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019. Available at: https://arxiv.org/pdf/1907.09682.pdf.
"""

import tensorflow as tf
import numpy as np

class SPKD():
	def __init__(
		self,
		teacher: tf.keras.Model,
		student: tf.keras.Model,
		teacherLayers: list,
		studentLayers: list,
		alreadySoftmax: bool = True,
		optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(),
		loss: tf.keras.losses = tf.keras.losses.CategoricalCrossentropy(),
		metrics = [tf.keras.metrics.CategoricalAccuracy()]

	):
		"""
		Description
		---------------
		Initialize the teacher model, the student model and their last layer index.

		Input(s)
		---------------
		teacher: A trained Keras Sequential or Functional model (Sub-class models are not supported).
		student: An untrained Keras Sequential or Functional model (Sub-class models are not supported).
		teacherLayers: List of teacher model layer's indexes where the user want to preserve the pairwise activation similarities of input samples.
		studentLayers: List of student model layer's indexes where the user want to preserve the pairwise activation similarities of input samples.
		alreadySoftmax : If the last layer is softmax it must be true, else it must be false (for teacher and student). By default true.
		optimizer: Optimizer instance. By default Adam.
		loss: Loss intance. By default CategoricalCrossentropy.
		metrics: List of metrics to be evaluated by the model during training and testing. By default CategoricalAccuracy.
		"""
		if np.array(teacherLayers).shape == np.array(studentLayers).shape:
			self.teacher = teacher
			self.student = student
			self.teacherLayers = teacherLayers
			self.studentLayers = studentLayers
			self.alreadySoftmax = alreadySoftmax
			self.optimizer = optimizer
			self.loss = loss
			self.metrics = metrics
		else:
			raise ValueError("teacherLayers and studentLayers shapes should match. Here we have teacherLayers.shape = {} and studentLayers.shape = {}".format(np.array(teacherLayers).shape, np.array(studentLayers).shape))

	def __G(
		self,
		tensor
	):
		"""
		Description
		---------------
		Transform the activation map into a (batch_size x batch_size) matrix.
		
		Input(s)
		---------------
		tensor: tf.Tensor containing the matrix of the activation map.

		Output(s)
		---------------
		output: a (batch_size x batch_size) tf.Tensor matrix.
		"""
		# Reshaping the activation map
		tensor = tf.reshape(tensor, [tf.shape(tensor).numpy()[0], np.prod(tf.shape(tensor).numpy()[1:])])
		# Transposing the tensor
		tensor_transposed = tf.transpose(tensor)
		# Getting G's approximation
		G_approx = tf.tensordot(tensor, tensor_transposed, 1)
		# Returning the G matrix
		output = tf.math.divide_no_nan(G_approx, tf.math.l2_normalize(G_approx, axis=1))
		return output

	def __localLSP(
		self,
		tensor_teacher,
		tensor_student
	):
		"""
		Description
		---------------
		Transform a (batch_size x batch_size) matrix into a number (which determines the loss between teacher and student) by applying a Frobenius norm to the substract between the teacher matrix and the student matrix.
		
		Input(s)
		---------------
		tensor_teacher: a (batch_size x batch_size) tf.Tensor matrix.
		tensor_student: a (batch_size x batch_size) tf.Tensor matrix.

		Output(s)
		---------------
		lsp: a (batch_size x batch_size) tf.Tensor matrix.
		"""
		# Applying a Frobenius norm to the substract between the teacher matrix and the student matrix
		lsp = tf.norm(tf.math.subtract(tensor_teacher, tensor_student), ord='euclidean')
		return lsp

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
				metricsTuples.append(('distillation_' + handler.name, handler.result().numpy()))
			else:
				metricsTuples.append(("val_distillation_" + handler.name, handler.result().numpy()))
		return metricsTuples

	def __compute_loss(
		self,
		x,
		y,
		training,
		teacherOutputsHandler,
		studentOutputsHandler,
		gamma
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
		teacherOutputsHandler: List with teacher handlers for intermediate results.
		studentOutputsHandler: List with student handlers for intermediate results.
		gamma: Balancing parameter.

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
		# Initializing local LSP
		localLSP = 0
		for i in range(len(teacherOutputsHandler)):
			teacherActivation = self.__G(teacherOutputsHandler[i](x, training=False))
			studentActivation = self.__G(studentOutputsHandler[i](x, training=False))
			localLSP += self.__localLSP(teacherActivation, studentActivation)
		# Computing the global LSP
		b = tf.shape(teacherActivation).numpy()[0]
		distilLoss = tf.math.divide(localLSP, tf.cast(tf.math.pow(b,2), tf.float32))
		if self.alreadySoftmax == False:
			studentLoss = self.loss(y, tf.keras.activations.softmax(studentPreds))
		else:
			studentLoss = self.loss(y, studentPreds)
		loss = studentLoss + (gamma * distilLoss.numpy())
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
		gamma: float = 2000,
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
		gamma: Balancing parameter. By default 2000.
		history: Boolean. If True, returns the losses and metrics history. If False, does not return history. By default False.

		Output(s)
		---------------
		distilled_model: Distilled Keras Sequential or Functional student model.
		lossesAndMetrics : If "history" equals True, it returns the losses and metrics history.
		"""
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
		# Getting outputs handlers
		teacherOutputs = []
		studentOutputs = []
		for i in range(len(self.teacherLayers)):
			teacherOutputModel = tf.keras.Model(inputs=self.teacher.input, outputs=self.teacher.get_layer(index=self.teacherLayers[i]).output)
			teacherOutputs.append(teacherOutputModel)
			studentOutputModel = tf.keras.Model(inputs=self.student.input, outputs=self.student.get_layer(index=self.studentLayers[i]).output)
			studentOutputs.append(studentOutputModel)
		# Training
		for epoch in range(epochs):
			print("Distillation Epoch {}/{}".format(epoch+1, epochs))
			pb = tf.keras.utils.Progbar(batchNbTrain + 1)
			for step, (x_batch_train, y_batch_train) in enumerate(trainData):
				with tf.GradientTape() as tape:
					teacherPredsTrain, studentPredsTrain, lossTrain, studentLossTrain, distilLossTrain = self.__compute_loss(
						x_batch_train,
						y_batch_train,
						True,
						teacherOutputs,
						studentOutputs,
						gamma
					)
				grads = tape.gradient(lossTrain, self.student.trainable_weights)
				self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
				# Metrics
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
				teacherPredsVal = self.teacher(x_batch_val, training=False)
				studentPredsVal = self.student(x_batch_val, training=False)
				# Computing losses and metrics
				# Computing losses
				teacherPredsVal, studentPredsVal, lossVal, studentLossVal, distilLossVal = self.__compute_loss(
						x_batch_val,
						y_batch_val,
						False,
						teacherOutputs,
						studentOutputs,
						gamma
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
				globalTuples = globalTuplesTrain + globalTuplesVal
			pb.add(1, values=globalTuples)
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
		# Returning distilled student
		if history == True:
			return self.student, lossesAndMetrics
		else:
			return self.student