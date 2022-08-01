"""
Reference: B. Peng, X. Jin, J. Liu, S. Zhou, Y. Wu, Y. Liu, D. Li and Z. Zhang, “Correlation Congruence for Knowledge Distillation”, 2019. Available at: https://arxiv.org/pdf/1904.01802.pdf.
"""

import tensorflow as tf
import numpy as np
import sklearn.metrics.pairwise

class CCKD():
	def __init__(
		self,
		teacher: tf.keras.Model,
		student: tf.keras.Model,
		alreadySoftmax: bool = True,
		optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(),
		studentLoss: tf.keras.losses = tf.keras.losses.CategoricalCrossentropy(),
		distilLoss: tf.keras.losses = tf.keras.losses.KLDivergence(),
		kernel: sklearn.metrics.pairwise = sklearn.metrics.pairwise.rbf_kernel,
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
		alreadySoftmax : If the last layer is softmax it must be true, else it must be false (for teacher and student). By default true.
		optimizer: Optimizer instance. By default Adam.
		studentLoss: Loss instance. By default CategoricalCrossentropy.
		distilLoss: Loss instance. By default KLDivergence.
		kernel: Kernel instance used to compute correlations. By default Radial Basis Function kernel.
		metrics: List of metrics to be evaluated by the model during training and testing. By default CategoricalAccuracy.
		"""

		self.teacher = teacher
		self.student = student
		self.alreadySoftmax = alreadySoftmax
		self.optimizer = optimizer
		self.studentLoss = studentLoss
		self.distilLoss = distilLoss
		self.kernel = kernel
		self.metrics = metrics

	def __taylor_approximation(
		self,
		corr,
		order = 1
	):
		"""
		Description
		---------------
		Approximate a correlation matrix using Taylor's series approximation.

		Input(s)
		---------------
		corr: the correlation matrix to approximate.
		order: the order of the Taylor's series to approximate with.

		Output(s)
		---------------
		y: 1D numpy array containing the row-wise approximate values.
		"""
		y = np.zeros(len(corr))
		for n, row in zip(range(order), corr):
			y = y + ((-1)**n * (row)**(2*n+1)) / np.math.factorial(2*n+1)
		return y

	def __compute_corr_congruence(
		self,
		teacherPreds,
		studentPreds,
		kernel,
		order = 2
	):
		"""
		Description
		---------------
		Compute the correlation congruence between the teacher's and student's predictions.

		Input(s)
		---------------
		teacherPreds: Teacher's predictions.
		studentPreds: Student's predictions.
		kernel: Kernel used to compute the correlation matrices.
		oder: Order of the Taylor's series to approximate correlation matrices with

		Output(s)
		---------------
		congruence: Correlation congruence, float value.
		"""
		corr_student = kernel(studentPreds)
		corr_teacher = kernel(teacherPreds)
		taylored_student = self.__taylor_approximation(corr_student, order=order)
		taylored_teacher = self.__taylor_approximation(corr_teacher, order=order)
		congruence = sum(abs(taylored_teacher - taylored_student))
		return congruence

	
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
				metricsTuples.append(("Val_" + handler.name, handler.result().numpy()))
		# Comparing to teacher
		for handler in metricsHandlerTeacher:
			if step == 0:
				handler.reset_state()
			handler.update_state(teacher_preds.numpy(), student_preds.numpy())
			if training == True:
				metricsTuples.append((handler.name, handler.result().numpy()))
			else:
				metricsTuples.append(("Val_" + handler.name, handler.result().numpy()))
		return metricsTuples
	
	def __compute_loss(
		self,
		x,
		y,
		training,
		alpha,
		beta,
		kernel,
		order,
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
		alpha: Loss balancing factor, dictates regular/distillation loss ratio.
		beta: Loss weigthing factor, dictates correlation congruence importance.
		kernel: Kernel used to compute the correlation matrices.
		order: Order of the Taylor's series to approximate correlation matrices with.
		temperature: Temperature for softening probability distributions. Larger temperature gives softer distributions.
		

		Output(s)
		---------------
		teacherPreds: Predictions of the teacher.
		studentPreds: Predictions of the student.
		loss: Value of the loss.
		studentLoss: Value of the studentLoss.
		distilLoss: Value of the distilLoss.
		congruence: Value of the correlation congruence.
		"""
		# Make predictions
		teacherPreds = self.teacher(x, training=False)
		studentPreds = self.student(x, training=training)
		if self.alreadySoftmax == False:
			teacherPreds = tf.keras.activations.softmax(teacherPreds)
			studentPreds = tf.keras.activations.softmax(studentPreds)
		distilLoss = self.distilLoss(
			teacherPreds / temperature,
			studentPreds / temperature
		)
		studentLoss = self.studentLoss(
			y,
			studentPreds
		)
		congruence = self.__compute_corr_congruence(
			teacherPreds,
			studentPreds,
			kernel = kernel,
			order = order
			)
		# Computing loss
		loss = (alpha * studentLoss) + ((1 - alpha) * distilLoss) + (beta * congruence)
		return teacherPreds, studentPreds, loss, studentLoss, distilLoss, congruence


	def __packing_history(
		self,
		loss,
		studentLoss,
		distilLoss,
		congruence,
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
		congruence: Value of the correlation congruence.
		metricsHandlerTruth: List with metrics instance for the ground truth labels
		metricsHandlerTeacher: List with metrics instance for the teacher labels.
		training: Boolean, if true training mode, if false validation mode.
		history: Dictionnary to fill.
		"""
		# Adding losses to history
		if training == True:
			losses = ['Loss', 'DistilLoss', 'StudentLoss', 'CorrCongruence']
		else:
			losses = ['Val_loss', 'Val_distilLoss', 'Val_studentLoss', 'Val_corrCongruence']
		for i in losses:
			if i not in history:
				history[i] = []
		history[losses[0]].append(loss.numpy())
		history[losses[1]].append(distilLoss.numpy())
		history[losses[2]].append(studentLoss.numpy())
		history[losses[3]].append(congruence)
		# Adding metrics to history
		# Comparing to ground truth
		for handler in metricsHandlerTruth:
			if training == True:
				if handler.name not in history:
					history[handler.name] = []
				history[handler.name].append(handler.result().numpy())
			else:
				if "Val_" + handler.name not in history:
					history["Val_" + handler.name] = []
				history["Val_" + handler.name].append(handler.result().numpy())
		# Comparing to teacher
		for handler in metricsHandlerTeacher:
			if training == True:
				if "Distillation_" + handler.name not in history:
					history["Distillation_" + handler.name] = []
				history["Distillation_" + handler.name].append(handler.result().numpy())
			else:
				if "Val_distillation_" + handler.name not in history:
					history["Val_distillation_" + handler.name] = []
				history["Val_distillation_" + handler.name].append(handler.result().numpy())
	
	def distil(
		self,
		trainData: tf.data.Dataset,
		valData: tf.data.Dataset,
		epochs: int = 1,
		trainBatchSize: int = None,
		valBatchSize: int = None,
		alpha: float = 0.1,
		beta: float = 0.5,
		order: int = 2,
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
		alpha: Loss balancing factor, dictates regular/distillation loss ratio. By default 0.1.
		beta: Loss weigthing factor, dictates correlation congruence importance. By default 0.5
		order: Order of the Taylor's series to approximate correlation matrices with. By default 2.
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
		# Training
		for epoch in range(epochs):
			print("Distillation Epoch {}/{}".format(epoch+1, epochs))
			pb_train = tf.keras.utils.Progbar(batchNbTrain)
			for step, (x_batch_train, y_batch_train) in enumerate(trainData):
				with tf.GradientTape() as tape:
					teacherPredsTrain, studentPredsTrain, lossTrain, studentLossTrain, distilLossTrain, corrCongruenceTrain = self.__compute_loss(
						x_batch_train,
						y_batch_train,
						True,
						alpha,
						beta,
						self.kernel,
						order,
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
					('Loss', lossTrain),
					('DistilLoss', distilLossTrain),
					('StudentLoss', studentLossTrain),
					('CorrCongruence',corrCongruenceTrain)
				]
				globalTuplesTrain = lossesTuplesTrain + metricsTuplesTrain
				pb_train.add(1, values=globalTuplesTrain)
				# Computing gradient
				gradients = tape.gradient(lossTrain, self.student.trainable_variables)
				# Update weights
				self.student.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
			# Adding training losses and metrics values to the model
			self.__packing_history(
				lossTrain,
				studentLossTrain,
				distilLossTrain,
				corrCongruenceTrain,
				metricsHandlerTruth,
				metricsHandlerTeacher,
				True,
				lossesAndMetrics
			)
			# Validation
			pb_val = tf.keras.utils.Progbar(batchNbVal)
			for step, (x_batch_val, y_batch_val) in enumerate(valData):
				teacherPredsVal = self.teacher(x_batch_val, training=False)
				studentPredsVal = self.student(x_batch_val, training=False)
				# Computing losses and metrics
				# Computing losses
				teacherPredsVal, studentPredsVal, lossVal, studentLossVal, distilLossVal, corrCongruenceVal = self.__compute_loss(
						x_batch_val,
						y_batch_val,
						True,
						alpha,
						beta,
						self.kernel,
						order,
						temperature
					)
				# Updating validation losses and metrics
				lossesTuplesVal = [
					('Val_loss', lossVal),
					('Val_distilLoss', distilLossVal),
					('Val_studentLoss', studentLossVal),
					('Val_corrCongruence', corrCongruenceVal)
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
				pb_val.add(1, values=globalTuplesVal)
			# Adding validation losses and metrics values to the model
			self.__packing_history(
				lossVal,
				studentLossVal,
				distilLossVal,
				corrCongruenceVal,
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
