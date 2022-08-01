from gc import callbacks
import tensorflow as tf
from sklearn.model_selection import train_test_split
from SPKD import SPKD
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import time
import csv  


def create_model_student_teacher(
	inputShape
):
	#Create the teacher
	teacher = tf.keras.Sequential(
		[
			tf.keras.Input(shape=imShape),
			# Block 1
			tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			tf.keras.layers.Dropout(0.3),
			# Block 2
			tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			tf.keras.layers.Dropout(0.3),
			# Block 3
			tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			tf.keras.layers.Dropout(0.5),
			# Block 4
			tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			tf.keras.layers.Dropout(0.5),
			# Block 5
			tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			tf.keras.layers.Dropout(0.5),
			# Block 6
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(512, activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(0.5),
			# Block 7
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(256, activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(0.5),
			# Block 8
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
		],
		name="Teacher_custom"
	)
	# Create the student
	student = tf.keras.Sequential(
		[
			tf.keras.Input(shape=(32,32,3)),
			# Block 1
			tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
			tf.keras.layers.Dropout(0.3),

			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(10, activation='softmax')
		],
		name="Student_custom",
	)
	return student, teacher

def get_data_cifar10(
	batch_size
):
	# Getting images
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)
	y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
	y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_val = x_val.astype('float32')
	x_train = x_train / 255.0
	x_test = x_test / 255.0
	x_val = x_val / 255.0
	train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
	test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
	val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
	return train, test, val

def plot_performances(
	teacherHistory,
	distilledHistory,
	studentHistory,
	pathToSave,
	name,
	teacherName = '',
	studentName = ''
):
	# Creating folders if not already created
	pathToSave = os.path.normcase(pathToSave)
	plt.figure(figsize=(10, 3))
	if os.path.isdir(pathToSave) == False:
		os.mkdir(pathToSave)
	thisPerfPath = os.path.join(pathToSave, os.path.normcase(name + "/"))
	if os.path.isdir(thisPerfPath) == False:
		os.mkdir(thisPerfPath)
	# Getting common keys
	common = []
	for tkey in teacherHistory.keys():
		for dkey in distilledHistory.keys():
			for skey in studentHistory.keys():
				if (tkey.lower() == dkey.lower() == skey.lower()) and "val" not in tkey.lower():
					common.append(tkey)
	# Plot each common key
	for key in common:
		if teacherName and studentName != '':
			plt.title(key + " per epoch - " + name + " (" + str(teacherName) + "/" + str(studentName) + ")")
		else:
			plt.title(key + " - " + name)
		dim=np.arange(1,len(teacherHistory[key]) + 1,1)
		plt.xticks(dim, labels='')
		plt.plot(dim, teacherHistory["val_" + key], color='blue')
		plt.plot(dim, distilledHistory["val_" + key], color='red')
		plt.plot(dim, studentHistory["val_" + key], color='green')
		plt.plot(dim, teacherHistory[key], color='blue', ls='dotted')
		plt.plot(dim, distilledHistory[key], color='red', ls='dotted')
		plt.plot(dim, studentHistory[key], color='green', ls='dotted')
		# Legend
		trainingShape = Line2D([0], [0], ls='dotted', color='black')
		validationShape = Line2D([0], [0], color='black')
		teacherColor = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12)
		distilledColor = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12)
		notDistilledColor = Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12)
		plt.legend(
			handles=[trainingShape, validationShape, teacherColor, distilledColor, notDistilledColor],
			labels=['Training', 'Validation','Teacher', 'Distilled', 'Not distilled'],
			bbox_to_anchor=[1.05, 0],
			borderaxespad=0,
			loc='lower left'
		)
		plt.grid()
		plt.subplots_adjust(right=0.7)
		plt.ylabel(key.lower())
		imPath = os.path.join(thisPerfPath,  key.lower() + ".png")
		plt.savefig(imPath)
		plt.clf()

def performances(
	imgSavePath,
	teacher,
	student,
	teacherFocus,
	studentFocus,
	teacherEpochs,
	distilEpochs,
	batch_size,
	gamma, 
	thistory,
	shistory, 
	time_exec_student,
	train,
	test,
	val
):
	# Distilling
	mySPKD = SPKD(
		teacher=teacher,
		student=student,
		teacherLayers=teacherFocus,
		studentLayers=studentFocus,
		alreadySoftmax=True
	)
	begin = time.time()
	distilled, history = mySPKD.distil(
		trainData=train,
		valData=val,
		epochs=distilEpochs,
		trainBatchSize=None,
		valBatchSize=None,
		gamma=gamma,
		history=True
	)
	distilled.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.CategoricalCrossentropy(),
		metrics=[tf.keras.metrics.CategoricalAccuracy()]
	)
	distilled.evaluate(test)
	time_exec_distillation = time.time() - begin
	data  = [batch_size, gamma, time_exec_student, time_exec_distillation]
	if os.path.isfile(imgSavePath +'/summary_stats_run.csv'):
		with open(imgSavePath +'/summary_stats_run.csv', 'a', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(data)
			f.close()
	else:
		with open(imgSavePath + '/summary_stats_run.csv', 'w', encoding='UTF8', newline='') as f:
			header = ['batch_size', 'gamma', 'time_exec_student_not_distilled', 'time_exec_student_distilled']
			writer = csv.writer(f)
			writer.writerow(header)
			writer.writerow(data)
			f.close()
	stepName = "b" + str(batch_size) + "_g" + str(gamma)
	plot_performances(
		teacherHistory=thistory.history,
		distilledHistory=history,
		studentHistory=shistory.history,
		pathToSave=imgSavePath,
		name=stepName,
		teacherName=teacher.name,
		studentName=student.name
	)

if __name__ == "__main__":
	imShape = (32, 32, 3)
	# To modify
	myPathToSaveImg = "../performances/" # existing folder where you want to save plots
	batch_range = [64]
	epochs = 100
	gammas = [0]
	for i in batch_range:
		train, test, val = get_data_cifar10(i)
		student, teacher = create_model_student_teacher(imShape)
		print('\n TRAINING Teacher & Student, batch_size={}\n'.format(i))
		# Compiling and training teacher
		teacher.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
			metrics=[tf.keras.metrics.CategoricalAccuracy()]
		)
		thistory = teacher.fit(train, epochs=epochs, validation_data = val)
		teacher.evaluate(test)
		# Compiling and training a clone of the student from scratch
		student_fs = tf.keras.models.clone_model(student)
		student_fs.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
			metrics=[tf.keras.metrics.CategoricalAccuracy()]
		)
		begin = time.time()
		shistory = student_fs.fit(train, epochs=epochs, validation_data = val)
		time_exec_student = time.time() - begin
		student_fs.evaluate(test)
		for j in gammas:
			print("\nEVALUATING batch_size={}, gamma={}\n".format(i,j))
			performances(
				imgSavePath=myPathToSaveImg,
				teacher=tf.keras.models.clone_model(teacher),
				student=tf.keras.models.clone_model(student),
				teacherFocus=[26],
				studentFocus=[2],
				gamma=j,
				teacherEpochs=epochs,
				distilEpochs=epochs,
				batch_size=i,
				thistory = thistory,
				shistory = shistory,
				time_exec_student = time_exec_student,
				train=train,
				test=test,
				val=val
			)