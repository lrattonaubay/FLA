import numpy as np
import tensorflow as tf
from model_TF import CNN_creation

def preds_exporter(dataset, nb_classes, teacher) :
	
	"""
    Description
    ---------------
    Exports teacher's predictions into .csv
    Input(s)
    ---------------
    dataset: Batched train dataset
    nb_classes: Number of labels that can be predicted
    teacher: Model with which predictions are made.
  	"""
	train, valid, test = dataset

	train_predictions = np.array([])
	valid_predictions = np.array([])
	test_predictions = np.array([])
	
	for (x, y)in train :
		batched_preds = teacher(x, training=False).numpy()
		train_predictions = np.append(train_predictions, batched_preds)
		
	for (x, y) in valid :
		batched_preds = teacher(x, training=False).numpy()
		valid_predictions = np.append(valid_predictions, batched_preds)
		
	for (x, y) in test :
		batched_preds = teacher(x, training=False).numpy()
		test_predictions = np.append(valid_predictions, batched_preds)
		
		train_predictions = np.reshape(train_predictions,(-1,nb_classes))
		valid_predictions = np.reshape(valid_predictions,(-1,nb_classes))
		test_predictions = np.reshape(test_predictions,(-1,nb_classes))

		np.savetxt('predictions/train.csv', train_predictions, delimiter =", ")
		np.savetxt('predictions/valid.csv', valid_predictions, delimiter =", ")
		np.savetxt('predictions/test.csv', test_predictions, delimiter =", ")

	teacher.summary()


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	# one-hot case
	if target.ndimension() > 1:
		target = target.max(1)[1]

	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = dict()
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
	return res



def split_normal_reduce(arc):
	"""
	Description
	---------------
	Splits the architecture given as an input into normal and reduce dictionaries

	Input(s)
	---------------
	arc: dict

	Output(s)
	---------------
	arc_normal: dict
	arc_reduce: dict
	"""
	arc_normal, arc_reduce = dict(),dict()
	for key in arc.keys():
		if "normal" in key:
			arc_normal[key]=arc[key]
		elif "reduce" in key:
			arc_reduce[key]=arc[key]
		else:
			print("Issue encountered : the following value is neither normal nor reduce", key)
	return arc_normal, arc_reduce

def convert_to_simple(arc):
	"""
	Description
	---------------
	Converts the architecture format into a simpler and more easily readable format.
	It also removes all the branches that are not used, based on the "switch" parameters

	Input(s)
	---------------
	arc: dict

	Output(s)
	---------------
	kept_arc: dict
	"""
	kept_arc_index = []
	kept_arc = dict()
	for value in arc.values():
		if type(value)==type([1,2]):
			kept_arc_index.append(value)
	prev_inc = 0
	increment = 2
	j=2
	for pair in kept_arc_index:
		keys_available = list(arc.keys())[prev_inc:increment]
		values = dict()
		for i in pair:
			values[i]= arc[keys_available[i]]
		kept_arc[j] = values
		prev_inc= increment
		increment += j+1
		j+=1
	return kept_arc

def split_prep(arc):
	arc_norm, arc_red = split_normal_reduce(arc)
	arc_norm = convert_to_simple(arc_norm)
	arc_red = convert_to_simple(arc_red)
	return arc_norm, arc_red


def convert_to_tf(dict_normal, dict_reduce, layers, channels, filename=None):
	optimizer = tf.keras.optimizers.SGD()
	loss = tf.keras.losses.categorical_crossentropy

	model = CNN_creation(channels, 7, layers, dict_normal, dict_reduce, input_shape=[48,48,1])
	model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])
	model.summary()

	if filename :
		model.save(filename, save_format='h5')