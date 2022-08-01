import datasets
from ops_TF import PoolBN, FactorizedReduce, DilConv, SepConv, StdConv
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from CKD import CKD
from codecarbon import EmissionsTracker
import tensorflow_model_optimization as tfmot
import tempfile


def train(args, model, dataset):

  train, test = dataset
  filename = "architectures/{}epochs_model_distilled".format(args.epochs)

  teacher = tf.keras.models.load_model('data/baseline.h5')

  myCKD = CKD(
      teacher,
      model,
      alreadySoftmax=False,
      optimizer=tf.keras.optimizers.Adam(),
      studentLoss=tf.keras.losses.CategoricalCrossentropy(),
    )
    
  distilled, history = myCKD.distil(
    trainData=train,
    valData=test,
    epochs=args.epochs,
    trainBatchSize=None,
    valBatchSize=None,
    history=True
    )
  
                # Compare results

  # Evaluate teacher
  _, baseline_model_accuracy = teacher.evaluate(test)
  trainableParams = np.sum([np.prod(v.get_shape()) for v in teacher.trainable_weights])
  nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in teacher.non_trainable_weights])
  totalParams = trainableParams + nonTrainableParams
  
  print("BASELINE SCORES")
  print("Accuracy = ",baseline_model_accuracy)
  print("Trainable Parameters number = ",trainableParams)
  print("Non Trainable Parameters number = ",nonTrainableParams)
  print("Total Parameters number = ",totalParams)


  # Evaluate distilled
  _, distilled_model_accuracy = distilled.evaluate(test)
  distilled.save(filename, save_format='h5')
  trainableParams = np.sum([np.prod(v.get_shape()) for v in distilled.trainable_weights])
  nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in distilled.non_trainable_weights])
  totalParams = trainableParams + nonTrainableParams
  
  print("FRUGAL MODEL SCORES")
  print("Accuracy = ",distilled_model_accuracy)
  print("Trainable Parameters number = ",trainableParams)
  print("Non Trainable Parameters number = ",nonTrainableParams)
  print("Total Parameters number = ",totalParams)    



if __name__ == "__main__":

  parser = ArgumentParser("darts")
  parser.add_argument("--batch-size", default=64, type=int)
  parser.add_argument("--epochs", default=50, type=int)
  parser.add_argument("--filename", default="layer1_epochs1_batch32_channels16", type=str)
  args = parser.parse_args()
  
  model_filename = "architectures/" + args.filename
  dataset = datasets.read_data(args.batch_size, post_KDARTS=True)
  model = tf.keras.models.load_model(model_filename)


  trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
  nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
  totalParams = trainableParams + nonTrainableParams

  print("FRUGAL MODEL BEFORE DISTILLATION")
  print("Trainable Parameters number = ",trainableParams)
  print("Non Trainable Parameters number = ",nonTrainableParams)
  print("Total Parameters number = ",totalParams)

  train(args, model, dataset)
