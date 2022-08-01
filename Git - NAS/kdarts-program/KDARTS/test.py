import datasets
import tensorflow as tf
import numpy as np

def evaluation(model, dataset):
    _, test = dataset
    score = model.evaluate(test)

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams

    print("Loss = ",score[0])
    print("Accuracy = ",score[1])
    print("Trainable Parameters number = ",trainableParams)
    print("Non Trainable Parameters number = ",nonTrainableParams)
    print("Total Parameters number = ",totalParams)


if __name__ == "__main__":
  
    dataset = datasets.read_data(32, post_KDARTS=True)
    model = tf.keras.models.load_model("architectures/30epochs_model_distilled")
    teacher = tf.keras.models.load_model('data/baseline.h5')

    print("============ TEACHER ============")
    evaluation(teacher, dataset)
    print("============ STUDENT ============")
    evaluation(model, dataset)

  
