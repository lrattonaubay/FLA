from collections import OrderedDict
import tensorflow as tf
from darts_utils import replace_layer_choice, replace_input_choice
from datetime import datetime 
import abc
from typing import Any


class DartsLayerChoice(tf.keras.layers.Layer):

    def __init__(self, layer_choice):

        super().__init__()

        self.op_choices = OrderedDict([(name, layer_choice[name]) for name in layer_choice.names])
        size_normal = []
        size_normal.append(len(self.op_choices))
        self.alpha = tf.random.normal(size_normal) * 1e-3


    def call(self, *args, **kwargs):
        op_results = tf.keras.layers.Concatenate([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return tf.sum(op_results * tf.keras.layers.Softmax(self.alpha, -1).reshape(*alpha_shape), 0)
   
    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[tf.math.argmax(self.alpha)]   


class DartsInputChoice(tf.keras.layers.Layer):
    def __init__(self, input_choice):

        super().__init__()

        size_normal = []
        size_normal.append(input_choice.n_candidates)
        self.alpha = tf.convert_to_tensor(tf.random.normal(size_normal) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1
        

    def call(self, inputs):
        inputs =  tf.keras.layers.Concatenate(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return tf.sum(inputs * tf.keras.layers.Softmax(self.alpha, -1).reshape(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return tf.argsort(-self.alpha).numpy().tolist()[:self.n_chosen]    
          
# Param BaseOneShotTrainer
class BaseOneShotTrainer(abc.ABC):
    """
    Build many (possibly all) architectures into a full graph, search (with train) and export the best.

    One-shot trainer has a ``fit`` function with no return value. Trainers should fit and search for the best architecture.
    Currently, all the inputs of trainer needs to be manually set before fit (including the search space, data loader
    to use training epochs, and etc.).

    It has an extra ``export`` function that exports an object representing the final searched architecture.
    """

    @abc.abstractmethod
    def fit(self) -> None:
        pass

    @abc.abstractmethod
    def export(self) -> Any:
        pass

class DartsTrainer(BaseOneShotTrainer):
 
    """""   
    DARTS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    learning_rate : float
        Learning rate to optimize the model.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    unrolled : float
        ``True`` if using second order optimization, else first order optimization.
    """
    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip=5.,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.log_frequency = log_frequency
        
        self.nas_modules = []
        # Modifier utils
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)

        self.model_optim = optimizer
        # use the same architecture weight for modules with duplicated names

        self.ctrl_optim = tf.keras.optimizers.Adam(learning_rate=arc_learning_rate, beta_1=0.5, beta_2=0.999, clipvalue=3.)
        self.grad_clip = grad_clip


        self._init_dataloader()

    def _init_dataloader(self):

        # Split the dataset for training and validation.
        x, y = self.dataset
        split = len(x) // 2
        x_train = x[:split]
        x_train = tf.cast(x_train, float)
        y_train = y[:split]
        y_train = tf.cast(y_train, float)
        x_val = x[split:]
        x_val = tf.cast(x_val, float)
        y_val = y[split:]
        y_val = tf.cast(y_val, float)

        # Prepare the training dataset with batch_size
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = self.train_dataset.batch(self.batch_size)

        # Prepare the validation dataset with batch_size
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_dataset = self.val_dataset.batch(self.batch_size)
    
    def _train_one_epoch(self, epoch, num_epochs, start_time):

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_dataset, self.val_dataset)):

            # phase 1. architecture step
            with tf.GradientTape() as tape:
                logits = self.model(val_X, training=True)
                loss_value = self.loss(val_y, logits)
            
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.ctrl_optim.apply_gradients((grad, var) for (grad, var) in zip(grads, self.model.trainable_variables) if grad is not None)
            # phase 2: child network step
            with tf.GradientTape() as tape2:

                logits = self.model(trn_X, training=True)
                loss_value = self.loss(trn_y, logits)
            
            grads = tape2.gradient(loss_value, self.model.trainable_variables)
            self.model_optim.apply_gradients((grad, var) for grad, var in zip(grads, self.model.trainable_variables) if grad is not None)

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(val_y, self.model(val_X, training=True))

            if self.log_frequency is not None and step % self.log_frequency == 0:
                print("( {}".format(datetime.now() - start_time), ")  Epoch [%d/%d] - Step [%d/%d]   Loss = %.4f"% (epoch+1, num_epochs, step, len(self.train_dataset)-1, float(loss_value)))
        
        print("Epoch {}: Loss: {:.4f}, Accuracy: {:.4%}".format(epoch+1, epoch_loss_avg.result(), epoch_accuracy.result()))

    def fit(self):
        start_time = datetime.now()
        for i in range(self.num_epochs):
            self._train_one_epoch(i, self.num_epochs, start_time)

   
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
