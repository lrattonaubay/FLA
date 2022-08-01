import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from nni.retiarii.oneshot.interface import BaseOneShotTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device

_logger = logging.getLogger(__name__)

class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]


class DartsTrainer(BaseOneShotTrainer):
    """
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
        self.train, self.valid = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency
        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)

        for _, module in self.nas_modules:
            module.to(self.device)

        self.model_optim = optimizer
        # use the same architecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.grad_clip = 5.

        self._init_dataloader()

    def _init_dataloader(self):

        """
        self.train_loader = []
        self.valid_loader = []
        x_train, y_train = self.train
        x_valid, y_valid = self.valid

        if len(x_train) == len(x_valid) :
            for i in range(len(x_train)//self.batch_size+1):
                y = (i+1)*self.batch_size
                self.train_loader.append((x_train[y-self.batch_size:y],y_train[y-self.batch_size:y]))
                self.valid_loader.append((x_valid[y-self.batch_size:y],y_valid[y-self.batch_size:y]))

                if i == len(self.train)//self.batch_size+1 :
                    self.train_loader.append((x_train[y-self.batch_size:-1],y_train[y-self.batch_size:-1]))
                    self.valid_loader.append((x_valid[y-self.batch_size:-1],y_valid[y-self.batch_size:-1]))
        """
    
        self.train_loader = torch.utils.data.DataLoader(self.train,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.workers)
        self.valid_loader = torch.utils.data.DataLoader(self.valid,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.workers)  
        

    def _train_one_epoch(self, epoch, train_teacherPreds, valid_teacherPreds):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
            val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)
            trainTeacherPred = to_device(train_teacherPreds[(step*self.batch_size) : ((step*self.batch_size) + self.batch_size)], self.device)
            validTeacherPred = to_device(valid_teacherPreds[(step*self.batch_size) : ((step*self.batch_size) + self.batch_size)], self.device)
            
            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                self._backward(val_X, val_y,validTeacherPred)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y, trainTeacherPred)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            # ORGANISATION ET AFFICHAGE DES PERFORMANCES
            metrics = self.metrics(logits, trn_y)
            metrics['loss'] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info('Epoch [%s/%s] Step [%s/%s]  %s', epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)


    def ckd_loss(self,y,studentLoss,studPreds,teacherPreds, alpha=0.3, temperature=3):
        distil = nn.KLDivLoss()
        if studPreds.size(dim=0) != teacherPreds.size(dim=0) :
            distilLoss = 0
        else: 
            distilLoss = distil(studPreds/temperature,teacherPreds/temperature)
        if studentLoss is None :
            studentLoss = self.loss(y,studPreds)

        ckdloss = ((alpha * studentLoss) + ((1 - alpha) * distilLoss))
        return ckdloss

    def _logits_and_loss(self, X, y,teacherPreds = None):

        logits = self.model(X)
        loss = self.loss(logits, y)
        ckd_loss = self.ckd_loss(y,loss,logits,teacherPreds)
    
        return logits, ckd_loss

    def _backward(self, val_X, val_y,teacherPreds):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, val_y,teacherPreds)
        loss.backward()

    def get_teacherPreds(empty):

        teacher_train_preds_path = "predictions/train.csv"
        teacher_valid_preds_path = "predictions/valid.csv"
        train_preds_from_csv = pd.read_csv(teacher_train_preds_path)
        valid_preds_from_csv = pd.read_csv(teacher_valid_preds_path)
        train_teacherPreds = torch.tensor(train_preds_from_csv.values,dtype=torch.float32)
        valid_teacherPreds = torch.tensor(valid_preds_from_csv.values,dtype=torch.float32)

        return train_teacherPreds, valid_teacherPreds

    def fit(self):
        train_teacherPreds, valid_teacherPreds = self.get_teacherPreds()
        for i in range(self.num_epochs):
            self._train_one_epoch(i,train_teacherPreds, valid_teacherPreds)

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result