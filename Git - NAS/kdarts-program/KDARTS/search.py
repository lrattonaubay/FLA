import json
import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn

import tensorflow as tf
import pandas as pd
import numpy as np

import datasets
from model import CNN
from utils import accuracy, preds_exporter, split_prep, convert_to_tf
from visualize import visualize
from darts_fused import DartsTrainer


logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--predictions", default=False, action="store_true")
    parser.add_argument("--export", default=False, action="store_true")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

    
        # Get dataset and teacher
    (dataset_tf), (train_pt, test_pt) = datasets.read_data(args.batch_size)
    teacher = tf.keras.models.load_model('data/baseline.h5')


    if args.predictions == True :
        
        # save the teacher's predictions into different files, show the teacher architecture and print its scores
        preds_exporter(dataset=dataset_tf, nb_classes=7, teacher=teacher)


        # build the model
    model = CNN(
        input_size=32, 
        in_channels=1, 
        channels=args.channels, 
        n_classes=7, 
        n_layers=args.layers
        )
        
        # init some parameters
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

        # prepare the model to training
    trainer = DartsTrainer(
        model=model,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
        optimizer=optim,
        num_epochs=args.epochs,
        dataset=train_pt,
        batch_size=args.batch_size,
        log_frequency=args.log_frequency,
        unrolled=args.unrolled
    )

        # Architecture search
    trainer.fit()
    final_architecture = trainer.export()
    json.dump(trainer.export(), open('checkpoint.json', 'w'))
    if args.visualization :
        visualize(final_architecture)
    
        # Convert the model to TesorFlow and export
    if args.export == True :
        filename = "architectures/layer{}_epochs{}_batch{}_channels{}".format(args.layers, args.epochs, args.batch_size, args.channels)
        dict_normal, dict_reduce = split_prep(final_architecture)
        convert_to_tf(dict_normal=dict_normal, dict_reduce=dict_reduce, layers=args.layers, channels=args.channels, filename=filename)
    