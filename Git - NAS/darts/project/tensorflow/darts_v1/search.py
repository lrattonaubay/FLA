from argparse import ArgumentParser

import tensorflow as tf

import datasets
from model import CNN
from utils import accuracy, visualize
from darts import DartsTrainer

if __name__ == "__main__":

    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    args = parser.parse_args()

        # Récupérer les données du dataset
    n_channel, dataset_train, dataset_valid = datasets.get_dataset("cifar10")

        
        # construction du modèle entier + appel des fonction d'entrainement (forward)
    model = CNN(
        input_size=32, 
        in_channels=n_channel, 
        channels=args.channels, 
        n_classes=10, 
        n_layers=args.layers
        )

        # calcule la cross entropy loss entre l'entrée et la cible
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Implémente la descente de gradient stochastique
    optim = tf.keras.optimizers.SGD(learning_rate= 0.025, momentum=0.9)


    trainer = DartsTrainer(
        model=model,
        loss=criterion,
        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
        optimizer=optim,
        num_epochs=args.epochs,
        dataset=dataset_train,
        batch_size=args.batch_size,
        log_frequency=args.log_frequency,
        unrolled=args.unrolled
    )

    trainer.fit()
    final_architecture = trainer.export()
    print('Final architecture:', final_architecture)
    if args.visualization :
        visualize(final_architecture)
    #json.dump(trainer.export(), open('checkpoint.json', 'w'))
