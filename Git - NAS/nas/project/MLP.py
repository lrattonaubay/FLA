import os
import warnings
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

from params import *


class MLPSearchSpace(object):

    def __init__(self, target_classes):

        """
        Description
        ---------------
        Generates the MLPSearchSpace object, using the number of target classes
        (the number of end results possible ).
        Input(s)
        ---------------
        target_classes: int
        """
        self.target_classes = target_classes
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
        """
        Description
        ---------------
        Generates a dictionary containing all possible nodes
        (number of nodes and activation function),
        using an id as a key.
        Output(s)
        ---------------
        vocab: dictionary
        """
        # Output :
            # the vocab file (a dict with ids as key,
            # and a combination of number of nodes and activation function as values)
            
    	# define the allowed nodes and activation functions
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        
        # initialize lists for keys and values of the vocabulary
        layer_params = []
        layer_id = []
        
        # for all activation functions for each node
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                # create a configuration tuple (node, activation)
                #q OUTPUT FORMAT : [(8, 'sigmoid'), (8, 'tanh'), (8, 'relu'), (8, 'elu'), (16, 'sigmoid'), ..., (512, 'elu')]
                layer_params.append((nodes[i], act_funcs[j]))   

                # create an id for each tuple
                # OUTUT FORMAT : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,...,28]
                layer_id.append(len(act_funcs) * i + j + 1)
        
        # zip the id and configurations into a dictionary 
        vocab = dict(zip(layer_id, layer_params))
        
        # add dropout in the vocabulary
        # OUTPUT FORMAT : {1: (8, 'sigmoid'), ..., 28: (512, 'elu'), 29: 'dropout'}
        vocab[len(vocab) + 1] = (('dropout'))
        
        # add the final softmax/sigmoid layer in the vocabulary
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        return vocab


    def encode_sequence(self, sequence):
        """
        Description
        ---------------
        Converts a list of nodes into the equivalent list of ids.
        Input(s)
        ---------------
        sequence: list of nodes
        Output(s)
        ---------------
        encoded_sequence: list of ids
        """
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        
        return encoded_sequence

    def decode_sequence(self, sequence):
        """
        Description
        ---------------
        Converts a list of ids into the equivalent list of nodes.
        Input(s)
        ---------------
        sequence: list of ids
        Output(s)
        ---------------
        decoded_sequence: list of nodes
        """
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence


class MLPGenerator(MLPSearchSpace):

    def __init__(self):
        """
        Description
        ---------------
        Generates the MLPGenerator object using variables predefined.
        """
        # the parameters of the MLP are prepared here
        self.target_classes = TARGET_CLASSES
        self.mlp_optimizer = MLP_OPTIMIZER
        self.mlp_lr = MLP_LEARNING_RATE
        self.mlp_decay = MLP_DECAY
        self.mlp_momentum = MLP_MOMENTUM
        self.mlp_dropout = MLP_DROPOUT
        self.mlp_loss_func = MLP_LOSS_FUNCTION
        self.mlp_one_shot = MLP_ONE_SHOT
        self.metrics = ['accuracy']

        super().__init__(TARGET_CLASSES)
        # This is used so that if the model already came up, it changes,
        # in order to test a new model.
        if self.mlp_one_shot:
        	
            # path to shared weights file 
            self.weights_file = './LOGS/shared_weights.pkl'
            
            # open an empty dataframe with columns for bigrams IDs and weights
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
            
            # pickle the dataframe
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)
    
    # function to create a keras model given a sequence and input data shape
    def create_model(self, sequence, mlp_input_shape):
        """
        Description
        ---------------
        Creates a keras model given a sequence and input data shape.
        Input(s)
        ---------------
        sequence: list of ids
        mlp_input_shape: list of ints
        Output(s)
        ---------------
        model: Sequential model
        """
        
        # decode sequence to get nodes and activations of each layer
        layer_configs = self.decode_sequence(sequence)

        # create a sequential model
        model = Sequential()

        # add a flatten layer if the input is 3 or higher dimensional
        if len(mlp_input_shape) > 1:
            model.add(Flatten(name='flatten', input_shape=mlp_input_shape))

            # for each element in the decoded sequence
            for i, layer_conf in enumerate(layer_configs):

                # add a model layer (Dense or Dropout)
                if layer_conf == 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))

        else:
            # for 2D inputs
            for i, layer_conf in enumerate(layer_configs):

                # add the first layer (requires the input shape parameter)
                if i == 0:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))

                # add subsequent layers (Dense or Dropout)
                elif layer_conf == 'dropout':
                    model.add(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))

        # return the keras model
        return model

    def compile_model(self, model):
        """
        Description
        ---------------
        Compile the model with the appropriate optimizer and loss function.
        Input(s)
        ---------------
        model : non-compiled model
        Output(s)
        ---------------
        model: compiled model
        """
        # input: the model to compile
        # output : the compiled model
        
        # get optimizer
        if self.mlp_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.mlp_lr, decay=self.mlp_decay, momentum=self.mlp_momentum)
        else:
            optim = getattr(optimizers, self.mlp_optimizer)(learning_rate=self.mlp_lr, decay=self.mlp_decay)

        # compile model 
        model.compile(loss=self.mlp_loss_func, optimizer=optim, metrics=self.metrics)

        # return the compiled keras model
        return model

    def set_model_weights(self, model):
        """
        Description
        ---------------
        Sets the weights to every node in a model.
        Input(s)
        ---------------
        model: Sequential model compiled
        """
        
        # get nodes and activations for each layer    
        layer_configs = ['input']
        for layer in model.layers:
            
            # add flatten since it affects the size of the weights
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            
            # don't add dropout since it doesn't affect weight sizes or activations
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        
        # get bigrams of relevant layers for weights transfer
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        
        # for all layers
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                
                # get all bigram values we already have weights for
                bigram_ids = self.shared_weights['bigram_id'].values
                
                # check if a bigram already exists in the dataframe
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                
                # set layer weights if there is a bigram match in the dataframe 
                if len(search_index) > 0:
                    print("Transferring weights for layer:", config_ids[j])
                    layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                j += 1


    # In this step, we compare the new node with previous existing nodes.
    # If a similar node was already tested, we go to the next node.
    def update_weights(self, model):
        """
        Description
        ---------------
        Updates the nodes with a new model.
        If the resulting model has already been tested, we skip to the next one
        Input(s)
        ---------------
        model : compiled Sequential model with weights initialized
        """
        # input : the model of which we want to update the weight
        # get nodes and activations for each layer
        layer_configs = ['input']
        for layer in model.layers:
            
            # add flatten since it affects the size of the weights
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            
            # don't add dropout since it doesn't affect weight sizes or activations
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        
        # get bigrams of relevant layers for weights transfer
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        
        # for all layers
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                
                #get all bigram values we already have weights for
                bigram_ids = self.shared_weights['bigram_id'].values
                
                # check if a bigram already exists in the dataframe
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                
                # add weights to df in a new row if weights aren't already available
                if len(search_index) == 0:
                    self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j],
                                                                      'weights': layer.get_weights()},
                                                                     ignore_index=True)
                # else update weights 
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                j += 1
        self.shared_weights.to_pickle(self.weights_file)


    # This is simply the step to train the model
    def train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None):
        """
        Description
        ---------------
        Trains a model for a given number of epochs, using the x_data as an input
        and y_data as an output. Returns the history of training of the model.
        Input(s)
        ---------------
        model: Sequential model to train
        x_data: array
        y_data: array
        nb_epochs: int
        validation_split: float
        callback: function
        Output(s)
        ---------------
        history: history
        """
        
        if self.mlp_one_shot:
            self.set_model_weights(model)
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
            self.update_weights(model)
        else:
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                validation_split=validation_split,
                                callbacks=callbacks,
                                verbose=0)
        return history
    
# Note : the controller creates the encodded model (a list of ids)
# It doesn't "understand" what it is doing.
