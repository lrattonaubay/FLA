import os
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from MLP import MLPSearchSpace

from params import *


class Controller(MLPSearchSpace):

    def __init__(self):
        """
        Description
        ---------------
        Creation of a controller object.
        """
        # defining training and sequence creation related parameters
        self.max_len = MAX_ARCHITECTURE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM
        self.use_predictor = CONTROLLER_USE_PREDICTOR
        
        # file path of controller weights to be stored at
        self.controller_weights = './LOGS/controller_weights.h5'

        # initializing a list for all the sequences created
        self.seq_data = []

        # inheriting from the search space
        super().__init__(TARGET_CLASSES)

        # number of classes for the controller (+ 1 for padding)
        self.controller_classes = len(self.vocab) + 1
    
    # Creation of the controller model (LSTM)
    def control_model(self, controller_input_shape):
        """
        Description
        ---------------
        Creates the controller model.
        Input(s)
        ---------------
        controller_input_shape: list of ints
        Output(s)
        ---------------
        model: Model using LSTM
        """
        # inputs : 
            # controller_input_shape : shape of the input
        main_input = Input(shape=controller_input_shape, name='main_input')
        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output])
        return model
    
    '''
    # Creation of an adverserial model with optimization + acc predictors.
    # Version 1
    def hybrid_control_model(self, controller_input_shape, controller_batch_size):
        """
        Description
        ---------------
        A brief description of the function.
        Input(s)
        ---------------
        Input1: Type
        Input2: Type
        ...
        InputN: Type
        Output(s)
        ---------------
        Output1: Type
        Output2: Type
        ...
        OutputN: Type
        """
        # input layer initialized with input shape and batch size
        main_input = Input(shape=controller_input_shape, batch_shape=controller_batch_size, name='main_input')
        
        # LSTM layer
        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        
        # two layers take the same LSTM layer as the input, 
        # the accuracy predictor as well as the sequence generation classification layer
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        
        # finally the Keras Model class is used to create a multi-output model
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
        return model
    '''
    
    # Version 2, the predictor is separate from the main LSTM output
    def hybrid_control_model2(self, controller_input_shape):
        """
        Description
        ---------------
        Creation of an adverserial model with optimization and accuracy
        predictors. In this version, the main LSTM output is separate from the
        predictors.
        Input(s)
        ---------------
        controller_input_shape: list of ints
        Output(s)
        ---------------
        model: Model
        """
    # input layer initialized with input shape and batch size
        main_input = Input(shape=controller_input_shape)#, batch_shape=controller_batch_size, name='main_input')
        
        # LSTM layer
        x1 = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        # output for the sequence generator network
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x1)
    
        # LSTM layer
        x2 = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        # single neuron sigmoid layer for accuracy prediction
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x2)
        
        # finally the Keras Model class is used to create a multi-output model
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
        return model
    
    
    # function used to train the controller with all necessary data.
    def train_control_model(self, model, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
        """
        Description
        ---------------
        Function used to train the controller with all necessary data.
        Input(s)
        ---------------
        model: Model
        x_data: array of inputs
        y_data: array of outputs
        loss_func: function
        controller_batch_size: int
        nb_epochs: int
        """
        # get the optimizer required for training
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr,
                                   decay=self.controller_decay,
                                   momentum=self.controller_momentum)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, 
                                                       decay=self.controller_decay)
                                                       
        # compile model depending on loss function and optimizer provided
        model.compile(optimizer=optim, loss={'main_output': loss_func})
        
        # load controller weights
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
            
        # train the controller
        print("TRAINING CONTROLLER...")
        model.fit(x_data,
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        
        # save controller weights
        model.save_weights(self.controller_weights)
    '''
    # Compatible with version 2
    # pred_target is the prediction of the target that's used as an assumption
    def train_control_model2(self, model, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):
        """
        Description
        ---------------
        Function used to train the controller with all necessary data.
        Input(s)
        ---------------
        model: Model
        x_data: array of inputs
        y_data: array of outputs
        pred_target: 
        ...
        InputN: Type
        """
        # get the optimizer required for training
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr,
                                   decay=self.controller_decay,
                                   momentum=self.controller_momentum)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, 
                                                       decay=self.controller_decay)
                                                       
        # compile model depending on loss function and optimizer provided
        model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                      loss_weights={'main_output': 1, 'predictor_output': 1})
    
        # load controller weights
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
            
        # train the controller
        print("TRAINING CONTROLLER...")
        model.fit(x_data,
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
                  'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        
        # save controller weights
        model.save_weights(self.controller_weights)
    '''
    
    def train_hybrid_model(self, model, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):
        """
        Description
        ---------------
        Function used to train the controller with all necessary data.
        pred_target is used to compare the predicted result for the model.
        Input(s)
        ---------------
        model: Model
        x_data: array of inputs
        y_data: array of outputs
        pred_target: list of ints
        loss_func: function
        controller_batch_size: int
        nb_epochs: int
        """
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                      loss_weights={'main_output': 1, 'predictor_output': 1})
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
        print("TRAINING CONTROLLER...")
        """
        print("pred_target", pred_target[-10:])
        print("size of y_data", len(y_data))
        """
        model.fit(x_data,
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
                   'predictor_output': np.array(pred_target[-len(y_data):]).reshape(len(y_data), 1, 1)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        model.save_weights(self.controller_weights)
    
    # this function samples the sequences in a probabilistic way.
    # This approach is used so that a softmax can help in exploring the search space.
    def sample_architecture_sequences(self, model, number_of_samples):
        """
        Description
        ---------------
        This function samples the sequences in a probabilistic way.
        This approach is used so that a softmax can help in exploring
        the search space. It returns the models generated samples
        Input(s)
        ---------------
        model: Model
        number_of_samples: int
        Output(s)
        ---------------
        samples: list
        """
        # define values needed for sampling 
        final_layer_id = len(self.vocab)
        dropout_id = final_layer_id - 1
        vocab_idx = [0] + list(self.vocab.keys())
        
        # initialize list for architecture samples
        samples = []
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')
        
        # while number of architectures sampled is less than required
        while len(samples) < number_of_samples:
            
            # initialise the empty list for architecture sequence
            seed = []
            
            # while len of generated sequence is less than maximum architecture length
            while len(seed) < self.max_len:
                
                # pad sequence for correctly shaped input for controller
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len - 1)
                
                # given the previous elements, get softmax distribution for the next element
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                probab = probab[0][0]
                
                # sample the next element randomly given the probability of next elements (the softmax distribution)
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]
                
                # first layer isn't dropout
                if next == dropout_id and len(seed) == 0:
                    continue
                # first layer is not final layer
                if next == final_layer_id and len(seed) == 0:
                    continue
                # if final layer, break out of inner loop
                if next == final_layer_id:
                    seed.append(next)
                    break
                # if sequence length is 1 less than maximum, add final
                # layer and break out of inner loop
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break
                # ignore padding
                if not next == 0:
                    seed.append(next)
            
            # check if the generated sequence has been generated before.
            # if not, add it to the sequence data. 
            if seed not in self.seq_data:
                samples.append(seed)
                self.seq_data.append(seed)
        return samples
    
    # This function returns the predicted accuracy of each model generated
    def get_predicted_accuracies_hybrid_model(self, model, seqs):
        """
        Description
        ---------------
        This function returns the predicted accuracy of each model generated.
        To do so, it uses the generated sequence (control)
        Input(s)
        ---------------
        model: Model
        seqs: list
        Output(s)
        ---------------
        pred_accuracies: list
        """
        # inputs :
            # model : the model of the AI (outer loop)
            # the generated sequence (by outer loop)
        pred_accuracies = []        
        for seq in seqs:
            # pad each sequence
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
            # get predicted accuracies
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies
