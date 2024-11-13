'''
Copyright (C) 2023 Fabio Bonassi

This file is part of ssnet.

ssnet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ssnet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with ssnet.  If not, see <http://www.gnu.org/licenses/>.
'''

from datetime import datetime
from typing import Callable, List

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

import ssnet

# %% 
# Load data
output_folder = 'training_output/PH'
data = scipy.io.loadmat('Datasets/PH/Dataset.mat')
U_train, Y_train = data['U_train'], data['Y_train']
U_val, Y_val = data['U_val'], data['Y_val']
U_test, Y_test = data['U_test'], data['Y_test']
U_min, U_max = data['U_min'], data['U_max']

# %%
# Create model

def train_rnn_model(rnn_layers: List[int], rnn_class: Callable, deltaiss_regularizer: torch.nn.Module, train_batch_size: int, 
                    Ts: int = 200, Ns: int = 200, iss_regularizer: torch.nn.Module = None, 
                    lr: float = 1e-3):
    """ Train a RNN model on the PH dataset

    Args:
        rnn_layers (List[int]): List of integers representing the number of units in each layer
        rnn_class (Callable): Class of the RNN to be used (e.g. ssnet.nn.StateSpaceGRU)
        deltaiss_regularizer (torch.nn.Module): Regularizer for the deltaISS
        train_batch_size (int): Batch size for training
        Ts (int, optional): Length of the training subsequences. Defaults to 200.
        Ns (int, optional): Number of training subsequences to extract. Defaults to 200.
        iss_regularizer (torch.nn.Module, optional): Regularizer for ISS. Defaults to None.
        lr (float, optional): Learning rate. Defaults to 1e-3.
    """

    # Define the input and output scalers
    input_scaler = ssnet.data.SequenceScaler(bias=(U_min + U_max) / 2, scale=(U_max - U_min) / 2)
    output_scaler = ssnet.data.MinMaxSequenceScaler()

    # Create the dataloaders (truncate the training sequences into Ns shorter subsequences of Length Ts)
    dataset = ssnet.data.tbptt(training=(U_train, Y_train), validation=(U_val, Y_val), testing=(U_test, Y_test),
                               Ts_train=Ts, Ns_train=Ns, 
                               Ts_val=-1, Ns_val=-1, # Validation dataset is not split into subsequences
                               input_scaler=input_scaler, output_scaler=output_scaler)

    train_loader = DataLoader(dataset.training, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset.validation, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset.testing, batch_size=1, shuffle=False)
    
    # Build the network architecture
    layers = []
    for i, nu in enumerate(rnn_layers):            
        last_layer = i == len(rnn_layers) - 1
        input_size = U_train.shape[1] if i == 0 else rnn_layers[i - 1]
        layers.append(rnn_class(in_features=input_size, units=nu, io_delay=last_layer))
    
    # Output layer
    layers.append(torch.nn.Linear(in_features=rnn_layers[-1], out_features=Y_train.shape[1]))

    net = ssnet.nn.StateSpaceNN(layers=layers, batch_first=True, input_scaler=input_scaler, output_scaler=output_scaler)
    net.init_optimizer(torch.optim.RMSprop, lr=lr, alpha=0.9, momentum=0.1, centered=True)
    

    # A bunch of settings to create a unique description string for the training
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers_str = '-'.join([str(nu) for nu in rnn_layers])
    if rnn_class == ssnet.nn.StateSpaceGRU:
        rnn_str = 'GRU'
    elif rnn_class == ssnet.nn.StateSpaceLSTM:
        rnn_str = 'LSTM'
    else:
        rnn_str = 'RNN'

    descr_str = f'{rnn_str}_{layers_str}_bs{train_batch_size}_Ts{Ts}_Ns{Ns}{"_iss" if iss_regularizer is not None else ""}'\
                f'{"_deltaiss" if deltaiss_regularizer is not None else ""}_{current_time}'

    # Callbacks!
    # - SigIntCallback: Stops the training if a SIGINT signal is received (Ctrl+C)
    # - TargetMetricCallback: Stops the training if the validation loss is below a certain threshold
    # - EarlyStoppingCallback: Stops the training if the validation loss does not improve for a certain number of epochs
    # - LoggingCallback: Logs the training progress to the console
    # - MatlabExportCallback: Exports the network weights to a .mat file
    # - PerformanceTestingCallback: Tests the network performance on the *test set* at regular intervals and generate a figure
    callbacks = ssnet.callbacks.CallbacksWrapper(tensorboard_instance=f'{output_folder}/{descr_str}', 
                                             matfile_instance=f'{output_folder}/{descr_str}/net.mat',
                                             callbacks=[ssnet.callbacks.SigIntCallback(), 
                                                        ssnet.callbacks.TargetMetricCallback(1e-5), 
                                                        ssnet.callbacks.EarlyStoppingCallback(patience=300, watch_from=100),
                                                        ssnet.callbacks.LoggingCallback(),
                                                        ssnet.callbacks.MatlabExportCallback(),
                                                        ssnet.callbacks.PerformanceTestingCallback(test_loader, plot_frequency=100)])

    # Do the training! 
    # - MSE loss for training and validation
    # - washout of 25 steps
    # - 3000 epochs max
    # - Optional regularization for ISS or deltaISS
    training_results = net.fit(criterion=torch.nn.MSELoss(), 
                               train_loader=train_loader, 
                               val_loader=val_loader, 
                               val_metric=torch.nn.MSELoss(),
                               iss_regularizer=iss_regularizer if deltaiss_regularizer is None else None, 
                               deltaiss_regularizer=deltaiss_regularizer,
                               callbacks=callbacks,
                               washout=25,
                               epochs=3000)

    return training_results, descr_str

# %%
# Train the model
diss_reg = ssnet.nn.GeneralizedPieceWiseRegularizer(clearance=0.04, omega_plus=8e-4, omega_minus=1e-7, steepness=10.0) # See Chapter 4 of the thesis for details

training_results, descr_str = train_rnn_model(rnn_layers=[8, 8], 
                                              rnn_class=ssnet.nn.StateSpaceGRU,
                                              Ts=300, 
                                              Ns=200, 
                                              deltaiss_regularizer=diss_reg, 
                                              train_batch_size=32)