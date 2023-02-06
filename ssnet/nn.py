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

from __future__ import annotations

import abc
import math
import pydoc
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat
from sorcery import dict_of
from torch.functional import Tensor
from torch.nn.modules.container import ParameterList
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import PostEpochTrainingData, TrainingCallback
from .utils import MatFileWriter

if TYPE_CHECKING:
    from .data import SequenceScaler
    from .nn import StateSpaceNN, StateSpaceRecurrentLayer

class StateSpaceNN(nn.Module):
    __constants__ = ['batch_first', 'trainable']
    batch_fist: bool
    trainable: bool
    input_scaler: SequenceScaler
    output_scaler: SequenceScaler

    _CFG_CONFIG: str = 'configuration'
    _CFG_LAYER_CLASS: str = 'layers_class'
    _CFG_LAYER_CONFIG: str = 'layers_configuration'
    _CFG_INPUT_SCALER_CONFIG: str = 'input_scaler_configuration'
    _CFG_OUTPUT_SCALER_CONFIG: str = 'output_scaler_configuration'
    _CFG_INPUT_SCALER_CLASS: str = 'input_scaler_class'
    _CFG_OUTPUT_SCALER_CLASS: str = 'output_scaler_class'
    _EXP_DESCR: str = 'description'
    _EXP_CONFIG: str = 'configuration'
    _EXP_STATEDICT: str = 'state_dict'
    _EXP_LAYERS: str = 'layers'
    _EXP_INPUT_SCALER: str = 'input_scaler'
    _EXP_OUTPUT_SCALER: str = 'output_scaler'
    
    def __init__(self, layers: List[nn.Module] | nn.Module, batch_first: bool = False, trainable: bool = True, 
                 input_scaler: SequenceScaler = None, output_scaler: SequenceScaler = None):
        """
        Neural Network in State-Space form

        Parameters
        ----------
        layers : List[nn.Module] | nn.Module
            List containing the ordered layers of the neural network
        batch_first: bool, optional
            Set to True if batches are on the 0-axis, by default False
        trainable: bool, optional
            Flag indicating if the Neural Network can be trained or not, by default True
        input_scaler: SequenceScaler, optional
            Scaler for the input data, by default None
        output_scaler: SequenceScaler, optional
            Scaler for the output data, by default None
        """
        super(StateSpaceNN, self).__init__()
        if not isinstance(layers, Iterable):
            layers = [layers]

        self.layers = nn.ModuleList(layers)
        self.cell_states = [(c.state_size if isinstance(c, StateSpaceRecurrentLayer) else 0) for c in layers]
        self.n_layers = len(layers)
        self.n_dyn_layers = len([c for c in layers if isinstance(c, StateSpaceRecurrentLayer)])
        self.batch_first = batch_first
        self.trainable = trainable
        self.optimizer = None
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        if not trainable:
            self.disable_training()

    @property
    def state_size(self) -> int:
        """
        Compute the overall state dimension

        Returns
        -------
        int
            Number of states of the entire network
        """
        return sum(self.cell_states)

    def initial_states(self, n_batches: int, stdv: float = 0.5) -> Tensor:
        """
        Generate random initial states for the SSNN

        Parameters
        ----------
        n_batches : int
            Number of batches
        stdv : float, optional
            Standard deviation of the random initial states, by default 0.5

        Returns
        -------
        Tensor
            Initial states
        """
        return stdv * torch.randn(n_batches, self.state_size, requires_grad=False)


    @contextmanager
    def evaluating(self):
        """Temporarily switch to evaluation mode."""
        istrain = self.training
        try:
            self.eval()
            yield self
        finally:
            if istrain:
                self.train()

    def extra_repr(self) -> str:
        """
        Get a string representation of the State Space NN
        """

        mrepr = f"states={self.state_size}"
        return mrepr

    def disable_training(self):
        """
        Make the StateSpaceNN untrainable by setting the requires_grad field of each parameter to False.
        """
        self.eval()
        self.trainable = False
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.layers:
            if hasattr(layer, 'trainable'): 
                layer.trainable = False

    def get_configuration(self, sanitize_callables: bool = True) -> dict:
        """
        Returns the configuration of the StateSpaceNN model

        Parameters
        ----------
        sanitize_callables : bool, optional
            Replace function pointers with strings, by default True

        Returns
        -------
        dict
            The configuration.
        """

        def get_submodule_configuration(m: object) -> dict:
            # Auxiliary function which builds the configuration of a generic module
            if not hasattr(m, '__constants__'):
                return {}

            config = {}
            for cname in m.__constants__:
                c = getattr(m, cname) 
                if isinstance(c, list):
                    if any([isinstance(ci, Callable) for ci in c]):
                        config[cname] = [ci.__name__ for ci in c] if sanitize_callables else c
                    else:
                        config[cname] = c
                else:
                    config[cname] = c.__name__ if isinstance(c, Callable) and sanitize_callables else c  # pylint: disable=maybe-no-member
            return config

        def class_to_string(obj: object) -> str:
            # Auxiliary function which turns a class into a string
            classname = obj.__class__.__name__
            module = obj.__module__
            if module == '__builtin__':
                return classname
            else:
                return f'{module}.{classname}'

        nn_config = {}

        # Wrapper configuration
        nn_config[StateSpaceNN._CFG_CONFIG] = get_submodule_configuration(self)

        # Layer classes
        nn_config[StateSpaceNN._CFG_LAYER_CLASS] = tuple([class_to_string(cell) for cell in self.layers])

        if self.input_scaler is not None:
            nn_config[StateSpaceNN._CFG_INPUT_SCALER_CLASS] = class_to_string(self.input_scaler)
            nn_config[StateSpaceNN._CFG_INPUT_SCALER_CONFIG] = get_submodule_configuration(self.input_scaler)
        if self.output_scaler is not None:
            nn_config[StateSpaceNN._CFG_OUTPUT_SCALER_CLASS] = class_to_string(self.output_scaler)
            nn_config[StateSpaceNN._CFG_OUTPUT_SCALER_CONFIG] = get_submodule_configuration(self.output_scaler)
        
        layers_configuration = []
        for cell in self.layers:
            cell_configuration = get_submodule_configuration(cell)
            layers_configuration.append(cell_configuration)

        nn_config[StateSpaceNN._CFG_LAYER_CONFIG] = tuple(layers_configuration)
        return nn_config

    @staticmethod
    def from_configuration(configuration: dict) -> StateSpaceNN:
        """
        Factory function which builds a StateSpaceNN from a configuration dictionary.

        Parameters
        ----------
        configuration : dict
            The configuration of the StateSpaceNN to build

        Returns
        -------
        StateSpaceNN
            The built StateSpaceNN object
        """
        layers = []

        for (layer_class, layer_config) in zip(configuration[StateSpaceNN._CFG_LAYER_CLASS], configuration[StateSpaceNN._CFG_LAYER_CONFIG]):
            cls = pydoc.locate(layer_class)
            layers.append(cls(**layer_config))

        input_scaler = None
        output_scaler = None
        if StateSpaceNN._CFG_INPUT_SCALER_CLASS in configuration:
            input_scaler = pydoc.locate(configuration[StateSpaceNN._CFG_INPUT_SCALER_CLASS])(**configuration[StateSpaceNN._CFG_INPUT_SCALER_CONFIG])
        if StateSpaceNN._CFG_OUTPUT_SCALER_CLASS in configuration:
            output_scaler = pydoc.locate(configuration[StateSpaceNN._CFG_OUTPUT_SCALER_CLASS])(**configuration[StateSpaceNN._CFG_OUTPUT_SCALER_CONFIG])

        net_config = configuration[StateSpaceNN._CFG_CONFIG]
        return StateSpaceNN(layers=layers, input_scaler=input_scaler, output_scaler=output_scaler, **net_config)

    def export_to_matlab(self, matfile: str | Path | MatFileWriter):
        """
        Export the StateSpaceNN to a Matlab file.

        Parameters
        ----------
        matfile : str | Path | MatFileWriter
            The String or Path to the Matlab file or a MatFileWriter object.
        """

        def tensor_to_np(tensors: dict[Tensor]) -> dict[np.ndarray]:
            """
            Auxiliary function which scans a dictionary of tensors and converts them to numpy arrays
            """
            nps = tensors.copy()
            for key, value in nps.items():
                if isinstance(value, torch.Tensor):
                    nps[key] = value.clone().detach().numpy()
            return nps

        config = self.get_configuration(sanitize_callables=True)
        layerdata = []

        for i, layer in enumerate(self.layers):
            layerdata += [{
                **config[StateSpaceNN._CFG_LAYER_CONFIG][i],
                'class': config[StateSpaceNN._CFG_LAYER_CLASS][i],
                'shortclass': layer.__class__.__name__,
                'weights': tensor_to_np(layer.state_dict())
            }]

        data = {StateSpaceNN._EXP_DESCR: repr(self),
                StateSpaceNN._EXP_CONFIG: config[StateSpaceNN._CFG_CONFIG],
                StateSpaceNN._EXP_LAYERS: layerdata}

        if self.input_scaler:
            scaler_cfg = tensor_to_np(config[StateSpaceNN._CFG_INPUT_SCALER_CONFIG])
            input_scaler_cfg = {**scaler_cfg,
                                'class': config[StateSpaceNN._CFG_INPUT_SCALER_CLASS],
                                'shortclass': self.input_scaler.__class__.__name__ }
            data[StateSpaceNN._EXP_INPUT_SCALER] = input_scaler_cfg

        if self.output_scaler:
            scaler_cfg = tensor_to_np(config[StateSpaceNN._CFG_OUTPUT_SCALER_CONFIG])
            output_scaler_cfg = {**scaler_cfg,
                                'class': config[StateSpaceNN._CFG_OUTPUT_SCALER_CLASS],
                                'shortclass': self.output_scaler.__class__.__name__ }
            data[StateSpaceNN._EXP_OUTPUT_SCALER] = output_scaler_cfg

        if isinstance(matfile, (str, Path)):
            savemat(matfile, data, appendmat=True)
        elif isinstance(matfile, MatFileWriter):
            matfile.push(**data)
        else:
            raise ValueError('Unsuitable type for argument matfile')

    def save_model(self, path: str | Path):
        """
        Save the StateSpaceNN to a file

        Parameters
        ----------
        path : str | Path
            The path of the file which is supposed to store the data
        """
        if isinstance(path, str):
            path = Path(path)

        data = {StateSpaceNN._EXP_DESCR: repr(self), 
                StateSpaceNN._EXP_STATEDICT: self.state_dict(),
                StateSpaceNN._EXP_CONFIG: self.get_configuration(sanitize_callables=True)}
        torch.save(data, path)   


    @staticmethod
    def load_model(path: str | Path, disable_training: bool = True) -> StateSpaceNN:
        """
        Load a (trained) StateSpaceNN model from data.

        Parameters
        ----------
        path : str | Path
            The path of the file that stores the network data.
        disable_training : bool, optional
            Disable the training of the loaded StateSpaceNN, by default True

        Returns
        -------
        StateSpaceNN
            The loaded StateSpaceNN
        """
        data = torch.load(path)
        config = data[StateSpaceNN._EXP_CONFIG]

        if disable_training:
            config[StateSpaceNN._CFG_CONFIG]['trainable'] = False
            for lconf in config[StateSpaceNN._CFG_LAYER_CONFIG]:
                if 'trainable' in lconf:
                    lconf['trainable'] = False

        net = StateSpaceNN.from_configuration(data[StateSpaceNN._EXP_CONFIG])
        net.load_state_dict(data[StateSpaceNN._EXP_STATEDICT])
        return net

    @staticmethod
    def concatenate(nns: Iterable[StateSpaceNN]) -> StateSpaceNN:
        """
        Concatenate several StateSpaceNN object

        Parameters
        ----------
        nns : Iterable[StateSpaceNN]
            A Tuple (or List) of StateSpaceNN to join.

        Returns
        -------
        StateSpaceNN
           The StateSpaceNN with joint layers.
        """
        if not (all([nn.batch_first for nn in nns]) or all([not nn.batch_first for nn in nns])):
            raise ValueError('The networks to join must have the same value for batch_first')

        batch_first = nns[0].batch_first
        any_trainable = any([nn.trainable for nn in nns])
        layers = []
        for nn in nns:
            if not nn.trainable:
                nn.disable_training()   # Make sure that untrainable layers have no gradient
            layers = layers + list(nn.layers)

        return StateSpaceNN(layers=layers, batch_first=batch_first, trainable=any_trainable)


    def _recurrence(self, u: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform one step of recurrence. Compute the output y(k) and the state x(k+1) given the initial state x(k) and the input u(k).
        This is an internal method. Use forward() instead.

        Parameters
        ----------
        u : Tensor
            The input at the current time instant, i.e. u(k)
        x0 : [type]
            The initial state, i.e. x(k)

        Returns
        -------
        Tuple[Tensor, Tensor]
            y : The output y(k)
            xp : The state x(k+1)
        """
        ui = u
        x0 = torch.split(x0, self.cell_states, dim=1)
        xt = [None] * self.n_layers
        for i, cell in enumerate(self.layers):
            if isinstance(cell, StateSpaceRecurrentLayer):
                ui, t = cell(ui, x0[i])
                xt[i] = t
            else:
                ui = cell(ui)
                xt[i] = torch.zeros_like(x0[i])
        # Debug here

        xp = torch.cat(xt, dim=1)
        return ui, xp

    def forward(self, u: Tensor, x0: Tensor = None, auto_normalization: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Forward simulation of the State Space Neural Network

        Parameters
        ----------
        u : Tensor
            Batch of input sequences with shape `(Time, batches, inputs)`
        x0 : Tensor, optional
            Initial states with shape `(batches, states)`, by default None
        auto_normalization : bool, optional
            If True, the input and output are normalized before the forward pass (using `input_scaler` and `output_scaler`), by default False

        Returns
        -------
        Tuple[Tensor, Tensor]
            y_seq : output sequence with shape `(Time, batches, outputs)`
            x_seq : state sequence with shape `(Time, batches, states)`
        """
        
        un = u.transpose(0, 1) if self.batch_first else u

        if auto_normalization and self.input_scaler:
            un = self.input_scaler.normalize(un)

        if x0 is None:
            x0 = self.initial_states(un.shape[-2])
        output = []
        state = []

        for t, ut in enumerate(un):
            y, x0 = self._recurrence(ut, x0)
            state.append(x0)
            output.append(y)

        y_seq = torch.stack(output, dim=0)
        x_seq = torch.stack(state, dim=0)

        if self.batch_first:
            x_seq = x_seq.transpose(0, 1)
            y_seq = y_seq.transpose(0, 1)

        if auto_normalization and self.output_scaler:
            y_seq = self.output_scaler.denormalize(y_seq)

        return y_seq, x_seq

    def iss_residuals(self) -> List[Tensor]:
        """
        Return the ISS residuals of all the layers of the network

        Returns
        -------
        List[Tensor]
            residuals: the ISS residuals of all the recurrent layers
        """
        residuals = []

        for cell in self.layers:
            if isinstance(cell, StateSpaceRecurrentLayer):
                res = cell.iss_residuals()
                if isinstance(res, Tuple):
                    residuals += list(res)
                elif isinstance(res, torch.Tensor):
                    residuals += [res]
                else:
                    raise ValueError("Unexpected data type for ISS residuals")

        return residuals

    def deltaiss_residuals(self) -> List[Tensor]:
        """
        Compute the residual of the {\delta}ISS constraint

        Returns
        -------
        List[Tensor]
            residuals: the residual of the {\delta}ISS constraint
        """
        residuals = []

        for cell in self.layers:
            if isinstance(cell, StateSpaceRecurrentLayer):
                res = cell.deltaiss_residuals()
                if isinstance(res, Tuple):
                    residuals += list(res)
                elif isinstance(res, torch.Tensor):
                    residuals += [res]
                else:
                    raise ValueError("Unexpected data type for deltaISS residuals")

        return residuals


    def init_optimizer(self, optimizer_init: Callable, lr: float = 1e-3, **opt_params):
        """
        Setup the optimization algorithm

        Parameters
        ----------
        optimizer_init : Callable
            Pointer to the init function of the optimizer
        lr : float, optional
            Learning rate, by default 1e-3
        opt_params : dict, optional
           Parameters of the optimization algorithm, by default {}

        Returns
        ----------
        dict
            The callbacks logs
        """
        self.optimizer = optimizer_init(self.parameters(), lr=lr, **opt_params)

    def fit(
        self,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_metric: torch.nn.Module | Callable,
        iss_regularizer: Callable = None,
        deltaiss_regularizer: Callable = None,
        epochs: int = 1e4,
        washout: int = 0,
        callbacks: TrainingCallback = None) -> dict:

        if self.optimizer is None:
            raise ValueError("An optimizer must be specified before training the network.")
        if not self.trainable:
            raise ValueError('Cannot train a NN for which training has been disabled.')

        # Extend the callbacks with gain-scheduling strategies on the loss function/regularizers
        _callbacks = callbacks
        if isinstance(criterion, TrainingCallback):
            _callbacks.append(criterion)
        if isinstance(iss_regularizer, TrainingCallback):
            _callbacks.append(iss_regularizer)
        if isinstance(deltaiss_regularizer, TrainingCallback):
            _callbacks.append(deltaiss_regularizer)

        loss = torch.tensor(np.inf, dtype=torch.float)

        self.train()
        _callbacks.on_training_start(self)
        train_msg = tqdm(range(1, epochs + 1), desc="Training")

        for epoch in train_msg:
            train_loss_avg = 0.0
            train_metric_avg = 0.0

            _callbacks.on_epoch_start(self, epoch=epoch)

            for batch in train_loader:
                u_batch, y_batch = batch

                self.optimizer.zero_grad()
                y_hat, _ = self(u_batch)

                loss = criterion(y_hat[:, washout:, :], y_batch[:, washout:, :])

                if iss_regularizer is not None:
                    loss = loss + iss_regularizer(self.iss_residuals()) / self.n_dyn_layers
                if deltaiss_regularizer is not None:
                    loss = loss + deltaiss_regularizer(self.deltaiss_residuals()) / self.n_dyn_layers

                loss.backward()
                self.optimizer.step()

                with torch.no_grad() and self.evaluating():
                    metric = val_metric(y_hat[:, washout:, :], y_batch[:, washout:, :])
                    train_metric_avg += metric.detach().item() / len(train_loader)

                # Save the loss for logging purposes
                train_loss_avg += loss.detach().item() / len(train_loader)

            # Validation
            with torch.no_grad() and self.evaluating():
                iss_residuals = [nu.detach().item() for nu in self.iss_residuals()]
                deltaiss_residuals = [nu.detach().item() for nu in self.deltaiss_residuals()]
                val_metric_avg = 0.0

                for batch in val_loader:
                    u_batch, y_batch = batch

                    self.optimizer.zero_grad()
                    y_hat, _ = self(u_batch)

                    metric = val_metric(y_hat[:, washout:, :], y_batch[:, washout:, :])

                    # Save the loss for logging purposes
                    val_metric_avg += metric.detach().item() / len(val_loader)

            epoch_data = PostEpochTrainingData(
                epoch=epoch,
                train_loss=train_loss_avg,
                train_metric=train_metric_avg,
                val_metric=val_metric_avg,
                iss_residuals=iss_residuals,
                deltaiss_residuals=deltaiss_residuals,
                enforce_deltaiss=deltaiss_regularizer is not None,
                enforce_iss=iss_regularizer is not None,
                metric_fcn=val_metric,
                washout=washout
            )

            if _callbacks.on_epoch_end(epoch_data, self):
                print(f"Training procedure stopped.")
                break

        # Call the on_training_end of the callbacks
        callback_logs = _callbacks.on_training_end(epoch_data, self)

        # Turn the network to eval mode
        self.eval()

        return callback_logs


class StateSpaceRecurrentLayer(nn.Module, metaclass=abc.ABCMeta):
    units: int
    __constants__ = []


    def __init__(
        self,
        name: str = "RecurrentLayer",
        init_input: Callable = None,
        init_kernel: Callable = None,
        init_bias: Callable = None):

        super(StateSpaceRecurrentLayer, self).__init__()
        self.name = name

        self.init_input_ = init_input
        self.init_kernel_ = init_kernel
        self.init_bias_ = init_bias

    def _check_init_inplace(self, initializer):
        """
        Assess that the initializer are in-place operation

        Parameters
        ----------
        initializer : Callable
            The initializer to check

        Raises
        ------
        ValueError
            The initializer is not an in-place operation
        """
        test_random = np.random.rand(1, 1)
        test_tensor = torch.tensor(test_random)
        initializer(test_tensor)
        if np.equal(test_random, test_tensor.detach()):
            raise ValueError("Initializers must be in-place operations!")

    def check_init_fcns(self):
        """
        Overwrite None initializers with defaults (`uniform_` for kernel ad inputs, `zeros_` for biases)
        """
        if self.init_input_ is None:
            self.init_input_ = lambda x: nn.init.uniform_(x, -1.0 / self.units, 1.0 / self.units)
        else:
            self._check_init_inplace(self.init_input_)

        if self.init_kernel_ is None:
            self.init_kernel_ = lambda x: nn.init.uniform_(x, -1.0 / self.units, 1.0 / self.units)
        else:
            self._check_init_inplace(self.init_kernel_)

        if self.init_bias_ is None:
            self.init_bias_ = nn.init.zeros_
        else:
            self._check_init_inplace(self.init_bias_)

    def reset_parameters(self):
        """
        Reset NN parameters
        """
        # Check if any initializer is None, and if necessary set it to its default.
        self.check_init_fcns()

        for wname, weight in self.named_parameters():
            ngates = max(weight.shape[-1] // self.units, 1)

            # In general initialization must be performed separately for each weight matrix
            chunked_weights = torch.chunk(weight, chunks=ngates, dim=-1)
            if wname.startswith("W"):
                init_fcn = self.init_input_
            elif wname.startswith("U"):
                init_fcn = self.init_kernel_
            elif wname.startswith("b"):
                init_fcn = self.init_bias_
            else:
                raise ValueError("Weight not recognized")

            for w in chunked_weights:
                init_fcn(w)

    def initial_states(self, n_batches: int, stdv: float = 0.5) -> Tensor:
        """
        Generate random initial states for the StateSpaceRecurrentLayer

        Parameters
        ----------
        n_batches : int
            Number of batches
        stdv : float, optional
            Standard deviation of the random initial states, by default 0.5

        Returns
        -------
        Tensor
            Initial states
        """
        return stdv * torch.randn(n_batches, self.state_size, requires_grad=False)

    def _gate_bounds(self, W: Tensor, U: Tensor, b: Tensor, activation: Callable, p: int = np.inf) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the matrices norms and the bound of a gate.

        Parameters
        ----------
        W : Tensor
            The W matrix of the gate, with shape `(states, inputs)`
        U : Tensor
            The U matrix of the gate, with shape `(states, states)`
        b : Tensor
            The bias of the gate, with shape `(states, 1)`
        activation : Callable
            The activation function of the gate
        p : int, optional
            The norm order, by default np.inf

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            sigma : The bound of the gate
            W_norm : The p-norm of W
            U_norm : The p-norm of U
        """
        if W.shape[0] != U.shape[0]:
            raise ValueError("The weight matrices W and U were not transposed")
        if b.ndim == 1:
            b = b.unsqueeze(1)

        sigarg = torch.cat([W, U, b], dim=1)
        sigarg = torch.linalg.norm(sigarg, ord=np.inf)
        return (
            activation(sigarg),
            torch.linalg.norm(W, ord=p),
            torch.linalg.norm(U, ord=p),
        )

    @abc.abstractmethod
    def forward(self, u, x0=None):
        pass

    @property
    @abc.abstractmethod
    def state_size(self):
        pass

    @abc.abstractmethod
    def iss_residuals(self):
        pass

    @abc.abstractmethod
    def deltaiss_residuals(self):
        pass


class StateSpaceVanillaRNN(StateSpaceRecurrentLayer):
    __constants__ = ['units', 'in_features', 'io_delay', 'state_activation', 'trainable',
                     'origin_equilibrium', 'name']
    units: int
    in_features: int
    io_delay: bool
    state_activation: Callable | str
    trainable: bool
    origin_equilibrium: bool
    name: str

    def __init__(
        self,
        units: int,
        in_features: int,
        io_delay: bool = True,
        state_activation=torch.tanh,
        trainable: bool = True,
        origin_equilibrium: bool = False,
        init_input: Callable = None,
        init_kernel: Callable = None,
        init_bias: Callable = None,
        name: str = "RNN",
    ):
        """
        StateSpaceVanillaRNN: a vanilla RNN implementation in State-Space form

        Parameters
        ----------
        units : int
            Number of neurons of the RNN
        in_features : int
            Number of inputs
        io_delay : bool, optional
            Return the state at the previous time instant, by default True
        state_activation : Callable, optional
            The activation function of the state-update law, by `default F.tanh`
        trainable : bool, optional
            Enable the training of the network, by default True
        origin_equilibrium : bool, optional
            Enforce the origin (u=0) to be an equilibrium of the system, by default False
        init_input : Callable, optional
            The in-place initializer of the input weights, by default None (`uniform_`)
        init_kernel : Callable, optional
            The in-place initializer of the kernel weights, by default None (`uniform_`)
        init_bias : Callable, optional
            The in-place initializer of the biases, by default None (`zeros_`)
        name: str, optional
            Name of the RNN
        """
        super(StateSpaceVanillaRNN, self).__init__(name, init_input=init_input, init_kernel=init_kernel, init_bias=init_bias)

        self.units = units
        self.in_features = in_features
        self.io_delay = io_delay
        self.state_activation =  getattr(torch, state_activation) if isinstance(state_activation, str) else state_activation 
        self.trainable = trainable
        self.origin_equilibrium = origin_equilibrium

        # Register parameters
        self.W = Parameter(torch.zeros(units, in_features), requires_grad=trainable)
        self.U = Parameter(torch.zeros(units, units), requires_grad=trainable)
        self.b = Parameter(torch.zeros(units), requires_grad=trainable and not origin_equilibrium)

        # Initialize them
        self.reset_parameters()

        if origin_equilibrium:
            torch.zero_(self.b)

    @property
    def state_size(self):
        return self.units

    def forward(self, u, x0=None):
        if x0 is None:
            x0 = self.initial_states(u.shape[-2])

        x = torch.matmul(u, self.W.t()) + torch.matmul(x0, self.U.t()) + self.b
        xp = self.state_activation(x)
        y = x0 if self.io_delay else xp

        return y, xp

    def iss_residuals(self):
        return []

    def deltaiss_residuals(self):
        return []

class StateSpaceGRU(StateSpaceRecurrentLayer):

    __constants__ = ['units', 'in_features', 'io_delay', 'gate_activation', 'input_activation',
                     'trainable', 'minimal', 'origin_equilibrium', 'forget_bias', 'name']
    units: int
    in_features: int
    io_delay: bool
    gate_activation: Callable
    input_activation: Callable 
    trainable: bool
    minimal: bool
    origin_equilibrium: bool
    forget_bias: bool
    name: str

    def __init__(
        self,
        units: int,
        in_features: int,
        io_delay: bool = True,
        gate_activation: Callable = torch.sigmoid,
        input_activation: Callable = torch.tanh,
        trainable: bool = True,
        minimal: bool = False,
        origin_equilibrium: bool = False,
        forget_bias: bool = False,
        init_input: Callable = None,
        init_kernel: Callable = None,
        init_bias: Callable = None,
        name: str = "GRU",
    ):
        """
        State Space Gated Recurrent Units

        Parameters
        ----------
        units : int
            Number of neurons of the GRU
        in_features : int
            Number of inputs
        io_delay : bool, optional
            Return the state at the previous time instant, by default True
        gate_activation : Callable, optional
            Activation function of the gate, by default `torch.sigmoid`
        input_activation : Callable, optional
            Input squashing function, by default `torch.tanh`
        trainable : bool, optional
            Enable the training of the network, by default True
        minimal : bool, optional
            Implement a minimal GRU with no forget gate, by default False
        origin_equilibrium : bool, optional
            Enforce the origin (u=0) to be an equilibrium of the system, by default False
        forget_bias : bool, optional
            Introduce an additional bias in the squashed input, multiplied to the forget gate, by default False
        init_input : Callable, optional
            The in-place initializer of the input weights, by default None (`uniform_`)
        init_kernel : Callable, optional
            The in-place initializer of the kernel weights, by default None (`uniform_`)
        init_bias : Callable, optional
            The in-place initializer of the biases, by default None (`zeros_`)
        name : str, optional
            Name of the GRU
        """
        super(StateSpaceGRU, self).__init__(name, init_input=init_input, init_kernel=init_kernel, init_bias=init_bias)
        self.units = units
        self.in_features = in_features
        self.io_delay = io_delay
        self.gate_activation = getattr(torch, gate_activation) if isinstance(gate_activation, str) else gate_activation 
        self.input_activation = getattr(torch, input_activation) if isinstance(input_activation, str) else input_activation
        self.trainable = trainable
        self.minimal = minimal
        self.origin_equilibrium = origin_equilibrium
        self.forget_bias = forget_bias

        self._n_gates = 2 + int(not minimal)

        self.Wzf = Parameter(torch.zeros(in_features, (self._n_gates - 1) * units), requires_grad=trainable)
        self.Uzf = Parameter(torch.zeros(units, (self._n_gates - 1) * units), requires_grad=trainable)

        self.Wr = Parameter(torch.zeros(in_features, units), requires_grad=trainable)
        self.Ur = Parameter(torch.zeros(units, units), requires_grad=trainable)

        self.bzf = Parameter(torch.zeros((self._n_gates - 1) * units), requires_grad=trainable)
        self.br = Parameter(torch.zeros(units), requires_grad=trainable and not origin_equilibrium)

        if self.forget_bias and not self.origin_equilibrium:
            self.brx = Parameter(torch.zeros(units), requires_grad=trainable and not origin_equilibrium)

        self.reset_parameters()

        if origin_equilibrium:
            torch.zero_(self.br)

    def extra_repr(self) -> str:
        return (
            f"units={self.units}, minimal={self.minimal}, in_features={self.in_features}, io_delay={self.io_delay}, "
            f"origin_equilibrium={self.origin_equilibrium}, forget_bias={self.forget_bias}, gate_activation={self.gate_activation.__name__}, "
            f"input_activation={self.input_activation.__name__}, trainable={self.trainable}"
        )

    @property
    def state_size(self) -> int:
        """
        Retrieve the number of states

        Returns
        -------
        int
            Number of states
        """
        return self.units

    def forward(self, u: Tensor, x0: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward-pass of the GRU

        Parameters
        ----------
        u : Tensor
            Input u(k) of the GRU layer, with shape `(batch_size, in_features)`
        x0 : Tensor, optional
            Current state x(k) of the GRU, with shape `(batch_size, n_units)`, by default None

        Returns
        -------
        Tuple[Tensor, Tensor]
            y : the output y(k), with shape `(batch_size, n_units)`
            xp : the new state x(k+1), with shape `(batch_size, n_units)`
        """
        if x0 is None:
            x0 = self.initial_states(u.shape[-2])

        if self.minimal:
            # In minimal GRUs f = 1
            # (shape: (batch_size, n_units))
            z = self.gate_activation(torch.matmul(u, self.Wzf) + torch.matmul(x0, self.Uzf) + self.bzf)
            r = self.input_activation(torch.matmul(u, self.Wr) + torch.matmul(x0, self.Ur) + self.br)  # (shape: (batch_size, n_units))
        else:
            gates = self.gate_activation(torch.matmul(u, self.Wzf) + torch.matmul(x0, self.Uzf) + self.bzf)  # (shape: (batch_size, 2*n_units))
            
            z, f = torch.chunk(gates, 2, dim=1) # (shape: (batch_size, n_units))
            if not self.forget_bias: # (shape: (batch_size, n_units))
                r = self.input_activation(torch.matmul(u, self.Wr) + torch.matmul(f * x0, self.Ur) + self.br)
            else:
                r = self.input_activation(torch.matmul(u, self.Wr) + f * (torch.matmul(x0, self.Ur) + self.brx) + self.br)  # (shape: (batch_size, n_units))

        # GRU state update x(k+1) = f(x(k), u(k))
        xp = z * x0 + (1 - z) * r  # (shape: (batch_size, n_units))
        # If `io_delay` return x(k), else x(k+1)
        y = x0 if self.io_delay else xp  # (shape: (batch_size, n_outputs))

        return y, xp

    def _get_norms(self, p: int = np.inf) -> Tuple[
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
    ]:
        """
        Retrieve the gates' bounds and the weights' norms of the GRU layer.

        Parameters
        ----------
        ord : int, optional
            The order of the weights' norms, by default np.inf

        Returns
        -------
        Dict
            (phi_r, sigma_z, sigma_f) : The bounds of the gates
            (Wr_norm, Wz_norm, Wf_norm) : The norms of the W matrices
            (Ur_norm, Uz_norm, Uf_norm) : The norms of the U matrices
        """
        Wr = self.Wr.t()  # (shape: (n_units, in_features))
        Ur = self.Ur.t()  # (shape: (n_units, n_units))
        br = self.br  # (shape: (n_units, 1))

        if self.minimal:
            Wz = self.Wzf.t()
            Uz = self.Uzf.t()
            Wf_norm = 0.0
            Uf_norm = 0.0
            sigma_f = 1.0
        else:
            # (shape: (n_units, in_features))
            Wz, Wf = torch.chunk(self.Wzf, 2, dim=1)
            # (shape: (n_units, n_units))
            Uz, Uf = torch.chunk(self.Uzf, 2, dim=1)
            bz, bf = torch.chunk(self.bzf, 2, dim=0)  # (shape: (n_units, 1))
            Wz, Wf, Uz, Uf = Wz.t(), Wf.t(), Uz.t(), Uf.t()

            sigma_f, Wf_norm, Uf_norm = self._gate_bounds(Wf, Uf, bf, self.gate_activation, p=p)

        sigma_z, Wz_norm, Uz_norm = self._gate_bounds(Wz, Uz, bz, activation=self.gate_activation, p=p)
        phi_r, Wr_norm, Ur_norm = self._gate_bounds(Wr, Ur, br, activation=self.input_activation, p=p)

        return dict_of(
            sigma_z,
            sigma_f,
            phi_r,
            Wz_norm,
            Wr_norm,
            Wf_norm,
            Uz_norm,
            Ur_norm,
            Uf_norm,
        )

    def iss_residuals(self) -> Tensor:
        """
        Compute the residual of the ISS constraint

        Returns
        -------
        Tensor
            v: the residual of the ISS constraint
        """
        if self.gate_activation is not torch.sigmoid or self.input_activation is not torch.tanh:
            raise ValueError(
                "The ISS condition is valid when `gate_activation` is set to `torch.sigmoid` and `input_activation` is set to `torch.tanh`!"
            )
        norms = self._get_norms(p=np.inf)
        return norms["Ur_norm"] * norms['sigma_f'] - 1

    def deltaiss_residuals(self) -> Tensor:
        """
        Compute the residual of the {\delta}ISS constraint

        Returns
        -------
        Tensor
            v: the residual of the {\delta}ISS constraint
        """
        if self.gate_activation is not torch.sigmoid or self.input_activation is not torch.tanh:
            raise ValueError(
                "The {\delta}ISS condition is valid when `gate_activation` is set to `torch.sigmoid` and `input_activation` is set to `torch.tanh`!"
            )
        norms = self._get_norms(p=np.inf)

        return norms["Ur_norm"] * (0.25 * norms["Uf_norm"] + norms["sigma_f"]) - 1 \
                    + 0.25 * (1 + norms["phi_r"]) / (1 - norms["sigma_z"]) * norms["Uz_norm"]


class StateSpaceLSTM(StateSpaceRecurrentLayer):
    __constants__ = ['units', 'in_features', 'io_delay', 'gate_activation', 'input_activation',
                     'state_activation', 'trainable', 'origin_equilibrium', 'name']

    units: int
    in_features: int
    io_delay: bool
    gate_activation: Callable
    input_activation: Callable
    state_activation: Callable
    trainable: bool
    origin_equilibrium: bool
    name: str

    def __init__(
        self,
        units: int,
        in_features: int,
        io_delay: bool = True,
        gate_activation: Callable | str = torch.sigmoid,
        input_activation: Callable | str = torch.tanh,
        state_activation: Callable | str = torch.tanh,
        trainable: bool = True,
        origin_equilibrium: bool = False,
        init_input: Callable = None,
        init_kernel: Callable = None,
        init_bias: Callable = None,
        name: str = "LSTM",
    ):
        """
        State Space Long Short-Term Memory network

        Parameters
        ----------
        units : int
            Number of neurons of the GRU
        in_features : int
            Number of inputs
        io_delay : bool, optional
            Return the state at the previous time instant, by default True
        gate_activation : Callable, optional
            Activation function of the gate, by default `torch.sigmoid`
        input_activation : Callable, optional
            Activation function of the input, by default `torch.tanh`
        state_activation : Callable, optional
            Activation function of the state-update law, by default `torch.tanh`
        trainable : bool, optional
            Enable the training of the network, by default True
        origin_equilibrium : bool, optional
            Enforce the origin (u=0) to be an equilibrium of the system, by default False
        init_input : Callable, optional
            The in-place initializer of the input weights, by default None (`uniform_`)
        init_kernel : Callable, optional
            The in-place initializer of the kernel weights, by default None (`uniform_`)
        init_bias : Callable, optional
            The in-place initializer of the biases, by default None (`zeros_`)
        name : str, optional
            Name of the LSTM
        """
        super(StateSpaceLSTM, self).__init__(name, init_input=init_input, init_kernel=init_kernel, init_bias=init_bias)
        self.units = units
        self.in_features = in_features
        self.io_delay = io_delay
        self.gate_activation = getattr(torch, gate_activation) if isinstance(gate_activation, str) else gate_activation 
        self.state_activation = getattr(torch, state_activation) if isinstance(state_activation, str) else state_activation 
        self.input_activation = getattr(torch, input_activation) if isinstance(input_activation, str) else input_activation 
        self.trainable = trainable
        self.origin_equilibrium = origin_equilibrium

        self._n_gates = 4

        self.Wfio = Parameter(torch.zeros(in_features, (self._n_gates - 1) * units), requires_grad=trainable)
        self.Ufio = Parameter(torch.zeros(units, (self._n_gates - 1) * units), requires_grad=trainable)
        self.bfio = Parameter(torch.zeros((self._n_gates - 1) * units), requires_grad=trainable)

        self.Wc = Parameter(torch.zeros(in_features, units), requires_grad=trainable)
        self.Uc = Parameter(torch.zeros(units, units), requires_grad=trainable)
        self.bc = Parameter(torch.zeros(units), requires_grad=trainable and not origin_equilibrium)

        self.reset_parameters()

        if origin_equilibrium:
            # Make sure that the c gate has null bias
            torch.zero_(self.bc)

    def extra_repr(self) -> str:
        return (
            f"units={self.units}, in_features={self.in_features}, io_delay={self.io_delay}, "
            f"origin_equilibrium={self.origin_equilibrium}, gate_activation={self.gate_activation.__name__}, "
            f"state_activation={self.state_activation.__name__},  input_activation={self.state_activation.__name__}, "
            f"trainable={self.trainable}"
        )

    @property
    def state_size(self) -> int:
        """
        Retrieve the number of states

        Returns
        -------
        int
            Number of states
        """
        return 2 * self.units

    def forward(self, u: Tensor, x0: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward-pass of the LSTM

        Note that the LSTM state vector contains both the hidden state and the cell state.

        Parameters
        ----------
        u : Tensor
            Input u(k) of the LSTM layer, with shape `(batches, inputs)`
        x0 : Tensor, optional
            Current state x(k) of the LSTM, with shape `(batches, states)`, where `states = 2*n`, by default None

        Returns
        -------
        Tuple[Tensor, Tensor]
            y : the output y(k), with shape `(batches, outputs)`
            xp : the new state x(k+1), with shape `(batches, states)`
        """
        if x0 is None:
            x0 = self.initial_states(u.shape[-2])

        # h denotes the hidden state
        # xi denotes the cell state
        # (shape: (batch_size, n_units)) each
        h0, xi0 = torch.chunk(x0, chunks=2, dim=1)
        if xi0.shape[1] != self.units:
            raise ValueError("Dimensionality of the state not correct")

        # Evaluate the gates
        gates_sig = self.gate_activation(torch.matmul(u, self.Wfio) + torch.matmul(xi0, self.Ufio) + self.bfio)  # (shape: (batch_size, 3*n_units))
        # (shape: (batch_size, n_units))
        c = self.input_activation(torch.matmul(u, self.Wc) + torch.matmul(xi0, self.Uc) + self.bc)
        # (shape: (batch_size, n_units)) each
        f, i, o = torch.chunk(gates_sig, self._n_gates - 1, dim=1)

        # LSTM equations
        # h(k+1) = f_h(x(k), u(k))
        hp = f * h0 + i * c  # (shape: (batch_size, n_units))

        # xi(k+1) = f_xi(x(k), u(k))
        
        xip = o * self.state_activation(hp)  # (shape: (batch_size, n_units))

        # Full LSTM state
        xp = torch.cat([hp, xip], dim=1) # (shape: (batch_size, 2*n_units))
        y = xi0 if self.io_delay else xip  # (shape: (batch_size, n_units))

        return y, xp

    def _get_norms(self, p: int = np.inf) -> Tuple[
        Tuple[Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        """
        [summary]

        Parameters
        ----------
        ord : int, optional
            [description], by default np.inf

        Returns
        -------
        Dict
            (sigma_f, sigma_i, sigma_o, phi_c) : The bounds of the gates
            (Wf_norm, Wi_norm, Wo_norm, Wc_norm) : The norms of the W matrices
            (Uf_norm, Ui_norm, Uo_norm, Uc_norm) : The norms of the U matrices
        """
        Wc = self.Wc.t()
        Uc = self.Uc.t()
        bc = self.bc

        Wf, Wi, Wo = torch.chunk(self.Wfio, self._n_gates - 1, dim=1)
        Uf, Ui, Uo = torch.chunk(self.Ufio, self._n_gates - 1, dim=1)
        bf, bi, bo = torch.chunk(self.bfio, self._n_gates - 1, dim=0)

        Wf, Wi, Wo = Wf.t(), Wi.t(), Wo.t()
        Uf, Ui, Uo = Uf.t(), Ui.t(), Uo.t()

        sigma_f, Wf_norm, Uf_norm = self._gate_bounds(Wf, Uf, bf, self.gate_activation, p=p)
        sigma_i, Wi_norm, Ui_norm = self._gate_bounds(Wi, Ui, bi, self.gate_activation, p=p)
        sigma_o, Wo_norm, Uo_norm = self._gate_bounds(Wo, Uo, bo, self.gate_activation, p=p)
        phi_c, Wc_norm, Uc_norm = self._gate_bounds(Wc, Uc, bc, self.state_activation, p=p)

        return dict_of(
            sigma_f,
            sigma_i,
            sigma_o,
            phi_c,
            Wf_norm,
            Wi_norm,
            Wo_norm,
            Wc_norm,
            Uf_norm,
            Ui_norm,
            Uo_norm,
            Uc_norm,
        )

    def iss_residuals(self) -> Tensor:
        """
        Compute the residual of the ISS constraint

        Returns
        -------
        Tensor
            v: the residual of the ISS constraint residuals
        """
        if self.gate_activation is not torch.sigmoid or self.state_activation is not torch.tanh:
            raise ValueError(
                "The ISS condition is valid when `gate_activation` is set to `torch.sigmoid` and `state_activation` is set to `torch.tanh`!"
            )

        norms = self._get_norms(p=2)

        return norms["sigma_f"] + norms["sigma_o"] * norms["sigma_i"] * norms["Uc_norm"] - 1
        
    def deltaiss_residuals(self) -> Tensor:
        """
        Compute the residual of the {\delta}ISS constraint

        Returns
        -------
        Tuple[Tensor, Tensor]
            (v1, v2): the residual of the {\delta}ISS constraint residuals
        """
        if self.gate_activation is not torch.sigmoid or self.state_activation is not torch.tanh:
            raise ValueError(
                "The {\delta}ISS condition is valid when `gate_activation` is set to `torch.sigmoid` and `state_activation` is set to `torch.tanh`!"
            )

        norms = self._get_norms(p=2)

        phi_x = self.state_activation(norms["sigma_i"] * norms["phi_c"] / (1 - norms["sigma_f"])
    )
        alpha = 0.25 * norms["Uf_norm"] * norms["sigma_i"] * norms["phi_c"] / (1 - norms["sigma_f"]) \
                + norms["sigma_i"] * norms["Uc_norm"] + 0.25 * norms["Ui_norm"] * norms["phi_c"]

        v1 = -1 + norms["sigma_f"] + alpha * norms["sigma_o"] + 0.25 * phi_x * norms["Uo_norm"] \
             - 0.25 * norms["sigma_f"] * phi_x * norms["Uo_norm"]
        v2 = 0.25 * norms["sigma_f"] * phi_x - 1

        return (v1, v2)


class StateSpaceNNARX(StateSpaceRecurrentLayer):
    __constants__ = ['units', 'in_features', 'out_features', 'horizon', 'input_feedthrough',
                     'activations', 'lipschitz', 'io_delay', 'trainable', 'origin_equilibrium', 'name']
    units: int
    in_features: int
    out_features: int
    horizon: int
    input_feedthrough: bool
    activations: List[Callable]
    lipschitz: List[float]
    io_delay: bool
    trainable: bool
    origin_equilibrium: bool
    name: str

    def __init__(
        self,
        units: List[int],
        in_features: int,
        out_features: int,
        horizon: int,
        input_feedthrough: bool = False,
        activations: List[Callable] | List[str] | Callable | str = torch.tanh,
        lipschitz: List[float] | float = None,
        io_delay: bool = True,
        trainable: bool = True,
        origin_equilibrium: bool = False,
        init_input: Callable = None,
        init_kernel: Callable = None,
        init_bias: Callable = None,
        name: str = "NNARX",
    ):
        """
        State Space Neural NARX network

        Parameters
        ----------
        units : List[int]
            List of the number of neurons of the FFNN
        in_features : int
            Number of inputs
        out_features : int
            Number of outputs
        input_feedthrough: bool, optional
            Supply the input vector u to all the layers of the FFNN, by default False
        activations : List[Callable], optional
            List containing the activation function of each layer. If a single activation function is supplied,
            it is applied to all layers. Note that the activation function f(x) must be such that f(0) = 0,
            and Lipschitz-continuous. By default `torch.sigmoid`
        lipschitz : List[float], optional
            List of the Lipschitz constants of the activation functions, only necessary for stability residuals computations.
            If omitted, and the activation functions are the canonical ones, it is inferred. By default None.
        io_delay : bool, optional
            Return the state at the previous time instant, by default True
        trainable : bool, optional
            Enable the training of the network, by default True
        origin_equilibrium : bool, optional
            Enforce the origin (u=0) to be an equilibrium of the system, by default False
        init_input : Callable, optional
            The in-place initializer of the input weights, by default None (`uniform_`)
        init_kernel : Callable, optional
            The in-place initializer of the kernel weights, by default None (`uniform_`)
        init_bias : Callable, optional
            The in-place initializer of the biases, by default None (`zeros_`)
        name : str, optional
            Name of the LSTM
        """
        super(StateSpaceNNARX, self).__init__(name, init_input=init_input, init_kernel=init_kernel, init_bias=init_bias)

        self.units = units if isinstance(units, Iterable) else [units]
        self.in_features = in_features
        self.out_features = out_features
        self.input_feedthrough = input_feedthrough
        self.io_delay = io_delay
        self.horizon = horizon
        self.N = horizon
        self.trainable = trainable
        self.origin_equilibrium = origin_equilibrium
        self.M = len(self.units)

        # Compute state size
        self._state_size = self.N * (self.in_features + self.out_features)
        
        if isinstance(activations, Iterable):
            if len(activations) == self.M:
                self.activations = [getattr(torch, a) if isinstance(a, str) else a for a in activations]
            else: 
                raise ValueError('Unexpected value for parameter activations')
        else:
            self.activations = [getattr(torch, activations) if isinstance(activations, str) else activations] * self.M


        # Check Lipschitz constants
        if isinstance(lipschitz, List) and len(lipschitz) == len(self.activations):
            self.lipschitz = lipschitz
        elif isinstance(lipschitz, float):
            self.lipschitz = [lipschitz] * self.M
        elif lipschitz is None:
            self.lipschitz = [0.0] * self.M
            for i, act in enumerate(self.activations):
                fname = act.__name__
                if fname == "tanh" or fname == "relu" or fname == "leaky_relu":
                    self.lipschitz[i] = 1.0
                elif fname == "sigmoid":
                    self.lipschitz[i] = 0.25
                else:
                    raise ValueError(
                        f"Unknown activation function {fname}, please specify lipschitz constants explicitly"
                    )
        else:
            raise ValueError('Unexpected value for parameter "lipschitz"')

        W = [None] * self.M
        U = [None] * (self.M if self.input_feedthrough else 1)
        b = [None] * self.M

        # Build parameters and save them in a ParameterList
        for i, n in enumerate(self.units):
            if i == 0:
                W[i] = Parameter(torch.zeros(self.in_features, n), requires_grad=trainable)
                U[i] = Parameter(torch.zeros(self._state_size, n), requires_grad=trainable)
                b[i] = Parameter(torch.zeros(n), requires_grad=trainable and not origin_equilibrium)
            else:
                U[i] = Parameter(torch.zeros(self.units[i - 1], n), requires_grad=trainable)
                b[i] = Parameter(torch.zeros(n), requires_grad=trainable and not origin_equilibrium)
                if self.input_feedthrough:
                    W[i] = Parameter(torch.zeros(self.in_features, n), requires_grad=trainable)

        self.W = ParameterList(W)
        self.U = ParameterList(U)
        self.b = ParameterList(b)
        self.U0 = Parameter(torch.zeros(self.units[-1], self.out_features), requires_grad=trainable)
        self.b0 = Parameter(torch.zeros(self.out_features), requires_grad=trainable and not origin_equilibrium)

        # Build the support matrices A, Bu, Bx, C
        _a = np.zeros((self.N, self.N))
        _a[range(0, self.N - 1), range(1, self.N)] = 1
        _b = np.zeros((self.N, 1))
        _b[-1, 0] = 1
        _bx = np.concatenate([np.eye(self.out_features), np.zeros([self.in_features, self.out_features])], axis=0)
        _bu = np.concatenate([np.zeros([self.out_features, self.in_features]), np.eye(self.in_features)], axis=0)

        _A = np.kron(_a, np.eye((self.in_features + self.out_features))).transpose()
        _Bu = np.kron(_b, _bu).transpose()
        _Bx = np.kron(_b, _bx).transpose()

        self._A = torch.from_numpy(_A).to(dtype=torch.float32)
        self._Bu = torch.from_numpy(_Bu).to(dtype=torch.float32)
        self._Bx = torch.from_numpy(_Bx).to(dtype=torch.float32)

        self.reset_parameters()

        if origin_equilibrium:
            # Make sure that the c gate has null bias
            torch.zero_(self.b0)
            for _b in self.b:
                torch.zero_(_b)

    def extra_repr(self) -> str:
        return (
            f"units={self.units}, N={self.N}, in_features={self.in_features}, "
            f"out_features={self.out_features}, io_delay={self.io_delay}, "
            f"origin_equilibrium={self.origin_equilibrium}, activations={[a.__name__ for a in self.activations]}, "
            f"input_feedthrough={self.input_feedthrough}, trainable={self.trainable}"
        )

    @property
    def state_size(self) -> int:
        """
        Retrieve the number of states

        Returns
        -------
        int
            Number of states
        """
        return self._state_size

    def forward(self, u: Tensor, x0: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward-pass of the NNARX

        Note that the LSTM state vector contains both the hidden state and the cell state.

        Parameters
        ----------
        u : Tensor
            Input u(k) of the NNARX, with shape `(batches, inputs)`
        x0 : Tensor, optional
            Current state x(k) of the NNARX, with shape `(batches, states)`, by default None

        Returns
        -------
        Tuple[Tensor, Tensor]
            y : the output y(k), with shape `(batches, outputs)`
            xp : the new state x(k+1), with shape `(batches, states)`
        """
        if x0 is None:
            x0 = self.initial_states(u.shape[-2])

        x = x0
        for i, _ in enumerate(self.U):
            if i > 0 and not self.input_feedthrough:
                x = self.activations[i](torch.matmul(x, self.U[i]) + self.b[i])
            else:
                x = self.activations[i](torch.matmul(u, self.W[i]) + torch.matmul(x, self.U[i]) + self.b[i])

        y = torch.matmul(x, self.U0) + self.b0
        xp = torch.matmul(x0, self._A) + torch.matmul(u, self._Bu) + torch.matmul(y, self._Bx)

        return y, xp

    def _get_norms(self) -> Tuple[List[Tensor], Tensor]:
        """
        Retrieve the norms of the U[i] matrices

        Returns
        -------
        Tuple[Tensor, Tensor]
            A Tensor containing all the norms of U matrices, and their product
        """
        U_norms_list = [torch.linalg.norm(Ui.t(), ord=2) for Ui in self.U] + \
            [torch.linalg.norm(self.U0.t(), ord=2)]
        U_norms = torch.tensor(U_norms_list)

        return U_norms, U_norms.prod()

    def iss_residuals(self) -> Tensor:
        """
        Compute the residual of the ISS constraint

        Returns
        -------
        Tensor
            v: the residual of the ISS constraint
        """
        _, U_prod = self._get_norms()
        return U_prod - 1 / (np.prod(self.lipschitz) * np.sqrt(self.N))

    def deltaiss_residuals(self) -> Tensor:
        """
        Compute the residual of the {\delta}ISS constraint

        Returns
        -------
        Tensor
            v: the residual of the {\delta}ISS constraint
        """
        return self.iss_residuals()

class PieceWiseRegularizer(nn.Module):
    def __init__(self, clearance=0.05, omega_plus: float = 1.0, omega_minus: float = 1e-4) -> None:
        super().__init__()
        self.clearance = clearance
        self.omega_plus = omega_plus
        self.omega_minus = omega_minus

    def _piecewise(self, x: torch.Tensor):
        return self.omega_plus * torch.max(torch.zeros_like(x), x + self.clearance) \
                - self.omega_minus * torch.min(torch.zeros_like(x), x + self.clearance)

    def forward(self, residuals: torch.Tensor | List[torch.Tensor]):
        if isinstance(residuals, List):
            losses = [self._piecewise(res) for res in residuals]
            return sum(losses)
        else:
            return self._piecewise(residuals)

class GeneralizedPieceWiseRegularizer(nn.Module):
    def __init__(self, clearance=0.05, omega_plus: float = 1.0, omega_minus: float = 1e-4, steepness: float = 2.5) -> None:
        super().__init__()
        self.clearance = clearance
        self.omega_plus = omega_plus
        self.omega_minus = omega_minus
        self.k = steepness

        assert clearance > 0, "Clearance must be positive"
        assert omega_plus > 0, "Positive gain must be positive"
        assert omega_minus > 0, "Negative gain must be positive"
        assert steepness > 0, "Steepness must be positive"

    def _smoothedpw(self, x: torch.Tensor):
        return self.omega_minus * (x + self.clearance) \
                + (self.omega_plus - self.omega_minus) / self.k * (torch.log(1 + torch.exp(self.k * (x + self.clearance))) - math.log(2.0))

    def forward(self, residuals: torch.Tensor | List[torch.Tensor]):
        if isinstance(residuals, List):
            losses = [self._smoothedpw(res) for res in residuals]
            return sum(losses)
        else:
            return self._smoothedpw(residuals)
