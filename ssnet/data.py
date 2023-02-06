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

import abc
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils import data

TensorExperimentData = List[Tuple[torch.Tensor, torch.Tensor]] 
ExperimentData = List[Tuple[torch.Tensor, torch.Tensor]] | List[Tuple[np.ndarray, np.ndarray]]

@dataclass
class SequenceScaler:
    """
    A SequenceScaler allows to normalize and denormalize the data feature-wise.
    """
    __constants__ = ['bias', 'scale']
    bias: torch.Tensor = None
    scale: torch.Tensor = None

    def __post_init__(self):
        if self.bias is not None and not isinstance(self.bias, torch.Tensor):
            self.bias = torch.tensor(self.bias)
        if self.scale is not None and not isinstance(self.scale, torch.Tensor):
            self.scale = torch.tensor(self.scale)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize the data
        """
        return (data - self.bias) / self.scale 
    
    def denormalize(self, ndata: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the data
        """
        return ndata * self.scale + self.bias
        
    def normalize_(self, data: torch.Tensor):
        """
        Normalize the data (in-place)
        """
        data.sub_(self.bias).div_(self.scale) 
    
    def denormalize_(self, ndata: torch.Tensor):
        """
        Denormalize the data (in-place)
        """
        ndata.mul_(self.scale).add_(self.bias)


class EmpiricalScaler(abc.ABC):
    """
    Abstract Scaler where the bias and scale are computed from data.
    """
    @abc.abstractclassmethod
    def from_data(self, X: torch.Tensor):
        pass

@dataclass
class MeanSequenceScaler(SequenceScaler, EmpiricalScaler):
    """
    Scaler having as bias the mean of the observed data, 
    and as scale the maximum deviation from the bias.
    """
    __constants__ =  SequenceScaler.__constants__ + ['rho', 'eps']
    rho: float = 0.0    # Scaling factor
    eps: float = 1e-5   # Avoid division by zero

    def from_data(self, X: torch.Tensor):
        dims = list(range(0, X.ndim-1))
        self.bias = X.mean(dim=dims)
        self.scale = (X - self.bias).abs().amax(dim=dims) * (1.0 + self.rho) + self.eps

@dataclass
class MinMaxSequenceScaler(SequenceScaler, EmpiricalScaler):
    """
    Scaler having as bias the median of the observed data,
    and as scale the maximum deviation from the median.
    """
    __constants__ = SequenceScaler.__constants__ + ['rho', 'eps']
    rho: float = 0.0    # Scaling factor
    eps: float = 1e-5   # Avoid division by zero

    def from_data(self, X: torch.Tensor):
        dims = list(range(0, X.ndim-1))
        self.bias = (X.amax(dim=dims) + X.amin(dim=dims)) / 2.0
        self.scale = (X - self.bias).abs().amax(dim=dims) * (1.0 + self.rho) + self.eps


@dataclass
class FullSequenceDataset:
    """
    Full Dataset of Sequences. 
    
    Contains
    ----------
    training: TensorDataset
        The TensorDataset storing the training subsequences
    validation: TensorDataset
        The TensorDataset storing the validation subsequences
    testing: TensorDataset, optional
        The optional TensorDataset storing the testing subsequences
    input_scaler: SequenceScaler, optional
        The scaler of the input sequence
    output_scaler: SequenceScaler, optional
        The scaler of the output sequence
    """
    training: data.TensorDataset
    validation: data.TensorDataset
    testing: data.TensorDataset = None

    input_scaler: SequenceScaler = None
    output_scaler: SequenceScaler = None

def tbptt(training: ExperimentData, 
          validation: ExperimentData, 
          Ns_train: int,
          Ts_train: int, 
          Ns_val: int = -1, 
          Ts_val: int = -1,
          testing: ExperimentData = None, 
          Ns_test: int = -1,
          Ts_test: int = -1,
          input_scaler: SequenceScaler = None, 
          output_scaler: SequenceScaler = None) -> FullSequenceDataset:
    """
    Extract subsequences from experiments via Truncated Back-Propagation Through Time

    Parameters
    ----------
    training : ExperimentData
        The training experiments. List of (U, Y) tuples. Each element of the list is a single experiment.
    validation : ExperimentData
        The validation experiments. List of (U, Y) tuples. Each element of the list is a single experiment.
    Ns_train : int
        The number of subsequences we want to extract for training.
    Ts_train : int
        The length of each training subsequence.
    Ns_val : int, optional
        The number of subsequences we want to extract for validaiton. 
        If -1, the validation sequences are not split in subsequences.
    Ts_val : int, optional
        The length of each validaiton subsequence. 
        If -1, the validation sequences are not split in subsequences.
    testing : ExperimentData, optional
        The testing experiments. List of (U, Y) tuples. Each element of the list is a single experiment.
        If None, no testing data is considered.
    Ns_test : int, optional
        The number of subsequences we want to extract for testing. 
        If -1, the testing sequences are not split in subsequences.
    Ts_test : int, optional
        The length of each testing subsequence. 
        If -1, the testing sequences are not split in subsequences.
    input_scaler : SequenceScaler, optional
        The SequenceScaler we want to use for scaling the input sequences U, by default None
    output_scaler : SequenceScaler, optional
        The SequenceScaler we want to use for scaling the output sequences Y, by default None

    Returns
    -------
    SequenceRawDataset
        The Dataset obtained via TBPTT.
    """

    def sample_initial_instants(experiments: TensorExperimentData, 
                                Ts: int, 
                                N_seq: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auxiliary function that returns random initial instants for the subsequences.

        Parameters
        ----------
        experiments : TensorExperimentData
            Set of experiments.
        Ts : int
            Length of the subsequences to extract.
        N_seq : int
            Number of subsequences to extract.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The first Tensor correspond to the experiment id. 
            The second Tensor correspond to the initial time instant.
        """
        T_exp = [u.shape[0] for u, _ in experiments]

        # If Ts < 0 and N_seq < N_exp, we assume that the initial steps of the experiments should be returned
        if (Ts < 0 or all([Ts >= t for t in T_exp])) and N_seq <= len(experiments):
            sampled_experiments = torch.arange(len(experiments), dtype=torch.long)
            sampled_initial_inst = torch.zeros_like(sampled_experiments)
            return sampled_experiments, sampled_initial_inst

        # Range of admissible initial points
        T_range = [t - Ts for t in T_exp]

        sampled_experiments = torch.tensor(list(data.WeightedRandomSampler(T_range, num_samples=N_seq, replacement=True)),
                                           dtype=torch.long)
        sampled_initial_inst = torch.empty_like(sampled_experiments)

        for e, n_e in zip(*sampled_experiments.unique(return_counts=True)):
            initial_indexes = data.RandomSampler(torch.arange(0, T_range[e]), num_samples=n_e.item())
            initial_indexes = torch.tensor(list(initial_indexes), dtype=torch.long)
            sampled_initial_inst[sampled_experiments == e] = initial_indexes

        return sampled_experiments, sampled_initial_inst

    def to_list_tensors(experiments: ExperimentData) -> TensorExperimentData:
        """
        Preprocess the experiment data to cast it into a List of Tuple[torch.Tensor, torch.Tensor]

        Parameters
        ----------
        experiments : ExperimentData
            The experiment data.

        Returns
        -------
        TensorExperimentData
            The preprocessed experiment data.
        """
        
        exps = [experiments] if not isinstance(experiments, List) else experiments
        exps = [(torch.tensor(e[0], dtype=torch.float32).clone(), torch.tensor(e[1], dtype=torch.float32).clone()) for e in exps]
        return exps

    # Preprocess the experimental data
    train_list = to_list_tensors(training)
    val_list = to_list_tensors(validation)
    test_list = to_list_tensors(testing) if testing else None
    
    # Set of all experiments
    experiments = train_list + val_list + test_list if test_list else train_list + val_list

    # Concatenate experiments with the only goal of scaling the data
    U = torch.cat([u for u, _ in experiments], dim=0)
    Y = torch.cat([y for _, y in experiments], dim=0)

    if input_scaler:
        if isinstance(input_scaler, EmpiricalScaler):
            input_scaler.from_data(U)
        for u_ex, _ in experiments:
            input_scaler.normalize_(u_ex)

    if output_scaler:
        if isinstance(output_scaler, EmpiricalScaler):
            output_scaler.from_data(Y)
        for _, y_ex in experiments:
            output_scaler.normalize_(y_ex)

    # Randomly sample initial instants for the TBPTT windows 
    sampled_train_exp, sampled_train_idx = sample_initial_instants(train_list, Ts=Ts_train, N_seq=Ns_train)
    sampled_val_exp, sampled_val_idx = sample_initial_instants(val_list, Ts=Ts_val, N_seq=Ns_val)
    if testing:
        sampled_test_exp, sampled_test_idx = sample_initial_instants(test_list, Ts=Ts_test, N_seq=Ns_test)
    
    # Build:
    # - Training dataset (U_train, Y_train)
    # - Validation dataset (U_val, Y_val)
    # - Testing dataset, if any, (U_test, Y_test)
    Ntr = sampled_train_exp.shape[0]
    Nva = sampled_val_exp.shape[0]
    Ttr = Ts_train
    Tva = min([t.shape[0] for t, _ in val_list]) if Ts_val < 0 else Ts_val

    n_in = train_list[0][0].shape[1]
    n_out = train_list[0][1].shape[1]
    
    U_train = torch.zeros(Ntr, Ttr, n_in)
    Y_train = torch.zeros(Ntr, Ttr, n_out)
    U_val = torch.zeros(Nva, Tva, n_in)
    Y_val = torch.zeros(Nva, Tva, n_out)

    # TODO: Avoid the for loop?
    for i, (exp, idx) in enumerate(zip(sampled_train_exp, sampled_train_idx)):
        U_train[i] = train_list[exp][0][idx:idx+Ttr]
        Y_train[i] = train_list[exp][1][idx:idx+Ttr]

    for i, (exp, idx) in enumerate(zip(sampled_val_exp, sampled_val_idx)):
        U_val[i] = val_list[exp][0][idx:idx+Tva]
        Y_val[i] = val_list[exp][1][idx:idx+Tva]

    if testing:
        Nte = sampled_test_exp.shape[0]
        Tte = min([t.shape[0] for t, _ in test_list]) if testing and Ts_test < 0 else Ts_test
        U_test = torch.zeros(Nte, Tte, n_in)
        Y_test = torch.zeros(Nte, Tte, n_out)
        
        for i, (exp, idx) in enumerate(zip(sampled_test_exp, sampled_test_idx)):
            U_test[i] = test_list[exp][0][idx:idx+Tte]
            Y_test[i] = test_list[exp][1][idx:idx+Tte]
            
        return FullSequenceDataset(training=data.TensorDataset(U_train, Y_train), 
                                   validation=data.TensorDataset(U_val, Y_val),
                                   testing=data.TensorDataset(U_test, Y_test),
                                   input_scaler=input_scaler,
                                   output_scaler=output_scaler)
    else:
        return FullSequenceDataset(training=data.TensorDataset(U_train, Y_train), 
                                   validation=data.TensorDataset(U_val, Y_val),
                                   input_scaler=input_scaler,
                                   output_scaler=output_scaler)
