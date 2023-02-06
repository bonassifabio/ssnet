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

import copy
import signal
from dataclasses import dataclass
from datetime import datetime
from math import inf
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import ttictoc
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import MatFileWriter

if TYPE_CHECKING:
    from .nn import StateSpaceNN

@dataclass()
class PostEpochTrainingData():
    epoch: int
    train_loss: float
    train_metric: float
    val_metric: float
    iss_residuals: List[float | torch.Tensor]
    deltaiss_residuals: List[float | torch.Tensor]
    enforce_iss: bool
    enforce_deltaiss: bool
    metric_fcn: Callable = nn.functional.mse_loss
    washout: int = 0

class TrainingCallback:

    def __init__(self) -> None:
        self.tensorboard: SummaryWriter = None
        self.matfile: MatFileWriter = None
        self.log: dict = None
        return

    def attach_tensorboard(self, tensorboard_instance: SummaryWriter):
        self.tensorboard = tensorboard_instance

    def attach_logger(self, log: dict):
        self.log = log

    def attach_matfile(self, matfile_instance: MatFileWriter):
        self.matfile = matfile_instance

    def on_training_start(self, net: StateSpaceNN):
        pass

    def on_epoch_start(self, net: StateSpaceNN, epoch: int):
        pass 
    
    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN):
        return False

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> dict:
        return {}


class CallbacksWrapper(TrainingCallback):

    def __init__(self, 
                 tensorboard_instance: SummaryWriter | str | Path = None, 
                 matfile_instance: MatFileWriter | str | Path = None,
                 callbacks: List[TrainingCallback] = None):
        super().__init__()
        
        self.to_close_matfile = False

        if tensorboard_instance and isinstance(tensorboard_instance, SummaryWriter):
            self.tensorboard = tensorboard_instance
        elif tensorboard_instance and isinstance(tensorboard_instance, (str, Path)):
            now = datetime.now().strftime('%Y%m%d-%H%M%S')
            tbpath = Path(tensorboard_instance).joinpath(now)
            self.tensorboard = SummaryWriter(tbpath)
        elif tensorboard_instance:
            raise ValueError('Incorrect tensorboard instance')
        
        if matfile_instance and isinstance(matfile_instance, MatFileWriter):
            self.matfile = matfile_instance
        if matfile_instance and isinstance(matfile_instance, (str, Path)):
            self.matfile = MatFileWriter(matfile_instance)
            self.to_close_matfile = True
        elif matfile_instance:
            raise ValueError('Incorrect MatFileWriter instance')

        self.callbacks = callbacks if callbacks else []
        self.log = {}

        for cbk in self.callbacks:
            cbk.attach_tensorboard(self.tensorboard)
            cbk.attach_logger(self.log)
            cbk.attach_matfile(self.matfile)

    def on_training_start(self, net: StateSpaceNN):
        if len(self.callbacks) == 0:
            return
        
        for cbk in self.callbacks:
            cbk.on_training_start(net)

    def on_epoch_start(self, net: StateSpaceNN, epoch: int):
        if len(self.callbacks) == 0:
            return
        
        for cbk in self.callbacks:
            cbk.on_epoch_start(net, epoch)

    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> bool:
        if len(self.callbacks) == 0:
            return False
        
        cbk_results = [ cbk.on_epoch_end(data, net) for cbk in self.callbacks ]
        if any(cbk_results):
            fired_callback = self.callbacks[[i for i, j in enumerate(cbk_results) if j][0]]
            print(f'\n{fired_callback.__class__.__name__} requested to orderly stop the training procedure.')
            return True

    def on_training_end(self, data: TrainingCallback, net: StateSpaceNN) -> dict:
        if len(self.callbacks) == 0:
            return
        
        outputs = {}
        for cbk in self.callbacks:
            out = cbk.on_training_end(data, net)
            outputs |= out

        if self.tensorboard:
            self.tensorboard.close()
        if self.matfile and self.to_close_matfile:
            self.matfile.close()

        return outputs


class LoggingCallback(TrainingCallback):

    def __init__(self) -> None:
        """
        Callback which logs the relevant informations during training

        Parameters
        ----------
        tensorboard_instance : SummaryWriter or str
            Instance of the TensorBoard writer (or path the folder)
        earlystopping : EarlyStoppingCallback, optional
            An EarlyStoppingCallback instance, which allows to keep track of the evolution of the best epochs, by default None
        net: StateSpaceNN, optional
            Instance of the StateSpaceNN being logged, by default None
        """
        super().__init__()

    def on_training_start(self, net: StateSpaceNN):
        if self.log is None:
            self.log = {}

        self.log['train_loss'] = []
        self.log['train_metric'] = []
        self.log['val_metric'] = []
        self.log['iss_residuals'] = []
        self.log['deltaiss_residuals'] = []
        self.log['epoch_time'] = []

        self.timer = ttictoc.Timer()

        if self.tensorboard is None:
            raise ValueError(f'{self.__class__.__name__} must belong to a {CallbacksWrapper.__class__} instance')
        if net is not None:
            self.tensorboard.add_text('Architecture', text_string=str(net))

    def _list2dict(self, x: List):
        keys = [str(k) for k, j in enumerate(x)]
        return dict(zip(keys, x))

    def on_epoch_start(self, net: StateSpaceNN, epoch: int):
        self.timer.start()

    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> bool:
        """
        This callback does not trigger the early stopping.
        """
        self.log['train_loss'].append(data.train_loss)
        self.log['val_metric'].append(data.val_metric)
        self.log['train_metric'].append(data.train_metric)
        self.log['iss_residuals'].append(data.iss_residuals)
        self.log['deltaiss_residuals'].append(data.deltaiss_residuals)
        self.log['epoch_time'].append(self.timer.stop())

        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag='Metric/Validation', scalar_value=data.val_metric, global_step=data.epoch)
            self.tensorboard.add_scalar(tag='Metric/Training', scalar_value=data.train_metric, global_step=data.epoch)
            self.tensorboard.add_scalar(tag='Training Loss', scalar_value=data.train_loss, global_step=data.epoch)
            self.tensorboard.add_scalars(main_tag='Residuals/ISS', tag_scalar_dict=self._list2dict(data.iss_residuals), global_step=data.epoch)
            self.tensorboard.add_scalars(main_tag='Residuals/deltaISS', tag_scalar_dict=self._list2dict(data.deltaiss_residuals), global_step=data.epoch)

        return False

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> dict:
        if self.matfile is not None:
            self.matfile.push(training_log=self.log)
        
        with torch.no_grad():
            iss_residuals = [nu.detach().item() for nu in net.iss_residuals()]
            deltaiss_residuals = [nu.detach().item() for nu in net.deltaiss_residuals()]
            is_iss = all([nu < 0 for nu in iss_residuals])
            is_deltaiss = all([nu < 0 for nu in deltaiss_residuals])
        
        return {'final_train_metric': self.log['train_metric'][-1], 'final_val_metric': self.log['val_metric'][-1],
                'final_iss_residuals': iss_residuals, 'final_deltaiss_residuals': deltaiss_residuals,
                'is_iss': is_iss, 'is_deltaiss': is_deltaiss,
                'avg_epoch_time': sum(self.log['epoch_time']) / float(len(self.log['epoch_time'])), 
                'training_epochs': data.epoch}

class SigIntCallback(TrainingCallback):

    def __init__(self) -> None:
        """
        Callback that detects the Ctrl+C event
        """
        super().__init__()

    def on_training_start(self, data: PostEpochTrainingData):
        self._sigint = False

        # Register the SigInt receiver
        signal.signal(signal.SIGINT, self.sigint_receiver)

    def sigint_receiver(self, sig, frame):
        self._sigint = True
    
    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> bool:
        """
        Returns True if the stopping condition is met
        """
        return self._sigint

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> dict:
        output = {'user_interrupt': self._sigint}
        self._sigint = False
        return output


class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience: int = 100, save_best: bool = True, watch_from: int = 0) -> None:
        """
        Early Stopping Callback. Stops the training procedure when the performances of the network stop improving.

        Parameters
        ----------
        patience : int, optional
            Number of epochs after which training is halt, by default 100
        save_best : bool, optional
            Save the best weights, by default True
        watch_from : int, optional
            Start the early stopping from this epoch, by default 0
        """
        super().__init__()
        self.patience = patience
        self.watch_from = watch_from 
        self.save_best = save_best

        #  Best global performances are evaluated without considering the stability conditions
        self._overall_best_epoch = None
        self._overall_best_metric = inf
        # Best stable performances are evaluated considering only stable epochs
        self._best_stable_epoch = None
        self._best_stable_metric = inf
        self._best_stable_weights = None
        self.es_fired = False

    @property
    def best_epoch(self):
        return self._best_stable_epoch
    
    @property
    def best_metric(self): 
        return self._best_stable_metric

    def on_training_start(self, net: StateSpaceNN):
        if self.log is None:
            self.log = {}

        self.log['best_stable_epoch'] = []
        self.log['best_stable_metric'] = []
        self.log['overall_best_epoch'] = []
        self.log['overall_best_metric'] = []

    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> bool:
        """
        Returns True if the stopping criterion is met.
        """
        iss_satisfied = not data.enforce_iss or all([r < 0.0 for r in data.iss_residuals])
        deltaiss_satisfied = not data.enforce_deltaiss or all([r < 0.0 for r in data.deltaiss_residuals])

        if data.val_metric < self._overall_best_metric:
            self._overall_best_epoch = data.epoch
            self._overall_best_metric = data.val_metric

        if data.val_metric < self.best_metric and iss_satisfied and deltaiss_satisfied:
            self._best_stable_metric = data.val_metric
            self._best_stable_epoch = data.epoch
            self._best_stable_weights = copy.deepcopy(net.state_dict())
        
        # Logging
        self.log['overall_best_epoch'].append(self._overall_best_epoch if self._overall_best_epoch else -inf)
        self.log['overall_best_metric'].append(self._overall_best_metric)

        if data.enforce_iss or data.enforce_deltaiss:
            self.log['best_stable_epoch'].append(self.best_epoch if self.best_epoch else -inf)
            self.log['best_stable_metric'].append(self.best_metric)
        elif 'best_stable_epoch' in self.log:
            del self.log['best_stable_epoch']
            del self.log['best_stable_metric']

        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag='Best/Overall-Epoch', scalar_value=self._overall_best_epoch, global_step=data.epoch)
            self.tensorboard.add_scalar(tag='Best/Overall-Metric', scalar_value=self._overall_best_metric, global_step=data.epoch)
            if data.enforce_iss or data.enforce_deltaiss and self.best_epoch:
                self.tensorboard.add_scalar(tag='Best/Stable-Epoch', scalar_value=self.best_epoch, global_step=data.epoch)
                self.tensorboard.add_scalar(tag='Best/Stable-Metric', scalar_value=self.best_metric, global_step=data.epoch)
        
        # Patience is evaluated considering the overall best epoch (not necessarily a stable epoch)
        self.es_fired = data.epoch >= self._overall_best_epoch + self.patience \
                            and self._best_stable_epoch \
                            and self._best_stable_epoch >= self.watch_from
        return self.es_fired

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> dict:
        output = {'early_stopping_fired': self.es_fired, 'overall_best_epoch': self._overall_best_epoch, 
                  'overall_best_metric': self._overall_best_metric}

        if self._best_stable_weights:
            print(f'Restoring best weight (epoch {self.best_epoch})...')
            net.load_state_dict(self._best_stable_weights)
            output |= {'best_stable_epoch': self.best_epoch, 'best_stable_metric': self.best_metric}
        else:
            print('No best weights found. Network weights will not be restored.')
            output |= {'best_stable_epoch': -1, 'best_stable_metric': -1}

        self.es_fired = False
        return output

class TargetMetricCallback(TrainingCallback):
    def __init__(self, target: float) -> None:
        """
        Stops the training procedure when the metric, evaluated on the validation set, reaches the target.

        Parameters
        ----------
        target : float
            The target value for the metric
        """
        super().__init__()
        self.target = target
        self.target_fired = False

    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> bool:
        """
        Returns True if the stopping criterion is met.
        """
        self.target_fired = data.val_metric <= self.target and \
                                (data.enforce_iss and all([r < 0.0 for r in data.iss_residuals]) or
                                data.enforce_deltaiss and all([r < 0.0 for r in data.deltaiss_residuals]))
        return self.target_fired

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> dict:
        output = {'target_metric_reached': self.target_fired}
        self.target_fired = False
        return output

class PerformanceTestingCallback(TrainingCallback):
    def __init__(self, test_loader: DataLoader, plot_fequency: int = None, dpi: float = 200, figsize: Tuple[float, float] = (5, 3)) -> None:
        """
        Test the performances of the model on an independent test set.

        Parameters
        ----------
        test_loader : DataLoader
            The DataLoader of the independent test set.
        plot_fequency : int, optional
            Frequency with which the performances should be plotted during training.
            By default None, meaning that performances are not tested during training.
        dpi : float, optional
            DPIs of the figure, by default 175
        figsize : Tuple[float, float], optional
            Size of the figure, in the format `(width, height)` in inches, by default (3, 2)
        """
        super().__init__()
        self.test_loader = test_loader
        self.plot_frequency = plot_fequency
        self.figsize = figsize
        self.dpi = dpi

    def _predict(self, net: StateSpaceNN):
        """
        Test the performances on one batch of the test loader
        """
        with net.evaluating() and torch.no_grad():
            u_test, y_test = next(iter(self.test_loader))
            y_hat, _ = net(u_test)

        return y_hat, y_test, u_test

    def _compute_FIT(self, y_hat: torch.Tensor, y_test: torch.Tensor) -> np.ndarray:
        mean_over = list(range(0, y_test.ndim - 1))
        mean_gt = y_test.mean(dim=mean_over, keepdim=True)
        sim_err = torch.linalg.vector_norm(y_hat - y_test, ord=2, dim=-1)
        y_ampl = torch.linalg.vector_norm(y_test - mean_gt, ord=2, dim=-1)
        err_dev = sim_err.sum() / y_ampl.sum()
        return (100 * (1 - err_dev)).detach().item()

    def _plot_test(self, data: PostEpochTrainingData, y_hat: torch.Tensor, y_test: torch.Tensor):
        """
        Plot the performances to a Matplotlib Figure
        """
        n_seq, _, n_out = y_hat.shape
        scaled_figsize = (self.figsize[0] * n_out, self.figsize[1] * n_seq)
        fig, axs = plt.subplots(nrows=n_seq, ncols=n_out, figsize=scaled_figsize, dpi=self.dpi, squeeze=False)

        y_hat_np = y_hat.detach().numpy()        
        y_test_np = y_test.detach().numpy()

        for j in range(0, n_seq):
            for i in range(0, n_out):
                cax = axs[j, i]
                cax.plot(y_test_np[j, :, i], 'b:')
                cax.plot(y_hat_np[j, :, i], 'r')
                # cax.legend(['Real', 'Prediction'])
                cax.set_title(f'Test - seq {j} out {i} - epoch {data.epoch}')

        fig.tight_layout()
        return fig

    def on_epoch_end(self, data: PostEpochTrainingData, net: StateSpaceNN):
        """
        Compute the performances on the testing dataset and, if necessary, plot them.
        """
        washout = data.washout
        y_hat, y_test, _ = self._predict(net)
        test_metric = data.metric_fcn(y_hat[:, washout:, :], y_test[:, washout:, :]).detach().item()
        fit_index = self._compute_FIT(y_hat, y_test)

        if self.tensorboard is not None:
            self.tensorboard.add_scalar('Metric/Test', scalar_value=test_metric, global_step=data.epoch)
            self.tensorboard.add_scalar('Metric/FIT', scalar_value=fit_index, global_step=data.epoch)
            
            if self.plot_frequency and self.plot_frequency > 0 and data.epoch % self.plot_frequency == 0:
                self.tensorboard.add_figure(tag='Test', figure=self._plot_test(data, y_hat, y_test),
                                            global_step=data.epoch, close=True)

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN) -> dict:
        """
        Final testing of the trained network. If a matfile instance is available, 
        save the trajectories there as well.
        """
        washout = data.washout
        y_hat, y_test, u_test = self._predict(net)
        test_metric = data.metric_fcn(y_hat[:, washout:, :], y_test[:, washout:, :]).detach().item()

        fit_index = self._compute_FIT(y_hat, y_test)

        if self.tensorboard is not None:
            self.tensorboard.add_scalar('Metric/Test', scalar_value=test_metric, global_step=data.epoch)
            self.tensorboard.add_scalar('Metric/FIT', scalar_value=fit_index, global_step=data.epoch)
            self.tensorboard.add_figure(tag='Test', figure=self._plot_test(data, y_hat, y_test), 
                                        global_step=data.epoch, close=True)

        if self.matfile:
            test_data = {'u_test_gt': u_test.clone().detach().numpy(),
                         'y_test_gt': y_test.clone().detach().numpy(),
                         'y_hat_nn': y_hat.clone().detach().numpy(),
                         'test_metric': test_metric,
                         'FIT': fit_index}
            self.matfile.push(testing=test_data)

        return {'final_test_metric': test_metric, 'FIT': fit_index}
        
class MatlabExportCallback(TrainingCallback):
    """
    Save the trained network to a Matlab file
    """
    def __init__(self):
        super().__init__()

    def on_training_end(self, data: PostEpochTrainingData, net: StateSpaceNN):
        if self.matfile:
            net.export_to_matlab(self.matfile)
        return {}
