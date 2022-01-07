# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .lr_scheduler import LRScheduler
from .optimizer import Optimizer


class CosineAnnealingLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule.
    
    Args:
        optimizer: wrapped optimizer.
        T_max: maximum number of iterations.
        eta_min: minimum learning rate. Default: 0.
    """

    def __init__(
        self, 
        optimizer: Optimizer,
        T_max: int,
        eta_min: float=0.0,
        current_epoch: int=-1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, current_epoch)

    def state_dict(self):
        r"""Returns the state of the scheduler as a :class:`dict`.
            It contains an entry for every variable in self.__dict__ which
            is not the optimizer.
        """
        return {
            key: value 
            for key, value in self.__dict__.items() 
            if key != 'optimizer' and 'base_lr'
        }

    def load_state_dict(self, state_dict):
        r"""Loads the schedulers state.

        Args:
            state_dict: scheduler state.
        """
        tmp_dict = {}
        for key in ["T_max", "eta_min", "current_epoch"]:
            if not key in state_dict.keys():
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_dict[key] = state_dict[key]

        self.__dict__.update(tmp_dict)

    def get_lr(self):
        if self.current_epoch == 0:
            return self.base_lrs
        elif (self.current_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.current_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.current_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]