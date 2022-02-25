# -*- coding: utf-8 -*-
from bisect import bisect_right
from typing import Iterable as Iter

from .lr_scheduler import LRScheduler
from .optimizer import Optimizer


class MultiStepLR(LRScheduler):
    r"""Decays the learning rate of each parameter group by gamma once the
        number of epoch reaches one of the milestones.

    Args:
        optimizer: wrapped optimizer.
        milestones: list of epoch indices which should be increasing.
        gamma: multiplicative factor of learning rate decay. Default: 0.1
        current_epoch: the index of current epoch. Default: -1
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iter[int],
        gamma: float = 0.1,
        current_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(
                    milestones
                )
            )

        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, current_epoch)

    def state_dict(self):
        r"""Returns the state of the scheduler as a :class:`dict`.
            It contains an entry for every variable in self.__dict__ which
            is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key in ["milestones", "gamma", "current_epoch"]
        }

    def load_state_dict(self, state_dict):
        r"""Loads the schedulers state.

        Args:
          state_dict: scheduler state.
        """
        tmp_dict = {}
        for key in ["milestones", "gamma", "current_epoch"]:
            if not key in state_dict.keys():
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_dict[key] = state_dict[key]

        self.__dict__.update(tmp_dict)

    def get_lr(self):
        return [
            base_lr * self.gamma ** bisect_right(self.milestones, self.current_epoch)
            for base_lr in self.base_lrs
        ]
