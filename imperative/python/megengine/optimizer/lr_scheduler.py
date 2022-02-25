# -*- coding: utf-8 -*-
from abc import ABCMeta

from .optimizer import Optimizer


class LRScheduler(metaclass=ABCMeta):
    r"""Base class for all learning rate based schedulers.

    Args:
        optimizer: wrapped optimizer.
        current_epoch: the index of current epoch. Default: -1
    """

    def __init__(  # pylint: disable=too-many-branches
        self, optimizer: Optimizer, current_epoch: int = -1
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "optimizer argument given to the lr_scheduler should be Optimizer"
            )
        self.optimizer = optimizer
        self.current_epoch = current_epoch
        if current_epoch == -1:
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in "
                        "param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], self.optimizer.param_groups)
        )

        self.step()

    def state_dict(self):
        r"""Returns the state of the scheduler as a :class:`dict`.
            It contains an entry for every variable in self.__dict__ which
            is not the optimizer.
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        r"""Loads the schedulers state.

        Args:
            state_dict: scheduler state.
        """
        raise NotImplementedError

    def get_lr(self):
        r"""Compute current learning rate for the scheduler."""
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr
