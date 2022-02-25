# -*- coding: utf-8 -*-
from ..functional import dropout
from .module import Module


class Dropout(Module):
    r"""Randomly sets input elements to zeros with the probability :math:`drop\_prob` during training.
    Commonly used in large networks to prevent overfitting.
    Note that we perform dropout only during training, we also rescale(multiply) the output tensor
    by :math:`\frac{1}{1 - drop\_prob}`. During inference :class:`~.Dropout` is equal to :class:`~.module.identity.Identity`.

    Args:
        drop_prob: The probability to drop (set to zero) each single element
    """

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if self.training:
            return dropout(inputs, self.drop_prob, training=True)
        else:
            return inputs

    def _module_info_string(self) -> str:
        return "drop_prob={drop_prob}".format(drop_prob=self.drop_prob)
