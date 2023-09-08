# -*- coding: utf-8 -*-
from ..functional import dropout
from .module import Module


class Dropout(Module):
    r"""Randomly sets some elements of inputs to zeros with the probability :math:`drop\_prob` during training.
    Commonly used in large networks for regularization and prevent overfitting, see `Improving Neural Networks by Preventing Co-Adaptation of Feature Detectors<https://arxiv.org/abs/1207.0580>`.
    Note that we perform dropout only during training, we also rescale(multiply) the output tensor
    by :math:`\frac{1}{1 - drop\_prob}`. During inference :class:`~.Dropout` is equal to :class:`~.module.identity.Identity`.

    Args:
        drop_prob(:class:`float`): The probability to drop (set to zero) each single element. Default: 0.0

    Shape:
        - input: `(*)`. Input can be of any shape.
        - output: `(*)`. Output is of the same shape as input.
    
    Examples:
        >>> import numpy as np
        >>> data = Tensor(np.ones(10000000, dtype=np.float32))
        >>> out = F.nn.dropout(data, 1.0 / 3.0, training=True)
        >>> assert not out.numpy().all()
        >>> out = F.nn.dropout(data, 1.0 / 3.0, training=False)
        >>> assert out.numpy().all()
        >>> out.numpy()
        array([1., 1., 1., ..., 1., 1., 1.], dtype=float32)

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
