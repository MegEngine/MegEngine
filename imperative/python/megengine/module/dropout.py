# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..functional import dropout
from .module import Module


class Dropout(Module):
    r"""
    Randomly sets input elements to zeros with the probability :math:`drop\_prob` during training.
    Commonly used in large networks to prevent overfitting.
    Note that we perform dropout only during training, we also rescale(multiply) the output tensor
    by :math:`\frac{1}{1 - drop\_prob}`. During inference :class:`~.Dropout` is equal to :class:`~.Identity`.

    :param drop_prob: The probability to drop (set to zero) each single element
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        if self.training:
            return dropout(inputs, self.drop_prob, training=True)
        else:
            return inputs

    def _module_info_string(self) -> str:
        return "drop_prob={drop_prob}".format(drop_prob=self.drop_prob)
