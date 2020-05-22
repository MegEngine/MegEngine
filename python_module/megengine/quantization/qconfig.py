# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial

from ..module import Module
from .fake_quant import FakeQuantize
from .observer import ExponentialMovingAverageObserver, MinMaxObserver


class QConfig:
    """
    A config class indicating how to do quantize toward :class:`~.QATModule`'s
    ``activation``, ``weight`` and ``bias``.

    And ``fake_quant`` parameter to indicate

    See :meth:`~.QATModule.set_qconfig` for detail usage.

    :param inp_observer: interface to instantiate an :class:`~.Observer` indicating
        how to collect scales and zero_point of input.
    :param weight_observer: similar to ``inp_observer`` but toward weight.
    :param act_observer: similar to ``inp_observer`` but toward activation.
    :param fake_quant: interface to instantiate a :class:`~.FakeQuantize` indicating
        how to do fake_quant calculation. can be invoked multi times to get different
        instance for each target tensor, for better control on enable and disable.
    :param bias_fake_quant: similar to ``fake_quant``, but usually need to set ``dtype``
        in advance, for bias's dtype is unable to be inferred from observer.

    Examples:

    .. code-block::

        # Default EMA QConfig for QAT.
        ema_fakequant_qconfig = QConfig(
            inp_observer=ExponentialMovingAverageObserver,
            weight_observer=ExponentialMovingAverageObserver,
            act_observer=ExponentialMovingAverageObserver,
            fake_quant=FakeQuantize,
        )
    """

    def __init__(
        self, act_observer, weight_observer, inp_observer, fake_quant, bias_fake_quant,
    ):
        if (
            isinstance(act_observer, Module)
            or isinstance(weight_observer, Module)
            or isinstance(inp_observer, Module)
        ):
            raise ValueError(
                "QConfig must not receive observer instance, please pass observer"
                " class generator using `partial(Observer, ...)` instead. Use"
                " partial(MyObserver, x=1) to override arguments to constructor if needed"
            )
        self.act_observer = act_observer
        self.weight_observer = weight_observer
        self.inp_observer = inp_observer
        self.fake_quant = fake_quant
        self.bias_fake_quant = bias_fake_quant


# Default QAT QConfigs
min_max_fakequant_qconfig = QConfig(
    inp_observer=MinMaxObserver,
    weight_observer=MinMaxObserver,
    act_observer=MinMaxObserver,
    fake_quant=FakeQuantize,
    bias_fake_quant=partial(FakeQuantize, dtype="qint32"),
)

ema_fakequant_qconfig = QConfig(
    inp_observer=ExponentialMovingAverageObserver,
    weight_observer=MinMaxObserver,
    act_observer=ExponentialMovingAverageObserver,
    fake_quant=FakeQuantize,
    bias_fake_quant=partial(FakeQuantize, dtype="qint32"),
)
