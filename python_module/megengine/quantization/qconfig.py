# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#'
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..module import Module
from .fake_quant import TQT, FakeQuantize
from .observer import (
    ExponentialMovingAverageObserver,
    HistogramObserver,
    MinMaxObserver,
)


class QConfig:
    r"""
    A config class indicating how to do quantize toward :class:`~.QATModule`'s
    ``activation`` and ``weight``. See :meth:`~.QATModule.set_qconfig` for detail usage.

    :param weight_observer: interface to instantiate an :class:`~.Observer` indicating
        how to collect scales and zero_point of wegiht.
    :param act_observer: similar to ``weight_observer`` but toward activation.
    :param fake_quant: interface to instantiate a :class:`~.FakeQuantize` indicating
        how to do fake_quant calculation. can be invoked multi times to get different
        instance for each target tensor, for better control on enable and disable.

    Examples:

    .. code-block::

        # Default EMA QConfig for QAT.
        ema_fakequant_qconfig = QConfig(
            weight_observer=MinMaxObserver,
            act_observer=ExponentialMovingAverageObserver,
            fake_quant=FakeQuantize,
        )
    """

    def __init__(
        self, act_observer, weight_observer, fake_quant,
    ):
        if isinstance(act_observer, Module) or isinstance(weight_observer, Module):
            raise ValueError(
                "QConfig must not receive observer instance, please pass observer"
                " class generator using `partial(Observer, ...)` instead. Use"
                " partial(MyObserver, x=1) to override arguments to constructor if needed"
            )
        self.act_observer = act_observer
        self.weight_observer = weight_observer
        self.fake_quant = fake_quant


tqt_quant_qconfig = QConfig(
    weight_observer=ExponentialMovingAverageObserver,
    act_observer=ExponentialMovingAverageObserver,
    fake_quant=TQT,
)

# Default QAT QConfigs
min_max_fakequant_qconfig = QConfig(
    weight_observer=MinMaxObserver,
    act_observer=MinMaxObserver,
    fake_quant=FakeQuantize,
)

ema_fakequant_qconfig = QConfig(
    weight_observer=MinMaxObserver,
    act_observer=ExponentialMovingAverageObserver,
    fake_quant=FakeQuantize,
)

calibration_qconfig = QConfig(
    weight_observer=MinMaxObserver, act_observer=HistogramObserver, fake_quant=None,
)
