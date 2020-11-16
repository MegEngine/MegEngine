# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#'
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial

from ..module import Module
from .fake_quant import TQT, FakeQuantize
from .observer import (
    ExponentialMovingAverageObserver,
    HistogramObserver,
    MinMaxObserver,
    SyncExponentialMovingAverageObserver,
    SyncMinMaxObserver,
)


class QConfig:
    r"""
    A config class indicating how to do quantize toward :class:`~.QATModule`'s
    ``activation`` and ``weight``. See :meth:`~.QATModule.set_qconfig` for detail usage.

    :param weight_observer: interface to instantiate an :class:`~.Observer` indicating
        how to collect scales and zero_point of wegiht.
    :param act_observer: similar to ``weight_observer`` but toward activation.
    :param weight_fake_quant: interface to instantiate a :class:`~.FakeQuantize` indicating
        how to do fake_quant calculation.
    :param act_observer: similar to ``weight_fake_quant`` but toward activation.

    Examples:

    .. code-block::

        # Default EMA QConfig for QAT.
        ema_fakequant_qconfig = QConfig(
            weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
            act_observer=partial(ExponentialMovingAverageObserver, dtype="qint8", narrow_range=False),
            weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
            act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
        )

    Each parameter is a ``class`` rather than an instance. And we recommand using ``functools.partial``
    to add initialization parameters of the ``class``, so that don't need to provide parameters in
    :meth:`~.QATModule.set_qconfig`.

    Usually we set ``narrow_range`` of weight related paramters to ``True`` and of activation related
    parameters to ``False``. For the result of multiplication and addition as ``a * b + c * d``, if
    four variables are all -128 of dtype ``qint8``, then the result will be ``2^15`` and cause overflow.
    Weights are commonly calculated in this way, so needed to narrow the range.
    """

    def __init__(
        self, weight_observer, act_observer, weight_fake_quant, act_fake_quant
    ):
        if isinstance(act_observer, Module) or isinstance(weight_observer, Module):
            raise ValueError(
                "QConfig must not receive observer instance, please pass observer"
                " class generator using `partial(Observer, ...)` instead. Use"
                " partial(MyObserver, x=1) to override arguments to constructor if needed"
            )
        self.weight_observer = weight_observer
        self.act_observer = act_observer
        self.weight_fake_quant = weight_fake_quant
        self.act_fake_quant = act_fake_quant


tqt_quant_qconfig = QConfig(
    weight_observer=partial(
        ExponentialMovingAverageObserver, dtype="qint8", narrow_range=True
    ),
    act_observer=partial(
        ExponentialMovingAverageObserver, dtype="qint8", narrow_range=False
    ),
    weight_fake_quant=partial(TQT, dtype="qint8", narrow_range=True),
    act_fake_quant=partial(TQT, dtype="qint8", narrow_range=False),
)

min_max_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
    act_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=False),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
    act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
)

ema_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
    act_observer=partial(
        ExponentialMovingAverageObserver, dtype="qint8", narrow_range=False
    ),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
    act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
)

sync_ema_fakequant_qconfig = QConfig(
    weight_observer=partial(SyncMinMaxObserver, dtype="qint8", narrow_range=True),
    act_observer=partial(
        SyncExponentialMovingAverageObserver, dtype="qint8", narrow_range=False
    ),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=True),
    act_fake_quant=partial(FakeQuantize, dtype="qint8", narrow_range=False),
)

ema_lowbit_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint4", narrow_range=False),
    act_observer=partial(
        ExponentialMovingAverageObserver, dtype="qint4", narrow_range=False
    ),
    weight_fake_quant=partial(FakeQuantize, dtype="qint4", narrow_range=False),
    act_fake_quant=partial(FakeQuantize, dtype="qint4", narrow_range=False),
)

calibration_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8", narrow_range=True),
    act_observer=partial(HistogramObserver, dtype="qint8", narrow_range=False),
    weight_fake_quant=None,
    act_fake_quant=None,
)
