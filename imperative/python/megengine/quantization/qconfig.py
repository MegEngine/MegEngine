from collections import namedtuple
from functools import partial

from ..module import Module
from .fake_quant import TQT, FakeQuantize
from .observer import (
    ExponentialMovingAverageObserver,
    HistogramObserver,
    MinMaxObserver,
    PassiveObserver,
    SyncExponentialMovingAverageObserver,
    SyncMinMaxObserver,
)


# use namedtuple to make class immutable, comparable and easy to print
class QConfig(
    namedtuple(
        "QConfig",
        ["weight_observer", "act_observer", "weight_fake_quant", "act_fake_quant"],
    )
):
    r"""A config class indicating how to do quantize toward :class:`~.QATModule` 's
    ``activation`` and ``weight``. See :meth:`~.QATModule.set_qconfig` for detail usage.

    Args:
        weight_observer: interface to instantiate an :class:`~.Observer` indicating
            how to collect scales and zero_point of wegiht.
        act_observer: similar to ``weight_observer`` but toward activation.
        weight_fake_quant: interface to instantiate a :class:`~.quantization.fake_quant.FakeQuantize` indicating
            how to do fake_quant calculation.
        act_observer: similar to ``weight_fake_quant`` but toward activation.
    
    Examples:
    
        .. code-block::
  
           # Default EMA QConfig for QAT.
           ema_fakequant_qconfig = QConfig(
               weight_observer=partial(MinMaxObserver, dtype="qint8_narrow"),
               act_observer=partial(ExponentialMovingAverageObserver, dtype="qint8"),
               weight_fake_quant=partial(FakeQuantize, dtype="qint8_narrow"),
               act_fake_quant=partial(FakeQuantize, dtype="qint8"),
           )
    
    Each parameter is a ``class`` rather than an instance. And we recommand using ``functools.partial``
    to add initialization parameters of the ``class``, so that don't need to provide parameters in
    :meth:`~.QATModule.set_qconfig`.
    
    Usually we choose narrow version dtype (like ``qint8_narrow``) for weight related
    paramters and normal version for activation related ones. For the result of
    multiplication and addition as ``a * b + c * d``, if four variables are all -128 of
    dtype ``qint8``, then the result will be ``2^15`` and cause overflow.
    Weights are commonly calculated in this way, so need to narrow qmin to -127.
    """

    def __new__(cls, weight_observer, act_observer, weight_fake_quant, act_fake_quant):
        if isinstance(act_observer, Module) or isinstance(weight_observer, Module):
            raise ValueError(
                "QConfig must not receive observer instance, please pass observer"
                " class generator using `partial(Observer, ...)` instead. Use"
                " partial(MyObserver, x=1) to override arguments to constructor if needed"
            )
        return super().__new__(
            cls, weight_observer, act_observer, weight_fake_quant, act_fake_quant
        )


min_max_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8_narrow"),
    act_observer=partial(MinMaxObserver, dtype="qint8"),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8_narrow"),
    act_fake_quant=partial(FakeQuantize, dtype="qint8"),
)

ema_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8_narrow"),
    act_observer=partial(ExponentialMovingAverageObserver, dtype="qint8"),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8_narrow"),
    act_fake_quant=partial(FakeQuantize, dtype="qint8"),
)

sync_ema_fakequant_qconfig = QConfig(
    weight_observer=partial(SyncMinMaxObserver, dtype="qint8_narrow"),
    act_observer=partial(SyncExponentialMovingAverageObserver, dtype="qint8"),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8_narrow"),
    act_fake_quant=partial(FakeQuantize, dtype="qint8"),
)

ema_lowbit_fakequant_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint4"),
    act_observer=partial(ExponentialMovingAverageObserver, dtype="qint4"),
    weight_fake_quant=partial(FakeQuantize, dtype="qint4"),
    act_fake_quant=partial(FakeQuantize, dtype="qint4"),
)

calibration_qconfig = QConfig(
    weight_observer=partial(MinMaxObserver, dtype="qint8_narrow"),
    act_observer=partial(HistogramObserver, dtype="qint8"),
    weight_fake_quant=None,
    act_fake_quant=None,
)

tqt_qconfig = QConfig(
    weight_observer=None,
    act_observer=None,
    weight_fake_quant=partial(TQT, dtype="qint8_narrow"),
    act_fake_quant=partial(TQT, dtype="qint8"),
)

passive_qconfig = QConfig(
    weight_observer=partial(PassiveObserver, dtype="qint8_narrow"),
    act_observer=partial(PassiveObserver, dtype="qint8"),
    weight_fake_quant=partial(FakeQuantize, dtype="qint8_narrow"),
    act_fake_quant=partial(FakeQuantize, dtype="qint8"),
)

easyquant_qconfig = passive_qconfig
