from functools import partial

from megengine.quantization import QConfig, tqt_qconfig
from megengine.quantization.fake_quant import TQT


def test_equal():
    qconfig = QConfig(
        weight_observer=None,
        act_observer=None,
        weight_fake_quant=partial(TQT, dtype="qint8", narrow_range=True),
        act_fake_quant=partial(TQT, dtype="qint8", narrow_range=False),
    )
    assert qconfig == tqt_qconfig
