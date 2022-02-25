from copy import deepcopy
from typing import Union

from ..core.tensor.dtype import QuantDtypeMeta
from ..quantization.fake_quant import QParamsModuleMixin, _FakeQuantize
from ..quantization.utils import QParams, QuantMode, fake_quant_tensor


class FakeQuantize(_FakeQuantize, QParamsModuleMixin):
    r"""A module to do quant and dequant according to :attr:`~.FakeQuantize.qparams`."""

    def __init__(
        self, dtype: Union[str, QuantDtypeMeta], enable: bool = True, **kwargs
    ):
        super().__init__(dtype, enable, **kwargs)
        self.qparams = None

    def fake_quant_forward(self, inp, qparams: QParams = None):
        if qparams is None:
            qparams = self.get_qparams()
        assert (
            qparams.dtype_meta is self.dtype
        ), "input qparams' dtype is not equal to self.dtype.\nqparams.dtype_meta={}\nself.dtype={}".format(
            qparams.dtype_meta, self.dtype
        )
        return fake_quant_tensor(inp, qparams)

    def get_qparams(self):
        return self.qparams

    def set_qparams(self, qparams: QParams):
        r"""Initialize :attr:`~.FakeQuantize.qparams`.
        
        Args:
            qparams: used to set initial ``scale`` and ``zero_point``.
        """
        if qparams.scale is None:
            raise AssertionError("Can not get an initialized scale")
        scale = qparams.scale
        if qparams.dtype_meta is None:
            qparams.dtype_meta = self.dtype
        else:
            assert (
                qparams.dtype_meta is self.dtype
            ), "input qparams' dtype is not equal to self.dtype.\nqparams.dtype_meta={}\nself.dtype={}".format(
                qparams.dtype_meta, self.dtype
            )
        dtype_meta = qparams.dtype_meta
        zero_point = qparams.zero_point
        mode = qparams.mode

        self.qparams = QParams(mode, dtype_meta, scale, zero_point)
