# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import abstractmethod

from ...core import Tensor
from ...quantization import FakeQuantize, Observer, QConfig
from ..module import Module


class QATModule(Module):
    r"""
    Base class of quantized-float related Module, basically for QAT and Calibration.

    Use :meth:`~.QATModule.from_float_module` to generate a instance from float :class:`~.Module`.
    Or use :func:`~.quantize.quantize_qat` to do it recursively and automatically.

    Can also be converted to :class:`~.QuantizedModule` for deployment using
    :func:`~.quantize.quantize` further.
    """

    with_weight = True
    with_act = True

    def __init__(self):
        super().__init__()

        self.weight_observer = None  # type: Observer
        self.act_observer = None  # type: Observer

        self.weight_fake_quant = None  # type: FakeQuantize
        self.act_fake_quant = None  # type: FakeQuantize

    def set_qconfig(self, qconfig: QConfig):
        r"""
        Set quantization related configs with ``qconfig``, including
        observer and fake_quant for weight and activation.
        """

        def safe_call(func):
            return func() if func is not None else None

        if self.with_act:
            self.act_observer = safe_call(qconfig.act_observer)
            self.act_fake_quant = safe_call(qconfig.act_fake_quant)
        if self.with_weight:
            self.weight_observer = safe_call(qconfig.weight_observer)
            self.weight_fake_quant = safe_call(qconfig.weight_fake_quant)

    def _enable_exec(self, with_module, func, enable):
        if not with_module:
            return
        if enable:
            func.enable()
        else:
            func.disable()

    def set_fake_quant(self, enable):
        self._enable_exec(self.with_act, self.act_fake_quant, enable)
        self._enable_exec(self.with_weight, self.weight_fake_quant, enable)

    def set_observer(self, enable):
        self._enable_exec(self.with_act, self.act_observer, enable)
        self._enable_exec(self.with_weight, self.weight_observer, enable)

    def _apply_fakequant_with_observer(
        self, target: Tensor, fake_quant: FakeQuantize, observer: Observer
    ):
        # do observer
        if observer is None:
            q_dict = None
            oup = target
        else:
            q_dict = observer.get_qparams()
            oup = observer(target)
        # do fake quant
        if fake_quant is not None:
            oup = fake_quant(oup, q_dict)
            # use qparams of fake_quant if have.
            if hasattr(fake_quant, "get_qparams"):
                q_dict = fake_quant.get_qparams()
        # set to tensor qparams.
        if q_dict is not None:
            oup.q_dict.update(q_dict)
        return oup

    def apply_quant_weight(self, target: Tensor):
        r"""
        Apply weight's observer and fake_quant from ``qconfig`` on ``target``.
        """
        return self._apply_fakequant_with_observer(
            target, self.weight_fake_quant, self.weight_observer
        )

    def apply_quant_activation(self, target: Tensor):
        r"""
        Apply weight's observer and fake_quant from ``qconfig`` on ``target``.
        """
        return self._apply_fakequant_with_observer(
            target, self.act_fake_quant, self.act_observer
        )

    def _get_method_result(
        self, method: str, fake_quant: FakeQuantize, observer: Observer
    ):
        if hasattr(fake_quant, method):
            return getattr(fake_quant, method)()
        elif hasattr(observer, method):
            return getattr(observer, method)()
        return None

    def get_weight_dtype(self):
        r"""
        Get weight's quantization dtype as the method from ``qconfig``.
        """
        return self._get_method_result(
            "get_dtype", self.weight_fake_quant, self.weight_observer
        )

    def get_activation_dtype(self):
        r"""
        Get activation's quantization dtype as the method from ``qconfig``.
        """
        return self._get_method_result(
            "get_dtype", self.act_fake_quant, self.act_observer
        )

    def get_weight_qparams(self):
        r"""
        Get weight's quantization parameters.
        """
        return self._get_method_result(
            "get_qparams", self.weight_fake_quant, self.weight_observer
        )

    def get_activation_qparams(self):
        r"""
        Get activation's quantization parameters.
        """
        return self._get_method_result(
            "get_qparams", self.act_fake_quant, self.act_observer
        )

    @classmethod
    @abstractmethod
    def from_float_module(cls, float_module: Module):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
