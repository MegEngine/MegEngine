from copy import copy, deepcopy
from functools import partial
from typing import Callable

import numpy as np

from .. import module as Float
from ..functional import concat, norm
from ..logger import get_logger
from ..module import Module
from ..module import qat as QAT
from ..module import quantized as Quantized
from ..module.qat import QATModule
from ..module.quantized import QuantizedModule
from ..tensor import Tensor
from ..utils.module_utils import set_expand_structure
from .qconfig import QConfig, ema_fakequant_qconfig

logger = get_logger(__name__)


def _get_quantable_module_names():
    def is_quantable(key: str):
        value = getattr(Quantized, key)
        return (
            isinstance(value, type)
            and issubclass(value, QuantizedModule)
            and value != QuantizedModule
        )

    # source should have all quantable modules' names
    quantable_module_names = [key for key in dir(Quantized) if is_quantable(key)]
    return quantable_module_names


def _get_convert_dict():
    quantable_module_names = _get_quantable_module_names()

    quantable_modules = [getattr(Float, key) for key in quantable_module_names]
    qat_modules = [getattr(QAT, key) for key in quantable_module_names]
    quantized_modules = [getattr(Quantized, key) for key in quantable_module_names]

    float2qat_dict = dict(zip(quantable_modules, qat_modules))
    qat2quantized_dict = dict(zip(qat_modules, quantized_modules))
    return float2qat_dict, qat2quantized_dict


_float2qat_dict, _qat2quantized_dict = _get_convert_dict()
qat_modules = tuple(_qat2quantized_dict.keys())


def quantize(module: Module, inplace: bool = True, mapping: dict = None):
    r"""Recursively convert :class:`~.QATModule` to :class:`~.QuantizedModule`
    through :meth:`~.Module.apply`.

    Args:
        module: root module to do convert recursively.
        inplace: whether to convert submodules in-place.
        mapping: a dict indicating how to convert custom modules from QATModule to
            QuantizedModule. Will be combined with internal default convert mapping dict.
    """

    if not inplace:
        module = deepcopy(module)

    convert_dict = copy(_qat2quantized_dict)
    if mapping is not None:
        convert_dict.update(mapping)
    qat_modules = tuple(convert_dict.keys())

    def is_qat(mod: Module):
        return isinstance(mod, qat_modules)

    # must use list to avoid replacement influencing successor modules
    for key, submodule, parent in list(
        module._flatten(with_key=True, with_parent=True, predicate=is_qat)
    ):
        new_mod = convert_dict[type(submodule)].from_qat_module(submodule)
        set_expand_structure(module, key, new_mod)

    return module


def quantize_qat(
    module: Module,
    inplace: bool = True,
    qconfig: QConfig = ema_fakequant_qconfig,
    mapping: dict = None,
):
    r"""Recursively convert float :class:`~.Module` to :class:`~.QATModule`
    through :meth:`~.Module.apply` and set qconfig relatively.

    Args:
        module: root module to do convert recursively.
        inplace: whether to convert submodules in-place.
        qconfig: an instance of :class:`~.QConfig` to be set as submodules' qconfig.
            default is ``ema_fakequant_qconfig``.
        mapping: a dict indicating how to convert custom modules from Module to QATModule.
            Will be combined with internal default convert mapping dict.
    """

    if not inplace:
        module = deepcopy(module)

    convert_dict = copy(_float2qat_dict)
    if mapping is not None:
        convert_dict.update(mapping)
    quantable_modules = tuple(convert_dict.keys())

    def is_quantable(mod: Module):
        return isinstance(mod, quantable_modules)

    # must use list to avoid replacement influencing successor modules
    for key, submodule, parent in list(
        module._flatten(with_key=True, with_parent=True, predicate=is_quantable)
    ):
        # only convert top quantable module.
        if is_quantable(parent) or submodule.quantize_disabled:
            continue

        new_mod = convert_dict[type(submodule)].from_float_module(submodule)
        set_expand_structure(module, key, new_mod)

    propagate_qconfig(module, qconfig)
    return module


def reset_qconfig(module: Module, qconfig: QConfig, inplace: bool = True):
    r"""Reset :class:`~._FakeQuantize` and :class:`~.Observer` according to ``qconfig``

    Args:
        module: root module to reset recursively.
        qconfig: an instance of :class:`~.QConfig` to be set as submodules' qconfig.
        inplace: whether to reset submodules in-place.
    """

    if not inplace:
        module = deepcopy(module)

    def safe_call(func, qparams):
        inst = func() if func is not None else None
        if inst is not None and getattr(inst, "set_qparams", None) is not None:
            inst.set_qparams(qparams)
        return inst

    def is_qat(mod: Module):
        return isinstance(mod, QATModule)

    for m in list(module._flatten(predicate=is_qat)):
        if m.with_weight:
            weight_params = m.get_weight_qparams()
            m.weight_observer = safe_call(qconfig.weight_observer, weight_params)
            m.weight_fake_quant = safe_call(qconfig.weight_fake_quant, weight_params)
        if m.with_act:
            act_params = m.get_activation_qparams()
            m.act_observer = safe_call(qconfig.act_observer, act_params)
            m.act_fake_quant = safe_call(qconfig.act_fake_quant, act_params)

    return module


def _propagate(module: Module, func_str: str, *args, **kargs):
    def fn(mod: Module):
        if isinstance(mod, QATModule):
            getattr(mod, func_str)(*args, **kargs)

    module.apply(fn)


def propagate_qconfig(module: QATModule, qconfig: QConfig):
    r"""Recursively set ``module``'s qconfig through :meth:`~.Module.apply`.

    Args:
        module: root module to traverse recursively.
        qconfig: a instance of :class:`~.QConfig` to be set as submodules' qconfig.
    """
    _propagate(module, "set_qconfig", qconfig)


def hook_qat_module(module: Module, func: Callable):
    r"""Add hooks for all :class:`~.QATModule` submodule"""

    def is_qat(mod: Module):
        return isinstance(mod, QATModule)

    hooks = []
    for submodule in list(module._flatten(predicate=is_qat)):
        hooks.append(submodule.register_forward_hook(func))

    return hooks


def apply_easy_quant(
    module: Module, data: Tensor, start: float = 0.8, stop: float = 1.2, num: int = 40
):
    r"""Implementation of ``EasyQuant``: https://arxiv.org/pdf/2006.16669.
    Search for optimal scales.

    Args:
        module: root module.
        data: input tensor used to search optimal scale.
        start: lower bound of the search interval.
        stop: upper bound of the search interval.
        num: number of samples to search.
        module: Module: 
    """

    batch_size = data.shape[0]

    def get_cosine(x, y):
        ndim = len(x.shape)
        axis = tuple(range(1, ndim))
        up = (x * y).sum(axis=axis)
        down = norm(x, axis=axis) * norm(y, axis=axis)
        sim = up / down
        return sim.mean(axis=0)

    def search(mod, inputs, outputs, where):

        mod._forward_hooks.clear()

        normal_in = [_[:batch_size] for _ in inputs]
        fakequant_in = [_[batch_size:] for _ in inputs]

        disable_fake_quant(mod)
        normal_out = mod(*normal_in)
        enable_fake_quant(mod)

        ob = getattr(mod, where)
        if ob is None:
            return

        orig_scale = ob.orig_scale
        cosine = optimal = 0
        for scale in np.linspace(start * orig_scale, stop * orig_scale, num):
            ob.scale = scale
            fakequant_out = mod(*fakequant_in)
            dis = get_cosine(normal_out, fakequant_out)
            if dis > cosine:
                cosine = dis
                optimal = scale
        if optimal == 0:
            logger.warning("EasyQuant finds no better scale")
        else:
            ob.scale = optimal

        fakequant_out = outputs[batch_size:]
        return concat([normal_out, fakequant_out])

    data = concat([data, data])

    hook_qat_module(module, partial(search, where="weight_observer"))
    module(data)

    hook_qat_module(module, partial(search, where="act_observer"))
    module(data)

    return module


def disable_fake_quant(module: Module):
    r"""Recursively disable ``module`` fake quantization in QATModule through :meth:`~.Module.apply`

    Args:
        module: root module to do disable fake quantization recursively.
    """

    _propagate(module, "set_fake_quant", False)


def disable_observer(module: Module):
    r"""Recursively disable ``module`` observer in QATModule through :meth:`~.Module.apply`

    Args:
        module: root module to do disable observer recursively.
    """

    _propagate(module, "set_observer", False)


def enable_fake_quant(module: Module):
    r"""Recursively enable ``module`` fake quantization in QATModule through :meth:`~.Module.apply`

    Args:
        module: root module to do enable fake quantization recursively.
    """

    _propagate(module, "set_fake_quant", True)


def enable_observer(module: Module):
    r"""Recursively enable ``module`` observer in QATModule through :meth:`~.Module.apply`

    Args:
        module: root module to do enable observer recursively.
    """

    _propagate(module, "set_observer", True)
