# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from copy import copy, deepcopy
from typing import Callable, Dict, Tuple

from .. import module as Float
from ..module import Module
from ..module import qat as QAT
from ..module import quantized as Quantized
from ..module.qat import QATModule
from ..module.quantized import QuantizedModule
from .fake_quant import TQT
from .qconfig import QConfig, ema_fakequant_qconfig


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


def _get_convert_dict() -> Tuple[
    Dict[Module, QATModule], Dict[QATModule, QuantizedModule]
]:
    quantable_module_names = _get_quantable_module_names()

    quantable_modules = [getattr(Float, key) for key in quantable_module_names]
    qat_modules = [getattr(QAT, key) for key in quantable_module_names]
    quantized_modules = [getattr(Quantized, key) for key in quantable_module_names]

    float2qat_dict = dict(zip(quantable_modules, qat_modules))
    qat2quantized_dict = dict(zip(qat_modules, quantized_modules))
    return float2qat_dict, qat2quantized_dict


_float2qat_dict, _qat2quantized_dict = _get_convert_dict()


def quantize(module: Module, inplace: bool = True, mapping: dict = None):
    r"""
    Recursively convert :class:`~.QATModule` to :class:`~.QuantizedModule`
    through :meth:`~.Module.apply`.

    :param module: root module to do convert recursively.
    :param inplace: whether to convert submodules in-place.
    :param mapping: a dict indicating how to convert custom modules from QATModule to
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
        if isinstance(parent, Float.Sequential):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            parent[int(key.split(".")[-1])] = new_mod
        else:
            setattr(parent, key.split(".")[-1], new_mod)

    return module


def quantize_qat(
    module: Module,
    inplace: bool = True,
    qconfig: QConfig = ema_fakequant_qconfig,
    mapping: dict = None,
):
    r"""
    Recursively convert float :class:`~.Module` to :class:`~.QATModule`
    through :meth:`~.Module.apply` and set qconfig relatively.

    :param module: root module to do convert recursively.
    :param inplace: whether to convert submodules in-place.
    :param qconfig: an instance of :class:`~.QConfig` to be set as submodules' qconfig.
        default is ``ema_fakequant_qconfig``.
    :param mapping: a dict indicating how to convert custom modules from Module to QATModule.
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
        if isinstance(parent, Float.Sequential):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            parent[int(key.split(".")[-1])] = new_mod
        else:
            setattr(parent, key.split(".")[-1], new_mod)

    propagate_qconfig(module, qconfig)
    return module


def _propagate(module: Module, func_str: str, *args, **kargs):
    def fn(mod: Module):
        if isinstance(mod, QATModule):
            getattr(mod, func_str)(*args, **kargs)

    module.apply(fn)


def propagate_qconfig(module: QATModule, qconfig: QConfig):
    r"""
    Recursively set ``module``'s qconfig through :meth:`~.Module.apply`.

    :param module: root module to traverse recursively.
    :param qconfig: a instance of :class:`~.QConfig` to be set as submodules' qconfig.
    """
    _propagate(module, "set_qconfig", qconfig)


def disable_fake_quant(module: Module):
    r"""
    Recursively disable ``module`` fake quantization in QATModule through :meth:`~.Module.apply`

    :param module: root module to do disable fake quantization recursively.
    """

    _propagate(module, "set_fake_quant", False)


def disable_observer(module: Module):
    r"""
    Recursively disable ``module`` observer in QATModule through :meth:`~.Module.apply`

    :param module: root module to do disable observer recursively.
    """

    _propagate(module, "set_observer", False)


def enable_fake_quant(module: Module):
    r"""
    Recursively enable ``module`` fake quantization in QATModule through :meth:`~.Module.apply`

    :param module: root module to do enable fake quantization recursively.
    """

    _propagate(module, "set_fake_quant", True)


def enable_observer(module: Module):
    r"""
    Recursively enable ``module`` observer in QATModule through :meth:`~.Module.apply`

    :param module: root module to do enable observer recursively.
    """

    _propagate(module, "set_observer", True)
