# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from copy import deepcopy
from typing import Dict, Tuple

from .. import module as Float
from ..module import Module
from ..module import qat as QAT
from ..module import quantized as Quantized
from ..module.qat import QATModule
from ..module.quantized import QuantizedModule
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


def quantize(module: Module, inplace=True):
    r"""
    Recursively convert :class:`~.QATModule` to :class:`~.QuantizedModule`
    through :meth:`~.Module.apply`.

    :param module: root module to do convert recursively.
    :param inplace: whether to convert submodules in-place.
    """

    if not inplace:
        module = deepcopy(module)

    qat_modules = tuple(_qat2quantized_dict.keys())

    def is_qat(mod: Module):
        return isinstance(mod, qat_modules)

    # no need to pass prefix and get pure key of parent Module.
    for key, submodule, parent in module._flatten(
        with_key=True, with_parent=True, predicate=is_qat
    ):
        new_mod = _qat2quantized_dict[type(submodule)].from_qat_module(submodule)
        if isinstance(parent, Float.Sequential):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            parent[int(key.split(".")[-1])] = new_mod
        else:
            setattr(parent, key.split(".")[-1], new_mod)

    return module


def quantize_qat(
    module: Module, inplace=True, qconfig: QConfig = ema_fakequant_qconfig
):
    r"""
    Recursively convert float :class:`~.Module` to :class:`~.QATModule`
    through :meth:`~.Module.apply` and set qconfig relatively.

    :param module: root module to do convert recursively.
    :param inplace: whether to convert submodules in-place.
    :param qconfig: an instance of :class:`~.QConfig` to be set as submodules' qconfig.
        default is ``ema_fakequant_qconfig``.
    """

    if not inplace:
        module = deepcopy(module)

    quantable_modules = tuple(_float2qat_dict.keys())

    def is_quantable(mod: Module):
        return isinstance(mod, quantable_modules)

    # no need to pass prefix and get pure key of parent Module.
    for key, submodule, parent in module._flatten(
        with_key=True, with_parent=True, predicate=is_quantable
    ):
        new_mod = _float2qat_dict[type(submodule)].from_float_module(submodule)
        if isinstance(parent, Float.Sequential):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            parent[int(key.split(".")[-1])] = new_mod
        else:
            setattr(parent, key.split(".")[-1], new_mod)

    propagate_qconfig(module, qconfig)
    return module


def propagate_qconfig(module: QATModule, qconfig: QConfig):
    r"""
    Recursively set ``module``'s qconfig through :meth:`~.Module.apply`.

    :param module: root module to traverse recursively.
    :param qconfig: a instance of :class:`~.QConfig` to be set as submodules' qconfig.
    """

    def fn(mod: Module):
        if isinstance(mod, QATModule):
            mod.set_qconfig(qconfig)

    module.apply(fn)


def disable_fake_quant(module: Module):
    r"""
    Recursively disable `module` fake quantization in QATModule through :meth:`~.Module.apply`

    :param module: root module to do disable fake quantization recursively.
    """

    def fn(mod):
        if isinstance(mod, QATModule):
            mod.act_fake_quant.disable()
            mod.weight_fake_quant.disable()

    module.apply(fn)


def disable_observer(module: Module):
    r"""
    Recursively disable `module` observer in QATModule through :meth:`~.Module.apply`

    :param module: root module to do disable observer recursively.
    """

    def fn(mod):
        if isinstance(mod, QATModule):
            mod.act_observer.disable()
            mod.weight_observer.disable()

    module.apply(fn)


def enable_fake_quant(module: Module):
    r"""
    Recursively enable `module` fake quantization in QATModule through :meth:`~.Module.apply`

    :param module: root module to do enable fake quantization recursively.
    """

    def fn(mod):
        if isinstance(mod, QATModule):
            mod.act_fake_quant.enable()
            mod.weight_fake_quant.enable()

    module.apply(fn)


def enable_observer(module: Module):
    r"""
    Recursively enable `module` observer in QATModule through :meth:`~.Module.apply`

    :param module: root module to do enable observer recursively.
    """

    def fn(mod):
        if isinstance(mod, QATModule):
            mod.act_observer.enable()
            mod.weight_observer.enable()

    module.apply(fn)
