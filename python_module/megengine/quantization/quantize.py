# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from copy import deepcopy

from ..module import Module, QATModule, Sequential, quantized
from .qconfig import QConfig, ema_fakequant_qconfig


def quantize(module: Module, inplace=True):
    r"""
    Recursively convert `module` to `quantized` mode through :meth:`~.Module.apply`.

    :param module: root module to do convert recursively.
    """

    if not inplace:
        module = deepcopy(module)

    def is_qat_module(obj):
        return isinstance(obj, QATModule)

    # no need to pass prefix and get pure key of parent Module.
    for key, submodule, parent in module._flatten(
        with_key=True, with_parent=True, predicate=is_qat_module
    ):
        if isinstance(parent, Sequential):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            parent[int(key.split(".")[-1])] = submodule.to_quantized()
        else:
            setattr(parent, key.split(".")[-1], submodule.to_quantized())


def quantize_qat(module: Module, qconfig: QConfig = ema_fakequant_qconfig):
    r"""
    Recursively convert `module` to `qat` mode through :meth:`~.Module.apply`
    and set qconfig relatively.

    :param module: root module to do convert recursively.
    :param qconfig: a instance of :class:`~.QConfig` to be set as submodules' qconfig.
        default is :any:`~.qconfig.ema_fakequant_qconfig`.
    """

    def fn(mod: Module):
        if isinstance(mod, QATModule):
            mod.set_qat_mode(QATModule.QATMode.QAT)
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
