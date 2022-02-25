# -*- coding: utf-8 -*-

import os

from .._imperative_rt.ops._custom import (
    _get_custom_op_list,
    _install,
    _make_custom_op,
    _uninstall,
    get_custom_op_abi_tag,
)

__all__ = ["load"]


def _gen_custom_op_maker(custom_op_name):
    def op_maker(**kwargs):
        return _make_custom_op(custom_op_name, kwargs)

    return op_maker


def load(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib_name = os.path.splitext(lib_path)[0]
    op_in_this_lib = _install(lib_name, lib_path)
    for op in op_in_this_lib:
        op_maker = _gen_custom_op_maker(op)
        globals()[op] = op_maker
        __all__.append(op)


def unload(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib_name = os.path.splitext(lib_path)[0]
    _uninstall(lib_name)
