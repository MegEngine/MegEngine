# -*- coding: utf-8 -*-

import os

from .._imperative_rt.ops._custom import (
    _get_custom_op_lib_info,
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
    op_in_this_lib = _install(lib_path, lib_path)
    for op in op_in_this_lib:
        op_maker = _gen_custom_op_maker(op)
        globals()[op] = op_maker
        __all__.append(op)


def unload(lib_path):
    lib_path = os.path.abspath(lib_path)
    op_in_lib = _uninstall(lib_path)
    for op in op_in_lib:
        del globals()[op]
        __all__.remove(op)


def _make_official_custom_op():
    official_opr_list = _get_custom_op_list()
    for op in official_opr_list:
        op_maker = _gen_custom_op_maker(op)
        if op not in globals():
            globals()[op] = op_maker
            __all__.append(op)


_make_official_custom_op()
