# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""tools for graph manipulation"""

import collections

from . import mgb as _mgb


def get_dep_vars(var, var_type=None):
    """return :class:`.SymbolVar` of type ``var_type`` that input ``var``
    depands on. If ``var_type`` is None, return all types.

    :type var: an instance or iterable of :class:`.SymbolVar`
    :type var_type: ``str`` or an iterable of ``str``
    "rtype: list of :class:`.SymbolVar`
    """
    outputs = []
    memo = set()

    if not isinstance(var, collections.Iterable):
        var = [var]

    if isinstance(var_type, str):
        var_type = [var_type]

    q = list(var)
    while q:
        v = q.pop()
        if v in memo:
            continue
        memo.add(v)
        q.extend(get_inputs(v))
        if var_type is not None:
            if get_type(v) in var_type:
                outputs.append(v)
        else:
            outputs.append(v)

    return outputs


def get_inputs(var):
    """get the inputs of owner opr of a variable

    :type var: :class:`.SymbolVar`
    :rtype: list of :class:`.SymbolVar`
    """
    assert isinstance(var, _mgb.SymbolVar)
    return _mgb._get_owner_opr_inputs(var)


def get_type(var):
    """get the type of owner opr of a variable

    :type var: :class:`.SymbolVar`
    :rtype: ``str``
    """
    assert isinstance(var, _mgb.SymbolVar)
    return _mgb._get_owner_opr_type(var)


def replace_vars(dst, varmap):
    """replace vars in the graph

    :param dst: target vars representing the graph
    :type dst: list of :class:`.SymbolVar`
    :param varmap: the map that specifies how to replace the vars
    :type varmap: dict that maps from src var to dst var

    :return: new vars that correspond to ``dst`` with all the dependencies
        replaced
    :rtype: list of :class:`.SymbolVar`
    """
    dst_vec = _mgb._VectorSymbolVar()
    repl_src_vec = _mgb._VectorSymbolVar()
    repl_dst_vec = _mgb._VectorSymbolVar()
    for i in dst:
        assert isinstance(i, _mgb.SymbolVar)
        dst_vec.push_back(i)

    for i, j in getattr(varmap, "items", lambda: varmap)():
        assert isinstance(i, _mgb.SymbolVar)
        assert isinstance(j, _mgb.SymbolVar)
        repl_src_vec.push_back(i)
        repl_dst_vec.push_back(j)

    return _mgb._replace_vars(repl_src_vec, repl_dst_vec, dst_vec)


def replace_oprs(dst, oprmap):
    """Replace operators in the graph. Roughly equivalent to

    :param dst: target vars representing the graph
    :type dst: list of :class:`.SymbolVar`
    :param oprmap: the map that specifies how to replace the operators
    :type oprmap: dict that maps from src operator to dst operator

    :return: new vars that correspond to ``dst`` with all the dependencies
        replaced
    :rtype: list of :class:`.SymbolVar`
    """
    dst_vec = _mgb._VectorSymbolVar()
    repl_src_vec = _mgb._VectorOperator()
    repl_dst_vec = _mgb._VectorOperator()
    for i in dst:
        assert isinstance(i, _mgb.SymbolVar)
        dst_vec.push_back(i)

    for i, j in getattr(oprmap, "items", lambda: oprmap)():
        assert isinstance(i, _mgb.Operator)
        assert isinstance(j, _mgb.Operator)
        repl_src_vec.push_back(i)
        repl_dst_vec.push_back(j)

    return _mgb._replace_oprs(repl_src_vec, repl_dst_vec, dst_vec)
