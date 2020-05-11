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

    if isinstance(var, _mgb.SymbolVar):
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


def get_opr_type(opr):
    """get the type of a opr

    :type var: :class:`.Operator`
    :rtype: ``str``
    """
    assert isinstance(opr, _mgb.Operator)
    return _mgb._get_opr_type(opr)


def graph_traversal(outputs):
    """helper function to traverse the computing graph and reeturn enough useful information

    :param outputs: model outputs
    :type outputs: :class:`.Symbolvar`
    :return:  tuple (map_oprs, map_vars, var2oprs, opr2receivers, indegree2opr, opr2indegree)
        WHERE
        map_oprs is dict from opr_id to actual opr
        map_vars is dict from var_id to actual var
        var2oprs is dict from var to dest oprs along with index
        opr2receivers is dict from current opr to next opr
        indegree2opr is dict from in_degree to opr in computing graph
        opr2indegree is dict from opr in computing graph to in_degree

        (indegree2opr, opr2indegree) are only used in topological sort in get_oprs_seq function
    """
    # meta information for comp graph
    map_oprs = collections.defaultdict(set)
    map_vars = collections.defaultdict(set)

    var2oprs = collections.defaultdict(list)
    opr2receivers = collections.defaultdict(list)

    queue = list(map(lambda x: x.owner_opr, outputs))
    visited = set(map(lambda x: x.id, queue))

    # iterate through whole comp_graph, fill in meta information
    indegree2opr = collections.defaultdict(set)
    opr2indegree = {}

    idx = 0
    while idx < len(queue):
        cur_opr = queue[idx]
        map_oprs[cur_opr.id] = cur_opr

        idx += 1

        indegree = 0
        for var_idx, var in enumerate(cur_opr.inputs):
            map_vars[var.id] = var
            var2oprs[var.id].append((cur_opr.id, var_idx))

            pre_opr = var.owner_opr

            if pre_opr.id not in visited:
                visited.add(pre_opr.id)
                queue.append(pre_opr)

            indegree += 1
            opr2receivers[pre_opr.id].append(cur_opr.id)

        indegree2opr[indegree].add(cur_opr.id)
        opr2indegree[cur_opr.id] = indegree

    return map_oprs, map_vars, var2oprs, opr2receivers, indegree2opr, opr2indegree


def get_oprs_seq(outputs, prune_reshape=False):
    """get oprs in some topological order for a dumped model

    :param outputs: model outputs
    :param prune_reshape: whether to prune the operators useless during inference
    :return: opr list with some correct execution order
    """

    def topological_sort(map_oprs, opr2receivers, indegree2opr, opr2indegree):
        # generate an execution order with topological sort algorithm
        oprs_seq = []
        nr_remain = len(map_oprs)
        while indegree2opr[0]:
            opr_id = indegree2opr[0].pop()
            opr = map_oprs[opr_id]
            nr_remain -= 1

            # skip const value generation operator
            if get_opr_type(opr) != "ImmutableTensor":
                oprs_seq.append(opr)

            for post_id in opr2receivers[opr_id]:
                indegree = opr2indegree[post_id]
                indegree2opr[indegree].remove(post_id)

                indegree -= 1
                indegree2opr[indegree].add(post_id)
                opr2indegree[post_id] = indegree

        assert nr_remain == 0, "there are {} remaining nodes; cyclic graph?".format(
            nr_remain
        )
        return oprs_seq

    # reshape op definition: reshape(input_tensor, dest_shape) -> output_tensor
    # when inferencing, shape of output_tensor is already known, so one can prune some operators related to dest_shape in the loaded graph
    def prune_reshape_oprs(outputs, oprs_seq, var2oprs):
        def iterative_pruning(cur_opr, post_opr, marked_opr_ids):
            useless = True
            for oup in cur_opr.outputs:
                if "workspace" not in oup.name:
                    var_idx = post_opr.inputs.index(oup)
                    var2oprs[oup.id].remove((post_opr.id, var_idx))
                    useless = useless and (len(var2oprs[oup.id]) == 0)

            if useless:
                marked_opr_ids.append(cur_opr.id)

                for inp in cur_opr.inputs:
                    iterative_pruning(inp.owner_opr, cur_opr, marked_opr_ids)

        reshape_vars = get_dep_vars(outputs, "Reshape")
        reshape_oprs = [var.owner_opr for var in reshape_vars]

        marked_opr_ids = []
        for reshape_opr in reshape_oprs:
            iterative_pruning(
                reshape_opr.inputs[1].owner_opr, reshape_opr, marked_opr_ids
            )

        # filter out all marked oprs
        return list(filter(lambda x: x.id not in marked_opr_ids, oprs_seq))

    map_oprs, _, var2oprs, opr2receivers, indegree2opr, opr2indegree = graph_traversal(
        outputs
    )
    oprs_seq = topological_sort(map_oprs, opr2receivers, indegree2opr, opr2indegree)
    if prune_reshape is True:
        oprs_seq = prune_reshape_oprs(outputs, oprs_seq, var2oprs.copy())
    return oprs_seq


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
