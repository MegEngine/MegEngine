# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Dict, List

from .. import _imperative_rt
from .._imperative_rt import OperatorNode, VarNode


def get_dep_vars(var: VarNode, var_type: str = None) -> List[VarNode]:
    """return :class:`.tensor.core.megbrain_graph.VarNode` of type ``var_type`` that input ``var``
    depands on. If ``var_type`` is None, return all types.
    """
    outputs = []
    memo = set()

    if isinstance(var, VarNode):
        var = [var]

    if isinstance(var_type, str):
        var_type = [var_type]

    q = list(var)
    while q:
        v = q.pop()
        if v in memo:
            continue
        memo.add(v)
        q.extend(get_owner_opr_inputs(v))
        if var_type is not None:
            if get_owner_opr_type(v) in var_type:
                outputs.append(v)
        else:
            outputs.append(v)

    return outputs


def get_owner_opr_inputs(var: VarNode) -> List[VarNode]:
    """get the inputs of owner opr of a variable
    """
    assert isinstance(var, VarNode)
    return var.owner.inputs


def get_owner_opr_type(var: VarNode) -> str:
    """get the type of owner opr of a variable

    """
    assert isinstance(var, VarNode)
    return var.owner.type


def get_opr_type(opr: OperatorNode) -> str:
    """get the type of a opr
    """
    assert isinstance(opr, OperatorNode)
    return opr.type


def graph_traversal(outputs: VarNode):
    """helper function to traverse the computing graph and return enough useful information

    :param outputs: model outputs
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

    queue = list(map(lambda x: x.owner, outputs))
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

            pre_opr = var.owner

            if pre_opr.id not in visited:
                visited.add(pre_opr.id)
                queue.append(pre_opr)

            indegree += 1
            opr2receivers[pre_opr.id].append(cur_opr.id)

        indegree2opr[indegree].add(cur_opr.id)
        opr2indegree[cur_opr.id] = indegree

    return map_oprs, map_vars, var2oprs, opr2receivers, indegree2opr, opr2indegree


def get_oprs_seq(outputs: List[VarNode], prune_reshape=False) -> List[OperatorNode]:
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
                    iterative_pruning(inp.owner, cur_opr, marked_opr_ids)

        reshape_vars = get_dep_vars(outputs, "Reshape")
        reshape_oprs = [var.owner for var in reshape_vars]

        marked_opr_ids = []
        for reshape_opr in reshape_oprs:
            iterative_pruning(reshape_opr.inputs[1].owner, reshape_opr, marked_opr_ids)

        # filter out all marked oprs
        return list(filter(lambda x: x.id not in marked_opr_ids, oprs_seq))

    map_oprs, _, var2oprs, opr2receivers, indegree2opr, opr2indegree = graph_traversal(
        outputs
    )
    oprs_seq = topological_sort(map_oprs, opr2receivers, indegree2opr, opr2indegree)
    if prune_reshape is True:
        oprs_seq = prune_reshape_oprs(outputs, oprs_seq, var2oprs.copy())
    return oprs_seq


def replace_vars(dst: VarNode, varmap: Dict[VarNode, VarNode]) -> List[VarNode]:
    """replace vars in the graph

    :param dst: target vars representing the graph
    :param varmap: the map that specifies how to replace the vars

    :return: new vars that correspond to ``dst`` with all the dependencies
        replaced
    """
    dst_vec = []
    repl_src_vec = []
    repl_dst_vec = []
    for i in dst:
        assert isinstance(i, VarNode)
        dst_vec.append(i)

    for i, j in getattr(varmap, "items", lambda: varmap)():
        assert isinstance(i, VarNode)
        assert isinstance(j, VarNode)
        repl_src_vec.append(i)
        repl_dst_vec.append(j)

    return _imperative_rt.graph._replace_vars(repl_src_vec, repl_dst_vec, dst_vec)


def replace_oprs(
    dst: List[VarNode], oprmap: Dict[OperatorNode, OperatorNode]
) -> List[VarNode]:
    """Replace operators in the graph.

    :param dst: target vars representing the graph
    :param oprmap: the map that specifies how to replace the operators

    :return: new vars that correspond to ``dst`` with all the dependencies
        replaced
    """
    dst_vec = []
    repl_src_vec = []
    repl_dst_vec = []
    for i in dst:
        assert isinstance(i, VarNode)
        dst_vec.append(i)

    for i, j in getattr(oprmap, "items", lambda: oprmap)():
        assert isinstance(i, OperatorNode)
        assert isinstance(j, OperatorNode)
        repl_src_vec.append(i)
        repl_dst_vec.append(j)

    return _imperative_rt.graph._replace_oprs(repl_src_vec, repl_dst_vec, dst_vec)


def set_priority_to_id(dest_vars):
    """For all oprs in the subgraph constructed by dest_vars
       set its priority to id if its original priority is zero
    :param dest_vars: target vars representing the graph
    """
    dest_vec = []
    for i in dest_vars:
        assert isinstance(i, VarNode)
        dest_vec.append(i)
    _imperative_rt.graph._set_priority_to_id(dest_vec)
