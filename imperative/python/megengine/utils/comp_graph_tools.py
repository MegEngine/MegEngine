# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np

from ..core import _imperative_rt
from ..core._imperative_rt import GraphProfiler
from ..core._imperative_rt import OperatorNode as _OpNode
from ..core._imperative_rt import VarNode as _VarNode
from ..core.tensor import megbrain_graph as G
from ..core.tensor.megbrain_graph import set_priority_to_id
from ..tensor import Tensor

__all__ = [
    "get_dep_vars",
    "get_owner_opr_inputs",
    "get_owner_opr_type",
    "get_opr_type",
    "graph_traversal",
    "get_oprs_seq",
    "replace_vars",
    "replace_oprs",
    "set_priority_to_id",
    "GraphInference",
]


def get_dep_vars(
    var: Union[_VarNode, List[_VarNode]], var_type: Union[str, List[str]] = None
) -> List[_VarNode]:
    """
    Returns :class:`.tensor.core.megbrain_graph.VarNode` of type ``var_type`` that input ``var``
    depands on. If ``var_type`` is None, returns all types.
    """
    outputs = []
    memo = set()

    if isinstance(var, _VarNode):
        var = [var]

    if isinstance(var_type, str):
        var_type = [var_type]

    q = list(var)
    while q:
        v = q.pop(0)
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


def get_owner_opr_inputs(var: _VarNode) -> List[_VarNode]:
    """
    Gets the inputs of owner opr of a variable.
    """
    return var.owner.inputs


def get_owner_opr_type(var: _VarNode) -> str:
    """
    Gets the type of owner opr of a variable.

    """
    return var.owner.type


def get_opr_type(opr: _OpNode) -> str:
    """
    Gets the type of an opr.
    """
    assert isinstance(opr, _OpNode)
    return opr.type


def graph_traversal(outputs: _VarNode):
    """
    Helper function to traverse the computing graph and return enough useful information.

    :param outputs: model outputs.
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

    queue = list(set(map(lambda x: x.owner, outputs)))
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


def get_oprs_seq(
    outputs: List[_VarNode], prune_reshape=False, prune_immtensor=True
) -> List[_OpNode]:
    """
    Gets oprs in some topological order for a dumped model.

    :param outputs: model outputs.
    :param prune_reshape: whether to prune the useless operators used by Reshape opr during inference.
    :param prune_immtensor: whether to prune the ImmutableTensor opr.
    :return: opr list with some correct execution order.
    """

    def topological_sort(map_oprs, opr2receivers, indegree2opr, opr2indegree):
        # generate an execution order with topological sort algorithm
        oprs_seq = []
        nr_remain = len(map_oprs)
        while indegree2opr[0]:
            opr_id = indegree2opr[0].pop()
            opr = map_oprs[opr_id]
            nr_remain -= 1
            if opr.type != "ImmutableTensor" or not prune_immtensor:
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
        def iterative_pruning(cur_opr, post_opr, marked_opr_ids, visited):
            useless = True
            for oup in cur_opr.outputs:
                if "workspace" not in oup.name:
                    var_idx = post_opr.inputs.index(oup)
                    var2oprs[oup.id].remove((post_opr.id, var_idx))
                    useless = useless and (len(var2oprs[oup.id]) == 0)

            if useless:
                marked_opr_ids.append(cur_opr.id)

                for opr in set([var.owner for var in cur_opr.inputs]):
                    if (opr.id, cur_opr.id) not in visited:
                        visited.add((opr.id, cur_opr.id))
                        iterative_pruning(opr, cur_opr, marked_opr_ids, visited)

        reshape_vars = get_dep_vars(outputs, "Reshape")
        reshape_oprs = [var.owner for var in reshape_vars]

        marked_opr_ids = []
        visited = set()
        for reshape_opr in reshape_oprs:
            iterative_pruning(
                reshape_opr.inputs[1].owner, reshape_opr, marked_opr_ids, visited
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


def replace_vars(
    dst: List[_VarNode], varmap: Dict[_VarNode, _VarNode]
) -> List[_VarNode]:
    """
    Replaces vars in the graph.

    :param dst: target vars representing the graph.
    :param varmap: the map that specifies how to replace the vars.

    :return: new vars that correspond to ``dst`` with all the dependencies
        replaced.
    """
    dst_vec = []
    repl_src_vec = []
    repl_dst_vec = []
    for i in dst:
        assert isinstance(i, _VarNode)
        dst_vec.append(i)

    for i, j in getattr(varmap, "items", lambda: varmap)():
        assert isinstance(i, _VarNode)
        assert isinstance(j, _VarNode)
        repl_src_vec.append(i)
        repl_dst_vec.append(j)

    return _imperative_rt.graph._replace_vars(repl_src_vec, repl_dst_vec, dst_vec)


def replace_oprs(dst: List[_VarNode], oprmap: Dict[_OpNode, _OpNode]) -> List[_VarNode]:
    """
    Replaces operators in the graph.

    :param dst: target vars representing the graph.
    :param oprmap: the map that specifies how to replace the operators.

    :return: new vars that correspond to ``dst`` with all the dependencies
        replaced.
    """
    dst_vec = []
    repl_src_vec = []
    repl_dst_vec = []
    for i in dst:
        assert isinstance(i, _VarNode)
        dst_vec.append(i)

    for i, j in getattr(oprmap, "items", lambda: oprmap)():
        assert isinstance(i, _OpNode)
        assert isinstance(j, _OpNode)
        repl_src_vec.append(i)
        repl_dst_vec.append(j)

    return _imperative_rt.graph._replace_oprs(repl_src_vec, repl_dst_vec, dst_vec)


def find_vars_by_name(dst: List[_VarNode], names: List[str]) -> List[_VarNode]:
    """
    Gets VarNode list by names in the graph.

    :param dst: target vars representing the graph.
    :param names: name list for target VarNode.

    :return: results found by names.
    """
    output_names = names.copy()
    all_vars = get_dep_vars(dst) + dst
    # use dict to keep outputs order the same as names.
    output_dict = {}
    for i in all_vars:
        if i.name in output_names:
            output_dict[i.name] = i
            output_names.remove(i.name)
    assert len(output_names) == 0, "Can not find varnode {} in this model".format(
        output_names
    )
    return [output_dict[i] for i in names]


def convert_inputs(
    dst: List[_VarNode], inputs: List[_VarNode] = None
) -> Tuple[List[_VarNode], Dict[str, _VarNode]]:
    """
    Replaces ``Host2DeviceCopy`` with :class:`~.InputNode` in the graph
    to :meth:`~.InputNode.set_value` and run.

    :param dst: target vars representing the graph.
    :param inputs: indicates which inputs to be replaced. All
        inputs(``Host2DeiceCopy``) will be replaced if not specified.

    :return: new vars that correspond to ``dst`` with all inputs
        replaced, and new inputs dict.
    """
    if inputs is None:
        inputs = get_dep_vars(dst, "Host2DeviceCopy")
    input_dict = OrderedDict()
    replace_dict = {}
    for inp in inputs:
        inp_node = G.InputNode(
            device=inp.comp_node, dtype=inp.dtype, shape=inp.shape, graph=inp.graph,
        )
        inp_node.name = inp.name
        input_dict[inp.name] = inp_node
        replace_dict[inp] = inp_node.outputs[0]
    new_output_nodes = replace_vars(dst, replace_dict)
    for old, new in zip(dst, new_output_nodes):
        new.name = old.name
    return new_output_nodes, input_dict


def convert_outputs(dst: List[_VarNode]) -> Tuple[List[_VarNode], Dict[str, _VarNode]]:
    """
    Wraps ``dst`` with :class:`~.OutputNode` in the graph to get outputs
    with :meth:`~.OutputNode.get_value`.

    :param dst: target vars representing the graph.

    :return: new vars that correspond to ``dst`` with all inputs
        replaced, and outputs dict.
    """
    output_dict = OrderedDict([(i.name, G.OutputNode(i)) for i in dst])
    new_output_nodes = [i.outputs[0] for i in output_dict.values()]
    return new_output_nodes, output_dict


def embed_inputs(
    dst: List[_VarNode], data: List[np.ndarray], inputs: List[_VarNode] = None
) -> Tuple[List[_VarNode], Dict[str, _VarNode]]:
    """
    Embeds ``data`` to the graph's inputs of ``dst``.

    :param dst: target vars representing the graph.
    :param data: data to be embeded.
    :param inputs: indicates which inputs to be replaced. All
        inputs(``Host2DeiceCopy``) will be replaced if not specified.
    :return: new vars that correspond to ``dst`` with all inputs
        replaced, and new inputs dict.
    """
    if inputs is None:
        inputs = get_dep_vars(dst, "Host2DeviceCopy")
    assert len(data) == len(inputs)
    input_dict = OrderedDict()
    replace_dict = {}
    for inp, d in zip(inputs, data):
        new_inp = _imperative_rt.make_shared(inp.graph, Tensor(d)._dev_tensor())
        new_inp.name = inp.name
        input_dict[inp.name] = new_inp
        replace_dict[inp] = new_inp
    new_output_nodes = replace_vars(dst, replace_dict)
    for old, new in zip(dst, new_output_nodes):
        new.name = old.name
    return new_output_nodes, input_dict


class GraphInference:
    """
    Loads a serialized computing graph as a GraphInference object which can be used
    to execute the computing graph.

    :param file: could be file object or filename.
    :param outputs: only compile the subgraph with outputs as its endpoints.
    """

    def __init__(
        self,
        file,
        outputs: List[str] = None,
        profiling: bool = False,
        optimize_for_inference: bool = False,
        **kwargs
    ):
        self._graph, _, output_nodes = G.load_graph(file)
        if outputs is not None:
            output_nodes = find_vars_by_name(output_nodes, outputs)
        self._origin_outputs = output_nodes

        # replace inputs with `InputNode`
        output_nodes, self._inp_dict = convert_inputs(output_nodes)

        # replace outputs with `OutputNode`
        output_nodes, self._oup_dict = convert_outputs(output_nodes)

        self._func = self._graph.compile(output_nodes)

    def run(
        self, *inp_args: np.ndarray, inp_dict: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        :param inp_args: list of input datas.
        :param inp_dict: dict of named input datas.
        :return: a dict {output_name: output_value}.
        """
        assert len(inp_args) <= len(
            self._inp_dict
        ), "This model expects {} inputs".format(len(self._inp_dict))
        inputs = {}
        inp_keys = list(self._inp_dict.keys())
        for ind, data in enumerate(inp_args):
            inputs[inp_keys[ind]] = data
        if inp_dict is not None:
            inputs.update(inp_dict)
        assert (
            inputs.keys() == self._inp_dict.keys()
        ), "This model expects inputs {}, but gets inputs {}".format(
            list(self._inp_dict.keys()), list(inputs.keys())
        )
        for key in self._inp_dict:
            self._inp_dict[key].set_value(
                Tensor(inputs[key], device=self._inp_dict[key].device)._dev_tensor()
            )
        self._func.execute()
        self._func.wait()

        result = OrderedDict()
        for key in self._oup_dict:
            result[key] = self._oup_dict[key].get_value().numpy()
        return result
