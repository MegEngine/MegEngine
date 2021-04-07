# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import fnmatch
import itertools
import re
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np

from ..core._imperative_rt import ComputingGraph
from ..core._imperative_rt.core2 import SymbolVar
from ..core.tensor import megbrain_graph as G
from ..logger import get_logger
from .comp_graph_tools import get_dep_vars, get_opr_type, get_oprs_seq
from .network_node import (
    Host2DeviceCopy,
    ImmutableTensor,
    NetworkNode,
    OpNode,
    VarNode,
    str_to_mge_class,
)

logger = get_logger(__name__)


class Network:
    def __init__(self):
        self.input_vars = []  # input var of graph
        self._orig_inputs = []
        self.output_vars = []  # output var of graph
        self._orig_outputs = []
        self.all_oprs_map = OrderedDict()
        self.all_vars_map = OrderedDict()
        self.graph = ComputingGraph()

    @classmethod
    def load(cls, model_path: str, outspec: List[str] = None):
        """
        Loads a computing graph as a Network object.
        :param model_path: file path of mge model.
        :param outspec: only load the subgraph with outspec as its endpoints.
        """
        self = cls()
        _, _, outputs = G.load_graph(model_path)
        if outspec is not None:
            output_spec = outspec.copy()
            all_vars = get_dep_vars(outputs) + outputs
            new_outputs = {}
            for i in all_vars:
                if i.name in output_spec:
                    new_outputs[i.name] = i
                    output_spec.remove(i.name)
            assert len(output_spec) == 0, "Can not find {} in this model".format(
                output_spec
            )
            outputs = [new_outputs[i] for i in outspec]
        self._orig_outputs = outputs
        for x in self._orig_outputs:
            self.output_vars.append(self._get_var(x))
        self.add_dep_oprs()
        for x in self._orig_inputs:
            self.input_vars.append(self._get_var(x))

        self.graph = self._orig_outputs[0].graph
        return self

    def _compile(self):
        self.all_oprs_map = {}
        self.all_vars_map = {}
        for opr in self.all_oprs:
            if isinstance(opr, (ImmutableTensor, Host2DeviceCopy)):
                opr.compile(self.graph)
            else:
                opr.compile()
            if opr.name is not None:
                opr._opr.name = opr.name
            self.all_oprs_map[opr._opr.id] = opr
            for o in opr.outputs:
                self.all_vars_map[o.var.id] = o

    def optimize_for_inference(self, dest_vars, **kwargs):
        r"""
        Applies optimize_for_inference pass for operator graph.

            :param dest_vars: list of output vars in the operator graph

            :Keyword Arguments:

                * enable_io16xc32 --
                    whether to use float16 for I/O between oprs and use
                    float32 as internal computation precision. Note the output var would be
                    changed to float16.
                * enable_ioc16 --
                    whether to use float16 for both I/O and computation
                    precision.

                * enable_hwcd4 --
                    whether to use NHWCD4 data layout. This is faster on some
                    OpenCL backend.
                * enable_nchw88 --
                    whether to use NCHW88 data layout, currently
                    used in X86 AVX backend.
                * enable_nchw44 --
                    whether to use NCHW44 data layout, currently
                    used in arm backend.
                * enable_nchw44_dot --
                    whether to use NCHW44_dot data layout, currently
                    used in armv8.2+dotprod backend.
                * enable_nchw4 --
                    whether to use NCHW4 data layout, currently
                    used in nvidia backend(based on cudnn).
                * enable_nchw32 --
                    whether to use NCHW32 data layout, currently
                    used in nvidia backend with tensorcore(based on cudnn).
                * enable_chwn4 --
                    whether to use CHWN4 data layout, currently
                    used in nvidia backend with tensorcore.

                * enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
                    into one opr.
                * enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z
                    input for inference on nvidia backend(this optimization pass will
                    result in mismatch of the precision of output of training and
                    inference)
        """

        if not isinstance(dest_vars, Sequence):
            dest_vars = [dest_vars]
        dest_vars = list(G.VarNode(var.var) for var in dest_vars)
        new_vars = G.optimize_for_inference(dest_vars, **kwargs)
        return list(self._get_var(var) for var in new_vars)

    def dump(
        self,
        file,
        *,
        keep_var_name: int = 1,
        keep_opr_name: bool = False,
        keep_param_name: bool = False,
        keep_opr_priority: bool = False,
        strip_info_file=None,
        append_json=False,
        optimize_for_inference=True,
        append=False,
        **kwargs
    ):
        """
        Serializes graph to file.

        :param file: output file, could be file object or filename.
        :param append: whether output is appended to ``file``.
            Only works when ``file`` is str.
        :param keep_var_name: level for keeping variable names:

            * 0: none of the names are kept
            * 1: (default)keep names of output vars
            * 2: keep names of all (output and internal) vars
        :param keep_opr_name: whether to keep operator names.
        :param keep_param_name: whether to keep param names, so param values can be
            easily manipulated after loading model
        :param keep_opr_priority: whether to keep priority setting for operators
        :param strip_info_file: a string for path or a file handler. if is not None,
            then the dump information for code strip would be written to ``strip_info_file``
        :param append_json: will be check when `strip_info_file` is not None. if set
            true, the information for code strip will be append to strip_info_file.
            if set false, will rewrite strip_info_file
        :param optimize_for_inference: enbale optmizations,
            will skip all optimize options if this is False. Default: True

        :Keyword Arguments:

            See also :py:meth:`optimize_for_inference`.

        """

        self._compile()
        out = [G.VarNode(var.var) for var in self.output_vars]

        if kwargs.pop("arg_names", False):
            logger.warning(
                '"arg_names" is not supported in Network.dump, rename input vars directly'
            )
        if kwargs.pop("output_names", False):
            logger.warning(
                '"output_names" is not supported in Network.dump, rename output vars directly'
            )

        if optimize_for_inference:
            out = G.optimize_for_inference(out, **kwargs)

        dump_content, _ = G.dump_graph(
            out,
            keep_var_name=keep_var_name,
            keep_opr_name=keep_opr_name,
            keep_param_name=keep_param_name,
            keep_opr_priority=keep_opr_priority,
            strip_info_file=strip_info_file,
            append_json=append_json,
        )
        if isinstance(file, str):
            permission = "wb" if append == False else "ab"
            file = open(file, permission)
        file.write(dump_content)

    def make_const(self, data, name=None, device=None):
        """Makes an ImmutableTensor OpNode to provide a parameter for the network.
        """
        node = ImmutableTensor(data, name, device, self.graph)
        node.compile(self.graph)
        return node.outputs[0]

    def make_input_node(self, shape, dtype, name=None, device=None):
        """Makes a Host2DeviceCopy OpNode to provide an input varnode for the network.
        """
        node = Host2DeviceCopy(shape, dtype, name, device)
        node.compile(self.graph)
        return node.outputs[0]

    def add_output(self, *vars: VarNode):
        """Adds vars into the network output node list
        """
        if not all([var.owner for var in vars]):
            self.add_dep_oprs(*vars)
        for var in vars:
            if var not in self.output_vars:
                self.output_vars.append(var)

    def remove_output(self, *vars: VarNode):
        """Removes vars from the network output node list.
        """
        for var in vars:
            if var in self.output_vars:
                self.output_vars.remove(var)

    def add_dep_oprs(self, *vars):
        if len(vars) == 0:
            vars = self.output_vars
        q = list(vars)
        while len(q) > 0:
            cur = q.pop(0)
            if cur.owner is not None:
                continue
            if cur.name is None:
                cur.name = cur.var.name
            self.all_vars_map[cur.var.id] = cur
            mge_opr = cur.var.owner
            if get_opr_type(mge_opr) == "Host2DeviceCopy":
                self._orig_inputs.extend(mge_opr.outputs)
            cur.owner = self._add_opr(mge_opr)
            if cur.owner is None:
                cur.owner = self.all_oprs_map[mge_opr.id]
                continue
            q.extend(cur.owner.inputs)
        return list(vars)

    def modify_opr_names(self, modifier):
        """Modifies names of operators **inplace**; useful for merging loaded
        network into another network

        :param modifier: a string to be prepended to the name, or a function
            that maps from name to name
        :type modifier: str or callable
        """
        if isinstance(modifier, str):
            om = modifier
            modifier = lambda v: "{}.{}".format(om, v)
        assert isinstance(modifier, collections.Callable)
        for i in self.all_oprs:
            v0 = i.name
            v1 = modifier(v0)
            assert isinstance(v1, str)
            i.name = v1

    def reset_batch_size(self, batchsize, *, blacklist=()):
        """Helper for reset batch size; first dimension of all data providers
        not in blacklist are assumed to be the batch size

        :param blacklist: data provider names whose first dimension is not
            batchbatch size
        """
        blacklist = set(blacklist)
        prev_batchsize = None
        for i in self.data_providers_filter:
            if i.name in blacklist:
                blacklist.remove(i.name)
            else:
                shp = list(i.shape)
                if prev_batchsize is None:
                    prev_batchsize = shp[0]
                else:
                    assert prev_batchsize == shp[0], (
                        "batchsize mismatch: batchsize={} "
                        "shape={} dp={}".format(prev_batchsize, shp, i.name)
                    )
                shp[0] = batchsize
                i.shape = tuple(shp)

        assert prev_batchsize is not None, "no data provider found"
        assert not blacklist, "unused items in blacklist: {}".format(blacklist)

    def replace_vars(self, repl_dict: Dict[VarNode, VarNode]):
        """
        Replaces vars in the graph.
        :param repl_dict: the map {old_var: new_var} that specifies how to replace the vars.
        """
        if not all([var.owner for var in repl_dict.values()]):
            print(repl_dict.values())
            self.add_dep_oprs(*list(repl_dict.values()))
        for var in self.all_vars:
            if var in repl_dict:
                repl_var = repl_dict[var]
                owner = repl_var.owner
                idx = owner.outputs.index(repl_var)
                owner.outputs[idx] = var
                var.__dict__.update(repl_var.__dict__)
                var.var = repl_var.var

    def replace_oprs(self, repl_dict: Dict[OpNode, OpNode]):
        """
        Replaces operators in the graph.
        :param oprmap: the map {old_opr: new_opr} that specifies how to replace the operators.
        """
        for opr in self.all_oprs:
            if opr in repl_dict:
                assert len(opr.outputs) == len(
                    repl_dict[opr].outputs
                ), "can not replace {} with {}".format(type(opr), type(repl_dict[opr]))
                repl_dict[opr].outputs = opr.outputs
                for ind, var in enumerate(opr.outputs):
                    var.owner = repl_dict[opr]
                    var.__dict__.update(repl_dict[opr].outputs[ind].__dict__)
                    var.var = repl_dict[opr].outputs[ind].var

    def get_opr_by_type(self, oprcls, unique=True):
        assert issubclass(oprcls, OpNode)
        rst = self.opr_filter.type(oprcls).as_list()
        if unique:
            assert len(rst) == 1, "{} operators of type {} found".format(
                len(rst), oprcls
            )
            (rst,) = rst
        return rst

    def get_opr_by_name(self, name, unique=True):
        rst = self.opr_filter.name(name).as_list()
        if unique:
            assert len(rst) == 1, "{} operators of type {} found".format(len(rst), name)
            (rst,) = rst
        return rst

    def get_var_by_name(self, name, unique=True):
        rst = self.var_filter.name(name).as_list()
        if unique:
            assert len(rst) == 1, "{} operators of type {} found".format(len(rst), name)
            (rst,) = rst
        return rst

    def get_var_receive_oprs(self, var):
        """ Gets all oprs which use var as input
        """
        return self.opr_filter.has_input(var).as_list()

    def get_dep_oprs(self, var):
        """Gets dependent oprs of var
        """
        return get_oprs_seq(var, False, False)

    @property
    def opr_filter(self):
        """Filter on all opnodes of the Network.
        """
        oprs = self.all_oprs
        return NodeFilter(itertools.islice(oprs, len(oprs)))

    @property
    def var_filter(self):
        """Filter on all varnode of the Network.
        """
        vars = self.all_vars
        return NodeFilter(itertools.islice(vars, len(vars)))

    @property
    def params_filter(self):  # all immutable tensor
        """Filter on all parameters (ImmutableTensor Opr) of the Network
        """
        return self.opr_filter.param_provider()

    @property
    def data_providers_filter(self):  # all host2devicecopy
        """Filter on all input nodes (Host2DeviceCopy Opr) of the Network
        """
        return self.opr_filter.data_provider()

    @property
    def dest_vars(self):
        """Output varnodes of the Network.
        """
        return self.output_vars

    @property
    def all_oprs(self):
        return get_oprs_seq(self.output_vars, False, False)

    @property
    def all_vars(self):
        return get_dep_vars(self.output_vars)

    @property
    def all_vars_dict(self):
        return self.var_filter.as_dict()

    @property
    def all_oprs_dict(self):
        return self.opr_filter.as_dict()

    # used for loading and building graph
    def _add_opr(self, opr):
        # TODO: use megbrain C++ RTTI to replace type string
        if opr.id not in self.all_oprs_map:
            opnode = str_to_mge_class(get_opr_type(opr)).load(opr)
            self.all_oprs_map[opr.id] = opnode
            for var in opr.inputs:
                opnode.add_inp_var(self._get_var(var))
            for var in opr.outputs:
                opnode.add_out_var(self._get_var(var))
            return opnode
        else:
            return None

    def _get_opr(self, x):
        if x.id in self.all_oprs_map:
            return self.all_oprs_map[x.id]
        else:
            return None

    def _get_var(self, x):
        # auto convert to VarNode of Network
        if x.id not in self.all_vars_map or self.all_vars_map[x.id].var != x:
            self.all_vars_map[x.id] = VarNode.load(x, self._get_opr(x.owner))
        return self.all_vars_map[x.id]


def as_varnode(obj):
    """convert a :class:`.VarNode` compatible object to :class:`.VarNode`.

    :param obj: it must be one of the following:

        1. a :class:`.VarNode` object
        2. a :class:`.OpNode` object that has unique output
        3. an iterable that produces either type 1 or 2, with length 1

    :rtype: :class:`.VarNode`
    """
    if type(obj) is VarNode:
        return obj

    if isinstance(obj, OpNode):
        assert len(obj.outputs) == 1, (
            "operator {} must have one output to be converted to VarNode; "
            "got {} actually".format(obj, len(obj.outputs))
        )
        ret = obj.outputs[0]
        assert type(ret) is VarNode
        return ret

    assert isinstance(
        obj, collections.Iterable
    ), "{} is not compatible with VarNode".format(obj)

    val = list(obj)
    assert (
        len(val) == 1
    ), "can not convert sequence of length {} to VarNode ({})".format(
        len(val), (lambda s: s if len(s) < 50 else s[:50] + " ...")(str(val))
    )
    return as_varnode(val[0])


def as_oprnode(obj):
    """convert a :class:`.OpNode` compatible object to
    :class:`.OpNode`; it works like :func:`as_varnode`."""
    if type(obj) is VarNode:
        return obj.owner

    if isinstance(obj, OpNode):
        return obj

    assert isinstance(
        obj, collections.Iterable
    ), "{} is not compatible with OpNode".format(obj)

    val = list(obj)
    assert (
        len(val) == 1
    ), "can not convert sequence of length {} to " "OpNode({})".format(len(val), val)
    return as_oprnode(val[0])


class NodeFilter:
    """Filter on node iterator. This class is an iterator of
    :class:`.NetworkNode` objects and multiple filtering conditions and
    mappers can be chained.

    Example::

        # find all :class:`.ImmutableTensor` nodes
        for i in NodeFilter(node_iter).param_provider():
            print(i)

        # find all :class:`.ImmutableTensor` nodes that end with ':W'
        for i in NodeFilter(node_iter).param_provider().name('*:W'):
            print(i)

        # number of inputs
        nr_input = NodeFilter(node_iter).data_provider().as_count()

    """

    _iter = None

    def __init__(self, node_iter):
        """
        :param node_iter: iterator to :class:`.NetworkNode`, or a
            :class:`.VarNode`-compatible object; in the later case, its
            dependent oprs would be used
        """
        if isinstance(node_iter, VarNode):
            oprs = get_oprs_seq(node_iter, False, False)
            node_iter = itertools.islice(oprs, len(oprs) - 1)
        if isinstance(node_iter, OpNode):
            oprs = get_oprs_seq(node_iter.inputs, False, False)
            node_iter = itertools.islice(oprs, len(oprs) - 1)

        assert isinstance(node_iter, collections.Iterable)
        if (not isinstance(node_iter, NodeFilter)) and type(
            self
        ) is not NodeFilterCheckType:
            node_iter = NodeFilterCheckType(node_iter, NetworkNode)
        self._iter = node_iter

    @classmethod
    def make_all_deps(cls, *dest_vars):
        """make a :class:`NodeFilter` that contains all deps of given vars"""
        return cls(list(get_oprs_seq(dest_vars, False, False)))

    def __iter__(self):
        """to be overwritten by subclass to implement filters"""
        return iter(self._iter)

    def type(self, node_type):
        """filter by specific node type

        :param node_type: node type class
        :return: a new :class:`NodeFilter` object
        """
        return NodeFilterType(self, node_type)

    def check_type(self, node_type):
        """assert that all oprs produced by this iterator are instances of
        certain type

        :param node_type: node type class
        :return: a new :class:`NodeFilter` object
        :raises TypeError: if type check failed
        """
        return NodeFilterCheckType(self, node_type)

    def not_type(self, node_type):
        """remove oprs of specific type

        :param node_type: node type class
        :return: a new :class:`NodeFilter` object
        """
        return NodeFilterNotType(self, node_type)

    def param_provider(self):
        """get :class:`.ParamProvider` oprs; shorthand for
        ``.type(ParamProvider)``"""

        return self.type(ImmutableTensor)

    def data_provider(self):
        """get :class:`.DataProvider` oprs; shorthand for
        ``.type(DataProvider)``"""

        return self.type(Host2DeviceCopy)

    def name(self, pattern, ignorecase=True):
        """filter by node name

        :param pattern: a string in glob syntax that can contain ``?`` and
            ``*`` to match a single or arbitrary characters.
        :type pattern: :class:`str`
        :param ignorecase: whether to ignroe case
        :type ignorecase: bool
        :return: a new :class:`NodeFilter` object
        """
        return NodeFilterName(self, pattern, ignorecase)

    def has_input(self, var):
        """an opr is kept if it has given var as one of its inputs

        :param var: var node to checked
        :return: a new :class:`NodeFilter` object
        """
        return NodeFilterHasInput(self, var)

    def as_list(self):
        """consume this iterator and return its content as a list

        :rtype: [:class:`.GraphNodeBase`]
        """
        return list(self)

    def as_unique(self):
        """assert that this iterator yields only one node and return it

        :return: the unique node
        :rtype: :class:`.GraphNodeBase`
        :raises ValueError: if this iterator does not yield a unique node
        """
        (opr,) = self
        return opr

    def as_dict(self):
        """construct an ordered dict to map from node names to objects in
        this iterator

        :rtype: :class:`OrderedDict`
        """
        return collections.OrderedDict((i.name, i) for i in self)

    def as_count(self):
        """consume this iterator and get the number of elements

        :rtype: int
        """
        return sum(1 for _ in self)


class NodeFilterType(NodeFilter):
    """see :meth:`NodeFilter.type`"""

    _node_type = None

    def __init__(self, node_iter, node_type):
        assert issubclass(node_type, NetworkNode), "bad opr type: {}".format(node_type)
        super().__init__(node_iter)
        self._node_type = node_type

    def __iter__(self):
        for i in self._iter:
            if isinstance(i, self._node_type):
                yield i


class NodeFilterNotType(NodeFilterType):
    """see :meth:`NodeFilter.not_type`"""

    def __iter__(self):
        for i in self._iter:
            if not isinstance(i, self._node_type):
                yield i


class NodeFilterCheckType(NodeFilterType):
    """see :meth:`NodeFilter.check_type`"""

    def __iter__(self):
        for i in self._iter:
            if not isinstance(i, self._node_type):
                raise TypeError(
                    "all nodes should be {}; got {!r}".format(self._node_type, i)
                )
            yield i


class NodeFilterHasInput(NodeFilter):
    """see :meth:`NodeFilter.has_input`"""

    _var = None

    def __init__(self, node_iter, var):
        var = as_varnode(var)
        super().__init__(node_iter)
        self.var = var

    def __iter__(self):
        for i in self._iter:
            assert isinstance(
                i, OpNode
            ), "has_input() must be used with OpNode; " "got {!r}".format(i)
            if any(self.var is _ for _ in i.inputs):
                yield i


class NodeFilterName(NodeFilter):
    """see :meth:`NodeFilter.name`"""

    _re = None

    def __init__(self, node_iter, pattern, ignorecase):
        super().__init__(node_iter)
        self.pattern = pattern
        self._re = self.make_re(pattern, ignorecase)

    @classmethod
    def make_re(cls, pattern, ignorecase=True):
        assert isinstance(pattern, str), "bad pattern: {!r}".format(pattern)
        assert isinstance(ignorecase, bool)
        flags = 0
        if ignorecase:
            flags |= re.IGNORECASE
        return re.compile(fnmatch.translate(pattern), flags=flags)

    def __iter__(self):
        for i in self._iter:
            if self.pattern == i.name or self._re.match(i.name):
                yield i
