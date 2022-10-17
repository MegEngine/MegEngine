# -*- coding: utf-8 -*-
import collections
import fnmatch
import itertools
import pickle
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

from ..core import _imperative_rt
from ..core._imperative_rt import ComputingGraph, SerializationMetadata
from ..core._trace_option import set_symbolic_shape as _set_symbolic_shape
from ..core.tensor import megbrain_graph as G
from ..logger import get_logger
from .comp_graph_tools import get_dep_vars, get_opr_type, get_oprs_seq
from .network_node import (
    ConstOpBase,
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
        self.all_oprs_map = OrderedDict()  # _imperative_rt.graph.VarNode.id: VarNode
        self.all_vars_map = (
            OrderedDict()
        )  # _imperative_rt.graph.OperatorNode.id: OpNode
        self.graph = ComputingGraph()
        self._metadata = None

    @property
    def metadata(self):
        r"""Load metadata as a dict."""
        if not self._metadata.is_valid:
            logger.info("metadata is not valid!")
            return None
        ret = dict()
        try:
            user_info = pickle.loads(self._metadata.user_info)
        except:  # pylint: disable=bare-except
            logger.warning(
                "can't parse user info by pickle, so return the original bytes object!"
            )
            user_info = self._metadata.user_info
        ret["user_info"] = user_info
        ret["graph_modified"] = self._metadata.graph_modified
        ret["optimized_for_inference"] = self._metadata.optimized_for_inference
        if ret["optimized_for_inference"]:
            ret.update(G.deserialize_infer_option(self._metadata.optimize_options))
        return ret

    @classmethod
    def load(cls, model_path: str, outspec: List[str] = None):
        r"""Loads a computing graph as a Network object.

        Args:
            model_path: file path of mge model.
            outspec: only load the subgraph with outspec as its endpoints.
        """
        self = cls()
        ret = G.load_graph(model_path)
        outputs, self._metadata = ret.output_vars_list, ret.metadata
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
            if isinstance(opr, (ConstOpBase, Host2DeviceCopy)):
                opr.compile(self.graph)
            else:
                opr.compile()
            if opr.name is not None:
                opr._opr.name = opr.name
            self.all_oprs_map[opr._opr.id] = opr
            for o in opr.outputs:
                self.all_vars_map[o.var.id] = o

    def optimize_for_inference(self, dest_vars, **kwargs):
        r"""Applies optimize_for_inference pass for operator graph.

        Args:
            dest_vars: list of output vars in the operator graph

        Keyword Arguments:

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
        * enable_nchw64 --
          whether to use NCHW64 data layout, used for fast int4
          support on Nvidia GPU.
        * enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
          into one opr.
        * enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z
          input for inference on nvidia backend(this optimization pass will
          result in mismatch of the precision of output of training and
          inference
        * enable_fuse_grain: fuse grain will be enable by default to fuse grain operator to huge operator, you can disable it.  
        )
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
        user_info: Any = None,
        enable_metadata=True,
        **kwargs
    ):
        r"""Serializes graph to file.

        Args:
            file: output file, could be file object or filename.
            append: whether output is appended to ``file``.
                Only works when ``file`` is str.
            keep_var_name: level for keeping variable names:

                * 0: none of the names are kept
                * 1: (default)keep names of output vars
                * 2: keep names of all (output and internal) vars

            keep_opr_name: whether to keep operator names.
            keep_param_name: whether to keep param names, so param values can be
                easily manipulated after loading model
            keep_opr_priority: whether to keep priority setting for operators
            strip_info_file: a string for path or a file handler. if is not None,
                then the dump information for code strip would be written to ``strip_info_file``
            append_json: will be check when `strip_info_file` is not None. if set
                true, the information for code strip will be append to strip_info_file.
                if set false, will rewrite strip_info_file
            optimize_for_inference: enbale optmizations,
                will skip all optimize options if this is False. Default: True
            user_info: any type object, which will be pickled to bytes.
            enable_metadata: whether to save metadata into output file.

        See more detials in :meth:`~.trace.dump`.
        """

        def _set_var_name(var):
            graph_var = G.VarNode(var.var)
            graph_var.name = var.name
            return graph_var

        self._compile()
        out = list(map(_set_var_name, self.output_vars))

        if kwargs.pop("arg_names", False):
            logger.warning(
                '"arg_names" is not supported in Network.dump, rename input vars directly'
            )
        if kwargs.pop("output_names", False):
            logger.warning(
                '"output_names" is not supported in Network.dump, rename output vars directly'
            )
        if optimize_for_inference:
            out, optimize_options = G.optimize_for_inference(out, **kwargs)

        metadata = SerializationMetadata()
        if enable_metadata:
            metadata.is_valid = True
            metadata.graph_modified = True
            metadata.user_info = pickle.dumps(user_info)
            if optimize_for_inference:
                metadata.optimize_options = optimize_options

        G.set_priority_to_id([o._node if isinstance(o, G.VarNode) else o for o in out])
        dump_content, dump_info = G.dump_graph(
            out,
            keep_var_name=keep_var_name,
            keep_opr_name=keep_opr_name,
            keep_param_name=keep_param_name,
            keep_opr_priority=keep_opr_priority,
            strip_info_file=strip_info_file,
            append_json=append_json,
            metadata=metadata,
        )
        if isinstance(file, str):
            permission = "wb" if append == False else "ab"
            file = open(file, permission)
        file.write(dump_content)
        return dump_info

    def make_const(self, data, name=None, device=None):
        r"""Makes an ImmutableTensor OpNode to provide a parameter for the network."""
        node = ImmutableTensor(data, name, device, self.graph)
        node.compile(self.graph)
        return node.outputs[0]

    def make_input_node(self, shape, dtype, name=None, device=None):
        r"""Makes a Host2DeviceCopy OpNode to provide an input varnode for the network."""
        node = Host2DeviceCopy(shape, dtype, name, device)
        node.compile(self.graph)
        return node.outputs[0]

    def add_output(self, *vars: VarNode):
        r"""Adds vars into the network output node list"""
        if not all([var.owner for var in vars]):
            self.add_dep_oprs(*vars)
        for var in vars:
            # use method 'is' instead of 'in' to avoid
            # compare VarNode use elemwise equal
            if not any(var is _ for _ in self.output_vars):
                self.output_vars.append(var)

    def remove_output(self, *vars: VarNode):
        r"""Removes vars from the network output node list"""
        for var in vars:
            # use list pop instead of remove to avoid
            # compare VarNode use elemwise equal
            is_removed = False
            for idx, out_var in enumerate(self.output_vars):
                if var is out_var:
                    self.output_vars.pop(idx)
                    is_removed = True
            if not is_removed:
                logger.warning(
                    "Failed to remove {}({}). Please check whether "
                    "this node is in the output list.".format(var.name, id(var))
                )

    def add_dep_oprs(self, *vars):
        if len(vars) == 0:
            vars = self.output_vars

        assert all(isinstance(var, VarNode) for var in vars), "Only support add VarNode"

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
        r"""Modifies names of operators **inplace**; useful for merging loaded
        network into another network

        Args:
            modifier(str or callable): a string to be prepended to the name, or a function
                that maps from name to name
        """
        if isinstance(modifier, str):
            om = modifier
            modifier = lambda v: "{}.{}".format(om, v)
        assert isinstance(modifier, collections.abc.Callable)
        for i in self.all_oprs:
            v0 = i.name
            v1 = modifier(v0)
            assert isinstance(v1, str)
            i.name = v1

    def reset_batch_size(self, batchsize, *, blacklist=()):
        r"""Helper for reset batch size; first dimension of all data providers
        not in blacklist are assumed to be the batch size

        Args:
            blacklist: data provider names whose first dimension is not
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
        self._compile()
        assert prev_batchsize is not None, "no data provider found"
        assert not blacklist, "unused items in blacklist: {}".format(blacklist)

    def replace_vars(self, repl_dict: Dict[VarNode, VarNode]):
        r"""Replaces vars in the graph.

        Args:
            repl_dict: the map {old_var: new_var} that specifies how to replace the vars.
        """
        if not all([var.owner for var in repl_dict.values()]):
            self.add_dep_oprs(*list(repl_dict.values()))
        for var in self.all_vars:
            if var in repl_dict:
                repl_var = repl_dict[var]
                if repl_var is var:
                    continue
                for opnode in var.users:
                    # use method 'is' instead of 'in' to avoid
                    # compare VarNode use elemwise equal
                    assert any([var is _ for _ in opnode.inputs])
                    opnode.inputs = [repl_var if var is i else i for i in opnode.inputs]
                    if opnode not in repl_var.users:
                        repl_var.users.append(opnode)
                var.users.clear()
        self._compile()

    def replace_oprs(self, repl_dict: Dict[OpNode, OpNode]):
        r"""Replaces operators in the graph.

        Args:
            repl_dict: the map {old_opr: new_opr} that specifies how to replace the operators.
        """
        for opr in self.all_oprs:
            if opr in repl_dict:
                assert len(opr.outputs) == len(
                    repl_dict[opr].outputs
                ), "can not replace {} with {}".format(type(opr), type(repl_dict[opr]))
                for ind, var in enumerate(opr.outputs):
                    var.owner = repl_dict[opr]
                    var.__dict__.update(repl_dict[opr].outputs[ind].__dict__)
                    var._reset_var(repl_dict[opr].outputs[ind].var)
                repl_dict[opr].outputs = opr.outputs
        self._compile()

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
        r"""Gets all oprs which use var as input"""
        return self.opr_filter.has_input(var).as_list()

    def get_dep_oprs(self, var):
        r"""Gets dependent oprs of var"""
        return get_oprs_seq(var, False, False)

    @property
    def opr_filter(self):
        r"""Filter on all opnodes of the Network."""
        oprs = self.all_oprs
        return NodeFilter(itertools.islice(oprs, len(oprs)))

    @property
    def var_filter(self):
        r"""Filter on all varnode of the Network."""
        vars = self.all_vars
        return NodeFilter(itertools.islice(vars, len(vars)))

    @property
    def params_filter(self):  # all immutable tensor
        r"""Filter on all parameters (ImmutableTensor Opr) of the Network"""
        return self.opr_filter.param_provider()

    @property
    def data_providers_filter(self):  # all host2devicecopy
        r"""Filter on all input nodes (Host2DeviceCopy Opr) of the Network"""
        return self.opr_filter.data_provider()

    @property
    def dest_vars(self):
        r"""Output varnodes of the Network."""
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

    def _add_opr(self, opr) -> Optional[OpNode]:
        r"""Used for loading and building graph."""
        assert isinstance(opr, _imperative_rt.graph.OperatorNode)

        # TODO: use megbrain C++ RTTI to replace type string
        if opr.id not in self.all_oprs_map:
            opnode = str_to_mge_class(get_opr_type(opr)).load(opr)
            self.all_oprs_map[opr.id] = opnode
            for var in opr.inputs:
                varnode = self._get_var(var)
                opnode.add_inp_var(varnode)
                varnode.users.append(opnode)
            for var in opr.outputs:
                opnode.add_out_var(self._get_var(var))
            return opnode
        else:
            # overwrite the opnode 'new' output VarNode with
            # original one when output number larger than 1,
            # or will cause dependence issue in _compiler step.
            if len(opr.outputs) > 1:
                opnode = self.all_oprs_map[opr.id]
                for idx, output in enumerate(opnode.outputs):
                    if output.var.id in self.all_vars_map:
                        opnode.outputs[idx] = self.all_vars_map[output.var.id]

            return None

    def _get_opr(self, x):
        if x.id in self.all_oprs_map:
            return self.all_oprs_map[x.id]
        else:
            return None

    def _get_var(self, x):
        r"""Convert :class:`~._imperative_rt.graph.VarNode` to :class:`~.VarNode`."""
        assert isinstance(x, _imperative_rt.graph.VarNode)
        if x.id not in self.all_vars_map or self.all_vars_map[x.id].var != x:
            self.all_vars_map[x.id] = VarNode.load(x, self._get_opr(x.owner))
        return self.all_vars_map[x.id]


def set_symbolic_shape(option: bool):
    r"""Set the VarNode use symbolic shape or not, return the last status.
    Please set to True and must recover after dump if want to change the input batch size.

    Args:
        option: True for enable symbolic shape.
    """
    return _set_symbolic_shape(option)


def as_varnode(obj):
    r"""convert a :class:`.utils.network_node.VarNode` compatible object to :class:`.utils.network_node.VarNode`.

    Args:
        obj: it must be one of the following:

            1. a :class:`.utils.network_node.VarNode` object
            2. a :class:`.utils.network_node.OpNode` object that has unique output
            3. an iterable that produces either type 1 or 2, with length 1

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
        obj, collections.abc.Iterable
    ), "{} is not compatible with VarNode".format(obj)

    val = list(obj)
    assert (
        len(val) == 1
    ), "can not convert sequence of length {} to VarNode ({})".format(
        len(val), (lambda s: s if len(s) < 50 else s[:50] + " ...")(str(val))
    )
    return as_varnode(val[0])


def as_oprnode(obj):
    r"""convert a :class:`.utils.network_node.OpNode` compatible object to
    :class:`.utils.network_node.OpNode`; it works like :func:`as_varnode`.
    """
    if type(obj) is VarNode:
        return obj.owner

    if isinstance(obj, OpNode):
        return obj

    assert isinstance(
        obj, collections.abc.Iterable
    ), "{} is not compatible with OpNode".format(obj)

    val = list(obj)
    assert (
        len(val) == 1
    ), "can not convert sequence of length {} to " "OpNode({})".format(len(val), val)
    return as_oprnode(val[0])


class NodeFilter:
    r"""Filter on node iterator. This class is an iterator of
    :class:`.NetworkNode` objects and multiple filtering conditions and
    mappers can be chained.

    Example:

        .. code-block::

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

        assert isinstance(node_iter, collections.abc.Iterable)
        if (not isinstance(node_iter, NodeFilter)) and type(
            self
        ) is not NodeFilterCheckType:
            node_iter = NodeFilterCheckType(node_iter, NetworkNode)
        self._iter = node_iter

    @classmethod
    def make_all_deps(cls, *dest_vars):
        r"""make a :class:`NodeFilter` that contains all deps of given vars"""
        return cls(list(get_oprs_seq(dest_vars, False, False)))

    def __iter__(self):
        r"""to be overwritten by subclass to implement filters"""
        return iter(self._iter)

    def type(self, node_type):
        r"""filter by specific node type

        Args:
            node_type: node type class

        Returns:
            a new :class:`NodeFilter` object
        """
        return NodeFilterType(self, node_type)

    def check_type(self, node_type):
        r"""assert that all oprs produced by this iterator are instances of
        certain type

        Args:
            node_type: node type class

        Returns:
            a new :class:`NodeFilter` object

        Raises:
            TypeError if type check failed
        """
        return NodeFilterCheckType(self, node_type)

    def not_type(self, node_type):
        r"""remove oprs of specific type

        Args:
            node_type: node type class

        Returns:
            a new :class:`NodeFilter` object
        """
        return NodeFilterNotType(self, node_type)

    def param_provider(self):
        r"""get :class:`~.ParamProvider` oprs; shorthand for
        ``.type(ParamProvider)``
        """

        return self.type(ImmutableTensor)

    def data_provider(self):
        r"""get :class:`.DataProvider` oprs; shorthand for
        ``.type(DataProvider)``
        """

        return self.type(Host2DeviceCopy)

    def name(self, pattern, ignorecase=True):
        r"""filter by node name

        Args:
            pattern(class:`str`): a string in glob syntax that can contain ``?`` and
                ``*`` to match a single or arbitrary characters.
            ignorecase(bool, optional): whether to ignroe case

        Returns:
            a new :class:`NodeFilter` object
        """
        return NodeFilterName(self, pattern, ignorecase)

    def has_input(self, var):
        r"""an opr is kept if it has given var as one of its inputs

        Args:
            var: var node to checked

        Returns:
            a new :class:`NodeFilter` object
        """
        return NodeFilterHasInput(self, var)

    def as_list(self):
        r"""consume this iterator and return its content as a list"""
        return list(self)

    def as_unique(self):
        r"""assert that this iterator yields only one node and return it

        Returns:
            class:`.GraphNodeBase`: the unique node

        Raises:
            ValueError if this iterator does not yield a unique node
        """
        (opr,) = self
        return opr

    def as_dict(self):
        r"""construct an ordered dict to map from node names to objects in
        this iterator
        """
        return collections.OrderedDict((i.name, i) for i in self)

    def as_count(self):
        r"""consume this iterator and get the number of elements"""
        return sum(1 for _ in self)


class NodeFilterType(NodeFilter):
    r"""see :meth:`NodeFilter.type`"""

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
    r"""see :meth:`NodeFilter.not_type`"""

    def __iter__(self):
        for i in self._iter:
            if not isinstance(i, self._node_type):
                yield i


class NodeFilterCheckType(NodeFilterType):
    r"""see :meth:`NodeFilter.check_type`"""

    def __iter__(self):
        for i in self._iter:
            if not isinstance(i, self._node_type):
                raise TypeError(
                    "all nodes should be {}; got {!r}".format(self._node_type, i)
                )
            yield i


class NodeFilterHasInput(NodeFilter):
    r"""see :meth:`NodeFilter.has_input`"""

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
    r"""see :meth:`NodeFilter.name`"""

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
