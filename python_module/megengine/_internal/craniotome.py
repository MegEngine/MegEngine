# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""used for creating a megbrain operator from python"""

import copy
import itertools
from abc import ABCMeta, abstractmethod, abstractproperty

from . import helper as _helper
from . import mgb as _mgb


class _CraniotomeBaseMeta(ABCMeta):
    _base_created = False

    def __init__(cls, name, bases, member_dict):
        if _CraniotomeBaseMeta._base_created:
            assert "__init__" not in member_dict, (
                "Craniotome operators should not overwrite __init__ method; "
                "use setup() instead."
            )
            forbidden = set(
                k for k in dir(CraniotomeBase) if k[0] == "_" and k[1] != "_"
            )
            forbidden.add("get_io_vars")
            check_key = member_dict.get("__check_key__", True)
            whitelist = ["__classcell__"]
            for k in member_dict.keys():
                assert k not in forbidden, "{} could not be overwritten".format(k)
                if (
                    check_key
                    and k.startswith("__")
                    and k.endswith("__")
                    and k not in whitelist
                    and not hasattr(CraniotomeBase, k)
                ):
                    raise KeyError(
                        "name {} in class {} does not exist in the baseclass".format(
                            k, name
                        )
                    )
        else:
            _CraniotomeBaseMeta._base_created = True
        super().__init__(name, bases, member_dict)


class CraniotomeBase(_mgb.CraniotomeDesc, metaclass=_CraniotomeBaseMeta):
    """base class used for extending megbrain core operators in python

    Note: all names starting and ending with two underscores in the subclasses
    would be checked and KeyError would be raised if the name does not exist in
    the base class. This behavor can be disabled by setting ``__check_key__``
    to ``False`` (see the testcase for more details)
    """

    # methods and attributes to be overwritten by subclasses

    __expand_single_outputs__ = True
    """if :attr:`__nr_outputs__` is 1, whether to return a single
    :class:`.SymbolVar` instead of a tuple in :meth:`make`"""

    __is_dynamic_output_shape__ = False
    """whether output shape could not be inferred from input shape. If value of
    this attribute is ``False``, :meth:`infer_shape` must be implemented. If
    this attribute is ``True`` but the operator has no inputs, then
    :meth:`infer_shape` would also be called to infer output shape before
    operator execution.
    """

    __disable_sys_mem_alloc__ = False
    """whether to disable system memory allocator. This is used when
    :attr:`__is_dynamic_output_shape__` is ``False`` but the output memory
    should not be managed by megbrain system (so it can be forwarded from
    external buffer)"""

    __allow_duplicate__ = True
    """whether this operator can be duplicated (e.g. used in sublinear
    memory)"""

    __allow_empty_out__ = False
    """whether empty output shape is allowed; if it is set as ``False``, then
    an exception would be raised if output var is empty to prevent erroneously
    forgetting initializing output vars"""

    @abstractproperty
    def __nr_inputs__(self):
        """number of input vars"""

    @abstractproperty
    def __nr_outputs__(self):
        """number of output vars"""

    @abstractmethod
    def execute(self, inputs, outputs):
        """execute the operator, read values from *inputs* by calling
        :meth:`.CompGraphCallbackValueProxy.get_value` and write results into
        *outputs* by calling :meth:`.SharedND.set_value`

        :param inputs: values for each input var
        :type inputs: tuple of :class:`.CompGraphCallbackValueProxy`
        :param outputs: values for each output var
        :type outputs: tuple of :class:`.SharedND`
        """

    def setup(self):
        """overwritten by subclass to accept kwargs passed to :meth:`make` to
        setup the operator"""

    def infer_shape(self, inp_shapes):
        """infer output shape from input shapes

        :type inp_shapes: tuple of tuple of ints
        :param inp_shapes: input shapes for each input var
        :rtype: tuple of tuple of ints
        :return: output shapes for each output var
        """
        raise NotImplementedError(
            "{}: infer_shape() not implemented; for operators with dynamic "
            "output shape, __is_dynamic_output_shape__ should be set to True".format(
                self
            )
        )

    def grad(self, wrt_idx, inputs, outputs, out_grad):
        """compute symbolic gradient; should be overwritten by differentiable
        subclasses

        :type wrt_idx: int
        :param wrt_idx: the input var with respect to which the gradient should
            be computed; please also see the notes below
        :type inputs: tuple of :class:`.SymbolVar`
        :param inputs: input symbol vars
        :type outputs: tuple of :class:`.SymbolVar`
        :param outputs: output symbol vars
        :type out_grad: tuple of (:class:`.SymbolVar` or None)
        :param out_grad: gradients of loss with respect to each output var

            .. note::

                In case when loss does not depend on some var (i.e. zero grad),
                the corresponding value in *out_grad* would be ``None``. It is
                guaranteed that at least one element in *out_grad* is not
                ``None``.

        .. note::

            This function can return either of the following:

                1. Gradient of the input specified by ``wrt_idx``
                2. A list containing gradients of all inputs. In this case,
                   ``wrt_idx`` can be ignored.

            And the so called gradient can be either one of:

                1. A :class:`.SymbolVar` representing the symbolic gradient
                   value
                2. ``0`` representing zero gradient
        """
        raise NotImplementedError("grad for {} not implemented".format(self))

    def init_output_dtype(self, input_dtypes):
        """infer output dtypes from input dtypes; return None to use default
        infer function in megbrain.

        .. note::
            This method must be implemented if there is no input var

        :param input_dtypes: input dtypes
        :type input_dtypes: list of :class:`numpy.dtype`
        :rtype: None or list of :class:`numpy.dtype`-compatible
        """

    def get_serialize_params(self):
        """get params for megbrain graph serialization. This function should
        return a list or tuple, containing one or two elements: the first
        element must be a string, representing the name passed to
        ``opr_loader_maker`` during deserializing; the second element, if
        exists, must be convertible to ``bytes`` and is used for dumping any
        extra opr params, which can be retrieved by ``load_buf_with_len``
        during deserializing.
        """
        raise NotImplementedError(
            "get_serialize_params() for {} not implemented".format(self)
        )

    def copy(self):
        """copy this craniotome descriptor; the default implementation creates
        a new object, and copies object ``__dict__``"""
        ret = type(self)()
        d0 = self.__dict__.copy()
        d0.pop("this")
        ret.__dict__.update(copy.deepcopy(d0))
        return ret

    def on_graph_compiled(self, used_outputs):
        """a callback that would be invoked when the graph is compiled; it
        would always have a matching :meth:`on_compiled_func_deleted` call

        :param used_outputs: indices of outputs that are needed for the
            computation
        :type used_outputs: ``tuple of int``
        """

    def on_compiled_func_deleted(self):
        """a callback that would be invoked when the compiled function is
        destructed; it would always have a matching :meth:`on_graph_compiled`
        call"""

    def get_io_vars(self):
        """get input vars, comp order dep vars and output vars

        :return: a dict with keys ``'input'``, ``'output'`` and
            ``'comp_order'`` that maps to corresponding list of vars
        """
        all_vars = list(self._get_all_io_vars())
        nr_inp = self.__nr_inputs__
        nr_out = self.__nr_outputs__
        nr_comp_order = self._get_nr_dev_comp_order_deps()
        s0 = nr_inp + nr_comp_order
        return dict(
            input=all_vars[:nr_inp],
            comp_order=all_vars[nr_inp:s0],
            output=all_vars[s0:],
        )

    @property
    def owner_opr_id(self):
        """ID of the operator that owns this descriptor"""
        return self._get_opr_id()

    @property
    def comp_node(self):
        """comp node on which this operator runs"""
        return self._get_comp_node()

    # below are methods that should not be changed

    def _hash(self):
        return int(hash(self)) % (1 << 64)

    def _setup_self(self, dst):
        dst.append(self)

    def _is_same(self, rhs):
        return bool(self == rhs)

    def _node_flag(self):
        return (
            (int(bool(self.__is_dynamic_output_shape__)) << 0)
            | (int(not self.__allow_duplicate__) << 1)
            | (int(bool(self.__allow_empty_out__)) << 2)
            | (int(bool(self.__disable_sys_mem_alloc__)) << 3)
        )

    def _get_opr_type_name(self):
        return str(self.__class__.__name__)

    def _get_nr_outputs(self):
        return int(self.__nr_outputs__)

    def _execute(self, inputs, outputs):
        inputs = tuple(inputs)
        outputs = tuple(outputs)
        if not self.__is_dynamic_output_shape__:
            out_shapes = [i.shape for i in outputs]
        self.execute(inputs, outputs)
        if not self.__is_dynamic_output_shape__:
            new_shapes = [i.shape for i in outputs]
            assert (
                out_shapes == new_shapes
            ), "output shape changed after executing {}: before={} after={}".format(
                self, out_shapes, new_shapes
            )

    def _infer_shape(self, inp_shapes):
        inp_shapes = tuple(tuple(map(int, i)) for i in inp_shapes)
        oshp_get = self.infer_shape(inp_shapes)
        assert (
            len(oshp_get) == self.__nr_outputs__
        ), "{}: expect {} outputs; got {}(val: {}) from infer_shape".format(
            self, self.__nr_outputs__, len(oshp_get), oshp_get
        )
        return _helper.cvt_to_vector_of_shape(oshp_get)

    def _grad(self, wrt_idx, inputs, outputs, out_grad):
        og = []
        for i in out_grad:
            if i.valid:
                og.append(i)
            else:
                og.append(None)
        rst = self.grad(int(wrt_idx), tuple(inputs), tuple(outputs), tuple(og))
        if not isinstance(rst, (list, tuple)):
            rst = [rst]
        else:
            assert len(rst) == len(
                inputs
            ), "{}: opr has {} inputs but {} grads are returned".format(
                self, len(inputs), len(rst)
            )

        for i in range(len(rst)):
            cur = rst[i]
            if cur is 0:
                rst[i] = _mgb.SymbolVar()
            else:
                assert isinstance(cur, _mgb.SymbolVar), (
                    "{}: invalid grad result; it should be either "
                    "0 or a SymbolVar, got {!r} instead".format(self, cur)
                )
        return rst

    def _get_nr_dev_comp_order_deps(self):
        return 0

    def _init_output_dtype(self, input_dtypes, ret):
        get = self.init_output_dtype(input_dtypes)
        if get is not None:
            assert isinstance(ret, (list, tuple)) and len(get) == len(ret)
            ret[:] = get
            return True
        assert self.__nr_inputs__, (
            "{}: init_output_dtype must be implemented "
            "if there is no input var".format(self)
        )
        return False

    def _setup_serialize_params(self, output):
        val = list(self.get_serialize_params())
        assert len(val) in [1, 2]
        name = val[0]
        assert isinstance(name, str)
        output.append(name)
        if len(val) == 2:
            output.append(bytes(val[1]))

    def _copy(self):
        ret = self.copy()
        assert type(ret) is type(
            self
        ), "copy() returned different type: src={} copied={}".format(
            type(self), type(ret)
        )
        assert ret is not self
        ret.__disown__()
        self._set_copy_result(ret)

    def _on_graph_compile_or_func_del(self, used_outputs):
        if used_outputs:
            self.on_graph_compiled(used_outputs)
        else:
            self.on_compiled_func_deleted()

    def __repr__(self):
        return "cranoiotome:{}".format(self.__class__.__name__)

    @classmethod
    def make(
        cls,
        *inputs,
        comp_graph=None,
        name=None,
        comp_node=None,
        config=None,
        dev_comp_order_deps=[],
        **kwargs
    ):
        """apply this operator on some input vars and return corresponding
        output vars

        :type inputs: tuple of :class:`.SymbolVar`
        :param inputs: input symvars; immediate values could also be accepted,
            as long as there is symvar to infer comp node and comp graph
        :param comp_graph: if there is no input vars, *comp_graph* must be
            provided to specify which computing graph to insert this operator
        :param dev_comp_order_deps: vars that must have been computed
            before executing this operator
        :param kwargs: extra keyword arguments to be passed to :meth:`setup` of
            this class
        :param name: name of the resulting operator
        :rtype: tuple of :class:`.SymbolVar`
        :return: output symvars
        """

        if not inputs and not dev_comp_order_deps:
            assert isinstance(
                comp_graph, _mgb.CompGraph
            ), "{}: comp_graph must be given if no inputs provided".format(self)

        desc = cls()
        desc.setup(**kwargs)
        assert (
            len(inputs) == desc.__nr_inputs__
        ), "{}: expected {} inputs, got {}".format(
            desc, desc.__nr_inputs__, len(inputs)
        )

        config = _helper.gen_config(name, comp_node, config)

        # get inp_vec
        inp_vec = _mgb._VectorSymbolVar()
        for i in _helper.canonize_input_vars(
            itertools.chain(inputs, dev_comp_order_deps),
            comp_graph=comp_graph,
            config=config,
        ):
            inp_vec.push_back(i)
        desc._get_nr_dev_comp_order_deps = lambda *, val=len(dev_comp_order_deps): val

        if comp_graph is not None:
            desc._get_comp_graph = lambda: comp_graph
        expand_single_outputs = desc.__expand_single_outputs__
        desc.__disown__()
        rst = _mgb.make_opr_from_craniotome_desc(desc, inp_vec, config)
        if expand_single_outputs and len(rst) == 1:
            return rst[0]
        return tuple(rst)


def make_opr(cls):
    """decorator used to wrap a :class:`.CraniotomeBase` subclass and return
    its :meth:`~.CraniotomeBase.make` method
    """
    assert issubclass(cls, CraniotomeBase)
    return cls.make
