import contextlib
import functools
import typing
import weakref

from ..core.ops.special import Const
from ..core.tensor import megbrain_graph as G
from ..core.tensor.core import OpBase, apply
from ..core.tensor.raw_tensor import OpDef, RawTensor, as_raw_tensor


class TraceMismatchError(RuntimeError):
    pass


active_trace = None
skip_tracing = False


@contextlib.contextmanager
def exclude_from_trace():
    global skip_tracing
    if skip_tracing:
        yield
        return
    try:
        skip_tracing = True
        if active_trace is not None:
            active_trace._begin_excluded_region()
        yield
    finally:
        skip_tracing = False


class TensorInfo:
    __slots__ = (
        # collected attributes
        "external",
        "exported",
        "data_read",
        "shape_read",
        "value_read",
        "device",
        "dtype",
        "bound_data",
        # resources for execution
        "varnode",
        "data_setter",
        "shape_reader",
        "value_reader",
        "data_reader",
    )

    def __init__(self):
        self.exported = None
        self.data_read = None
        self.shape_read = None
        self.value_read = None
        self.bound_data = None

        self.data_setter = None
        self.shape_reader = None
        self.value_reader = None
        self.data_reader = None


class trace:
    def __new__(cls, *args, **kwargs):
        if not args:
            return functools.partial(cls, **kwargs)
        self = super().__new__(cls)
        self.__init__(*args, **kwargs)
        return self

    def __init__(self, function, symbolic=False, capture_as_const=False):
        self.__wrapped__ = function
        self._symbolic = symbolic
        self._capture_as_const = capture_as_const
        self._capture_static_shape = False

        self._untraced = True
        self._tinfo = []  # handle -> TensorInfo
        self._seq = []
        self._pc = 0
        self._graph = None
        self._need_reset_nodes = None
        self._lazy_eval_graph = None
        self._lazy_eval_tensors = weakref.WeakSet()
        self._active_tensors = weakref.WeakSet()

    def _new_handle(self):
        handle = len(self._tinfo)
        info = TensorInfo()
        self._tinfo.append(info)
        return handle, info

    def _apply_op(self, op, args):
        assert not self._untraced
        # check against trace
        if self._pc >= len(self._seq):
            raise TraceMismatchError("trace should end here, but more op observed")
        record = self._seq[self._pc]
        op_, ihandles, ohandles = record
        if op != op_:
            raise TraceMismatchError("op different from last time")
        if len(ihandles) != len(args):
            raise TraceMismatchError("op input size different from last time")

        for h, x in zip(ihandles, args):
            info = self._tinfo[h]
            if info.external:
                if (
                    x.__class__ is CompiledTensorProxy
                    and not self._tinfo[x._CompiledTensorProxy__handle].exported
                ):
                    raise TraceMismatchError(
                        "failed to capture: input was an external tensor "
                        "last time, got an internal tensor this time"
                    )
                if info.bound_data:
                    if x.__class__ is CompiledTensorProxy:
                        raise TraceMismatchError(
                            "const capture violated: was an external tensor "
                            "last time, got an internal tensor this time"
                        )
                    if x._handle != info.bound_data._handle:
                        raise TraceMismatchError(
                            "const capture violated: got "
                            "a different tensor this time"
                        )
                else:
                    if info.dtype != x.dtype:
                        raise TraceMismatchError(
                            "failed to capture: different dtype from last time"
                        )
                    if info.device != x.device:
                        raise TraceMismatchError(
                            "failed to capture: different device from last time"
                        )
                    info.data_setter.set_value(x._dev_tensor())
            else:
                if x.__class__ is not CompiledTensorProxy:
                    raise TraceMismatchError(
                        "unexpected capture: trying to use an external tensor as input, "
                        "but that input was an internal tensor last time"
                    )
                if x._CompiledTensorProxy__handle != h:
                    raise TraceMismatchError(
                        "mis-wiring: input edge to an data flow "
                        "graph node is different from last time"
                    )

        self._pc += 1
        outputs = tuple([CompiledTensorProxy(h) for h in ohandles])
        self._active_tensors.update(outputs)
        return outputs

    def _record_op(self, op, inputs, outputs):
        if skip_tracing:
            for x in inputs:
                h = getattr(x, "_TraceMixin__handle", None)
                if h is not None:
                    self._tinfo[h].data_read = True
            return

        ihandles = []
        for x in inputs:
            h = getattr(x, "_TraceMixin__handle", None)
            if h is None or (not self._capture_as_const and self._tinfo[h].exported):
                h, info = self._new_handle()
                info.external = True
                info.device = x.device
                info.dtype = x.dtype
                if self._capture_as_const:
                    info.bound_data = x

            ihandles.append(h)

        ohandles = []
        for x in outputs:
            h, info = self._new_handle()
            ohandles.append(h)
            info.external = False
            TraceMixin._TraceMixin__inject(x, h)

        self._seq.append((op, tuple(ihandles), tuple(ohandles)))
        self._active_tensors.update(outputs)

    @contextlib.contextmanager
    def _setup(self):
        global active_trace
        if active_trace:
            raise NotImplementedError("sorry, not implemented: nested trace")
        active_trace = self

        if self._untraced:
            apply.enable(apply_with_tracing)
            if self._symbolic:
                apply.enable(apply_symbolic_mode)
                self._lazy_eval_graph = G.Graph()
        else:
            apply.enable(apply_compiled_mode)
            if self._graph is None:
                self._compile()
            self._graph.execute()

        yield

        escaped_tensors = tuple(self._active_tensors)
        self._active_tensors.clear()

        if self._untraced:
            for x in escaped_tensors:
                info = self._tinfo[x._TraceMixin__handle]
                info.data_read = True
                x._TraceMixin__restore()
            if self._symbolic:
                # eval lazy eval tensors
                lazy_eval_tensors = tuple(self._lazy_eval_tensors)
                if lazy_eval_tensors:
                    readers = [
                        G.OutputNode(x._LazyEvalTensor__varnode).outputs[0]
                        for x in lazy_eval_tensors
                    ]
                    self._lazy_eval_graph.compile(*readers)
                    self._lazy_eval_graph()
                    for r, x in zip(readers, lazy_eval_tensors):
                        assign_raw_tensor(x, as_raw_tensor(r.op.get_value()))
                    self._lazy_eval_graph = None
                    self._lazy_eval_tensors = None
            self._untraced = False
        else:
            if self._pc != len(self._seq):
                raise TraceMismatchError("premature end")
            for x in escaped_tensors:
                assign_raw_tensor(x, as_raw_tensor(x._dev_tensor()))
            self._graph.wait()
            self._reset_exec_env()
            self._pc = 0

        apply.disable(apply_with_tracing)
        apply.disable(apply_symbolic_mode)
        apply.disable(apply_compiled_mode)
        active_trace = None

    def _begin_excluded_region(self):
        if self._untraced:
            # conditionally reading a compiled tensor in excluded region
            # is permitted, so we have to assume every tensor might be read
            for x in self._active_tensors:
                info = self._tinfo[x._TraceMixin__handle]
                info.exported = True
                info.data_read = True

    def _compile(self):
        graph = self._graph = G.Graph()
        graph.options.no_force_inplace = True
        # graph.options.graph_opt_level = 0
        need_reset_nodes = self._need_reset_nodes = []
        # links enforce ordering of I/O nodes
        links = ()
        for op, ihandles, ohandles in self._seq:
            ivars = []
            readers = []
            for h in ihandles:
                info = self._tinfo[h]
                if not hasattr(info, "varnode"):
                    assert info.external
                    if info.bound_data:
                        info.varnode = graph.make_const(info.bound_data._dev_tensor())
                    else:
                        opnode = info.data_setter = G.InputNode(
                            *links, device=info.device, dtype=info.dtype, graph=graph
                        )
                        need_reset_nodes.append(opnode)
                        info.varnode, *links = opnode.outputs

                ivars.append(info.varnode)
            ovars = apply(op, *ivars)
            assert len(ovars) == len(ohandles)
            for h, v in zip(ohandles, ovars):
                info = self._tinfo[h]
                info.varnode = v

                def add_reader(opnode):
                    nonlocal links
                    need_reset_nodes.append(opnode)
                    readers.append(opnode.outputs[0])
                    links = opnode.outputs

                if info.data_read:
                    # Shape can be obtained from data so doesn't need its own
                    # output node. On the other hand, value is read separately
                    # to leverage eager h2d copy
                    info.shape_read = False
                    opnode = info.data_reader = G.OutputNode(v, *links)
                    add_reader(opnode)
                if info.value_read:
                    opnode = info.value_reader = G.ValueOutputNode(v, *links)
                    add_reader(opnode)
                if info.shape_read:
                    opnode = info.shape_reader = G.AttrOutputNode(v, *links)
                    add_reader(opnode)

        graph.compile(*readers)

    def _reset_exec_env(self):
        for opnode in self._need_reset_nodes:
            opnode.reset()

    def _require_shape(self, handle):
        info = self._tinfo[handle]
        info.shape_read = True

    def _require_value(self, handle):
        info = self._tinfo[handle]
        info.value_read = True

    def _require_data(self, handle):
        info = self._tinfo[handle]
        info.data_read = True

    def __call__(self, *args, **kwargs):
        with self._setup():
            return self.__wrapped__(*args, **kwargs)


class CompiledTensorProxy(RawTensor):
    """
    Duck-typed RawTensor
    """

    def __init__(self, handle):
        self.__handle = handle
        self.__info = active_trace._tinfo[handle]
        self.__shape = None
        self.__data = None
        self.__value = None

    @property
    def dtype(self):
        return self.__info.varnode.dtype

    @property
    def device(self):
        return self.__info.varnode.device

    @property
    def shape(self):
        if self.__shape is None:
            if self.__info.shape_read:
                self.__shape = self.__info.shape_reader.get_value().shape
            elif self.__info.data_read:
                self.__shape = self._dev_tensor().shape
            else:
                raise TraceMismatchError("shape of this tensor is not read in trace")
        return self.__shape

    def numpy(self):
        if self.__value is None:
            if self.__info.value_read:
                self.__value = self.__info.value_reader.get_value()
            elif self.__info.data_read:
                self.__value = self._dev_tensor().numpy()
            else:
                raise TraceMismatchError("value of this tensor is not read in trace")
        return self.__value

    def _dev_tensor(self):
        if self.__data is None:
            if not self.__info.data_read:
                raise TraceMismatchError("raw data of this tensor is not read in trace")
            self.__data = self.__info.data_reader.get_value()
        return self.__data

    def __del__(self):
        if self.__info.shape_read and self.__shape is not None:
            self.__info.shape_reader.drop_value()
        if self.__info.value_read and self.__value is not None:
            self.__info.value_reader.drop_value()
        if self.__info.data_read and self.__data is not None:
            self.__info.data_reader.drop_value()


class LazyEvalTensor(RawTensor):
    def __init__(self, varnode):
        self.__varnode = varnode

    @property
    def dtype(self):
        return self.__varnode.dtype

    @property
    def device(self):
        return self.__varnode.device

    @property
    def shape(self):
        return self.__varnode.shape

    def numpy(self):
        return self.__varnode.value

    def _dev_tensor(self):
        raise RuntimeError("cannot access data during symbolic tracing")


class TraceMixin:
    __subclass_cache = {}

    def __inject(self, handle):
        cache = __class__.__subclass_cache
        cls = self.__class__
        subcls = cache.get(cls)
        if subcls is None:
            subcls = cache[cls] = type("Traced" + cls.__name__, (__class__, cls), {})
        self.__class__ = subcls
        self.__handle = handle
        self.__cls = cls
        return self

    def __restore(self):
        cls = self.__cls
        del self.__handle
        del self.__cls
        self.__class__ = cls
        return self

    @property
    def shape(self):
        if not skip_tracing:
            active_trace._require_shape(self.__handle)
        return super().shape

    def numpy(self):
        if not skip_tracing:
            active_trace._require_value(self.__handle)
        return super().numpy()

    def _dev_tensor(self):
        if not skip_tracing:
            active_trace._require_data(self.__handle)
        return super()._dev_tensor()


class TracedRawTensor(TraceMixin, RawTensor):
    pass


class TracedLazyTensor(TraceMixin, LazyEvalTensor):
    pass


def assign_raw_tensor(lhs, rhs):
    handle = rhs._handle
    rhs.__dict__.clear()
    lhs.__dict__.clear()
    lhs.__class__ = RawTensor
    lhs.__init__(handle)


# this hook turns RawTensor into LazyEvalTensor
@apply.register()
def apply_symbolic_mode(op: OpDef, *args: RawTensor):
    graph = active_trace._lazy_eval_graph
    ivars = [
        getattr(x, "_LazyEvalTensor__varnode", None)
        or graph.make_const(x._dev_tensor())
        for x in args
    ]
    ovars = apply(op, *ivars)
    outputs = [LazyEvalTensor(v) for v in ovars]
    active_trace._lazy_eval_tensors.update(outputs)
    return outputs


apply.disable(apply_symbolic_mode)


@apply.register()
def apply_compiled_mode(op: OpDef, *args: RawTensor):
    if skip_tracing:
        args = [
            as_raw_tensor(x._dev_tensor()) if x.__class__ is CompiledTensorProxy else x
            for x in args
        ]
        return apply.super(op, *args)
    return active_trace._apply_op(op, args)


apply.disable(apply_compiled_mode)


# this hook injects TraceMixin
@apply.register()
def apply_with_tracing(op: OpDef, *args: RawTensor):
    outputs = apply.super(op, *args)
    active_trace._record_op(op, args, outputs)
    return outputs


apply.disable(apply_with_tracing)


# @apply.register()
# def _(op: Const, *args: RawTensor):
#     return active_trace._apply_const(op, args)


class BrokenRawTensor(RawTensor):
    def __getattribute__(self, _):
        raise RuntimeError("broken due to misuse of tracing")

    def __setattr__(self, *_):
        raise RuntimeError("broken due to misuse of tracing")
