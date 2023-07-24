import io
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from ..core._imperative_rt import ops as mops
from ..core._imperative_rt.core2 import OpInfo, VarInfo
from . import dtype
from .lib.mlir import ir
from .lib.mlir.dialects import hlo

func_id = 0


def _default_func_name():
    global func_id
    func_id += 1
    return f"please_realize_func_name_system_{func_id}"


def _is_rng_op(opr):
    return isinstance(
        opr,
        (
            mops.Dropout,
            mops.BetaRNG,
            mops.GammaRNG,
            mops.GaussianRNG,
            mops.PermutationRNG,
            mops.PoissonRNG,
            mops.ShuffleRNG,
            mops.UniformRNG,
        ),
    )


class AbstractVar:
    def __init__(self, _id, _shape, _dtype) -> None:
        self.id = _id
        self.shape = _shape
        self.dtype = _dtype
        self.bound_data = None


class Pass(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, tr) -> Any:
        pass


# because xla pass key as a tensor, while mge pass key as a param, so we need to add a
# rng key tensor to the graph and set it as the input of the graph and rng op
class RngKeyAdder(Pass):
    def __call__(self, tr) -> Any:
        has_rng_opr = False

        for eqn in tr.eqns:
            if _is_rng_op(eqn.op):
                has_rng_opr = True
                break

        if not has_rng_opr:
            return tr

        # it should be [2, np.uint64], however, megengine donot support np.uint64/np.int64/np.uint32
        inp_rng_state_var = AbstractVar(tr.next_vid, [2, 2], np.dtype(np.int32))
        tr.add_input(inp_rng_state_var)

        new_eqns = []
        for eqn in tr.eqns:
            if not _is_rng_op(eqn.op):
                new_eqns.append(eqn)
                continue

            oup_rng_state_var = AbstractVar(tr.next_vid, [2, 2], np.dtype(np.int32))
            tr.add_var(oup_rng_state_var)

            inputs, outputs = list(eqn.inputs), list(eqn.outputs)
            inputs.append(inp_rng_state_var.id)
            outputs.append(oup_rng_state_var.id)
            new_eqn = OpInfo(eqn.op, inputs, outputs, eqn.kind)
            new_eqns.append(new_eqn)
            inp_rng_state_var = oup_rng_state_var

        tr.eqns = new_eqns
        tr.set_var_as_oup(inp_rng_state_var)

        return tr


# in megengine, dropout return a bit-mask while xla hard to represent, so we let xla
# return a uint8 mask, which means the mask is 8 times larger than mge
class DropoutMaskCanonicalizer(Pass):
    def __call__(self, tr) -> Any:
        for eqn in tr.eqns:
            if not isinstance(eqn.op, mops.Dropout):
                continue

            inputs, outputs = list(eqn.inputs), list(eqn.outputs)
            mask_var = tr.vars[outputs[1]]
            inp_shape = tr.vars[inputs[0]].shape
            new_mask_var = AbstractVar(
                mask_var.id, (int(np.prod(inp_shape)),), mask_var.dtype
            )
            tr.vars[mask_var.id] = new_mask_var

        return tr


class TraceResult:
    def __init__(self, traced, func_name=None) -> None:
        self.func_name = func_name if func_name is not None else _default_func_name()
        self.traced = traced
        self.eqns = []
        self.vars = {}
        self.inputs = []
        self.outputs = []
        self.consts = []
        self.custom_vid = 0

        self.effects = []

        for var in self.traced.vars:
            self.add_var(var)
            self.custom_vid = max(var.id + 1, self.custom_vid)

            if var.kind == "external" and var.inp_mark:
                self.inputs.append(var.id)

            if var.data_required:
                self.outputs.append(var.id)

            if var.kind == "const":
                self.consts.append(var.id)

        for op in self.traced.ops:
            self.eqns.append(op)

    @property
    def _var_inputs(self):
        return [self.vars[i] for i in self.inputs]

    @property
    def _var_outputs(self):
        return [self.vars[i] for i in self.outputs]

    @property
    def _var_consts(self):
        return [self.vars[i] for i in self.consts]

    @property
    def next_vid(self):
        ret = self.custom_vid
        self.custom_vid += 1
        return ret

    def add_var(self, var):
        assert var.id not in self.vars
        self.vars[var.id] = var

    def add_input(self, inp_var):
        self.add_var(inp_var)
        self.inputs.append(inp_var.id)

    def set_var_as_oup(self, oup_var):
        assert oup_var.id in self.vars
        self.outputs.append(oup_var.id)

    def get_var(self, idx):
        assert isinstance(idx, int)
        return self.vars[idx]

    def is_input(self, var):
        if isinstance(var, int):
            var = self.vars[var]
        return var.kind == "external"

    def is_output(self, var):
        if isinstance(var, int):
            var = self.vars[var]
        return var.data_required

    def _str_var(self, var):
        def _str_shape(shp):
            return "x".join([str(d) for d in shp])

        dtype_to_str = {
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32",
            "int64": "i64",
            "uint8": "u8",
            "uint32": "u32",
            "uint64": "u64",
            "bool": "i1-bool",
        }

        if isinstance(var, int):
            var = self.vars[var]
        var_dtype = None
        try:
            var_dtype = dtype_to_str[str(var.dtype)]
        except RuntimeError:
            var_dtype = "unknown"

        var_bound_data = (
            ("," + ",".join(str(var.bound_data).split()))
            if var.bound_data is not None and var.bound_data.size < 5
            else ""
        )

        return f"{var.id}%:<{_str_shape(var.shape)},{var_dtype}{var_bound_data}>"

    def _str_eqn(self, eqn):
        inps = ", ".join(map(self._str_var, eqn.inputs))
        oups = ", ".join(map(self._str_var, eqn.outputs))
        str_op = str(eqn.op)
        if isinstance(eqn.op, mops.Reduce):
            assert str(eqn.op.mode).startswith("Reduce.Mode.")
            str_op = str_op + str(eqn.op.mode)[len("Reduce.Mode.") :]
        ret = f"{oups} = {str_op}({inps})"
        return ret

    def __str__(self) -> str:
        func_inps_str = ", ".join(map(self._str_var, self.inputs))
        func_oups_str = ", ".join(map(self._str_var, self.outputs))
        func_const_str = "\n        ".join(map(self._str_var, self.consts))
        ret = f"{self.func_name}({func_inps_str}) -> ({func_oups_str}) {{\n    "
        if len(self.consts) > 0:
            ret += f"const:\n        {func_const_str}\n    "
        ret += "\n    ".join(map(self._str_eqn, self.eqns))
        ret += "\n}"
        return ret


_dtype_to_ir_type: Dict[np.dtype, Callable[[], ir.Type]] = {
    np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
    np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
    np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
    np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
    np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
    np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
    np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
    np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
    np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
    np.dtype(dtype.bfloat16): ir.BF16Type.get,
    np.dtype(np.float16): ir.F16Type.get,
    np.dtype(np.float32): ir.F32Type.get,
    np.dtype(np.float64): ir.F64Type.get,
    np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
    np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}


def mge_dtype_to_ir_type(mge_dtype):
    mge_dtype = np.dtype(mge_dtype)
    assert isinstance(
        mge_dtype, np.dtype
    ), f"arg should be numpy dtype, but is {mge_dtype}"
    ir_type_factory = _dtype_to_ir_type[mge_dtype]
    return ir_type_factory()


def mge_varinfo_to_ir_type(mge_varinfo):
    assert isinstance(mge_varinfo, (VarInfo, AbstractVar)), "args should be VarInfo"
    shape = mge_varinfo.shape
    return ir.RankedTensorType.get(shape, mge_dtype_to_ir_type(mge_varinfo.dtype))


def mge_varinfo_to_ir_type_tuple(mge_varinfo):
    return (mge_varinfo_to_ir_type(mge_varinfo),)


def make_ir_type_according_meta(src_shape: Tuple, src_dtype: np.dtype):
    return ir.RankedTensorType.get(src_shape, mge_dtype_to_ir_type(src_dtype))


def make_ir_type_according_meta_tuple(src_shape: Tuple, src_dtype: np.dtype):
    return (make_ir_type_according_meta(src_shape, src_dtype),)


_constant_handlers = {}


def _numpy_array_constant(x: np.ndarray, canonicalize_types) -> Sequence[ir.Value]:
    if canonicalize_types:
        x = np.asarray(x, dtype.canonicalize_dtype(x.dtype))
    element_type = mge_dtype_to_ir_type(x.dtype)
    shape = x.shape
    if x.dtype == np.bool_:
        nelems = x.size
        x = np.packbits(x, bitorder="little")
        if nelems == 1:
            x = np.array(0 if x.item() == 0 else 0xFF, np.uint8)
    elif x.dtype == dtype.bfloat16:
        x = x.view(np.uint16)
    x = np.ascontiguousarray(x)
    attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)
    return (hlo.ConstantOp(attr).result,)


def _ndarray_constant_handler(
    val: np.ndarray, canonicalize_types
) -> Sequence[ir.Value]:
    if np.any(np.equal(0, val.strides)) and val.size > 0:
        (zero_stride_axes,) = np.where(np.equal(0, val.strides))
        (other_axes,) = np.where(np.not_equal(0, val.strides))
        collapsed_val = val[
            tuple(
                0 if ax in zero_stride_axes else slice(None) for ax in range(val.ndim)
            )
        ]
        if canonicalize_types:
            collapsed_val = np.asarray(
                collapsed_val, dtype.canonicalize_dtype(collapsed_val.dtype)
            )
        out = hlo.BroadcastInDimOp(
            ir.RankedTensorType.get(
                val.shape, mge_dtype_to_ir_type(collapsed_val.dtype)
            ),
            _numpy_array_constant(collapsed_val, canonicalize_types=False)[0],
            dense_int_elements(other_axes),
        ).result
        return (out,)
    else:
        return _numpy_array_constant(val, canonicalize_types)


_constant_handlers[np.ndarray] = _ndarray_constant_handler
for _scalar_type in [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.bool_,
    np.longlong,
    dtype.bfloat16,
]:
    _constant_handlers[_scalar_type] = _ndarray_constant_handler


def _python_scalar_constant_handler(dtype, val, canonicalize_dtypes):
    return _numpy_array_constant(np.array(val, dtype), canonicalize_dtypes)


for pt, dt in dtype._python_scalar_dtype_to_npdtypes.items():
    _constant_handlers[pt] = partial(_python_scalar_constant_handler, dt)


def _mge_varinfo_constant_handler(val, canonicalize_dtypes):
    assert isinstance(val, VarInfo)
    assert val.bound_data is not None and val.kind == "const"
    assert isinstance(val.bound_data, np.ndarray)
    return _numpy_array_constant(
        np.asarray(val.bound_data, val.dtype), canonicalize_dtypes
    )


_constant_handlers[VarInfo] = _mge_varinfo_constant_handler


def ir_constant_tuple(val: Any, canonicalize_types: bool = True) -> Sequence[ir.Value]:
    for t in type(val).__mro__:
        handler = _constant_handlers.get(t)
        if handler:
            out = handler(val, canonicalize_types)
            assert all(isinstance(v, ir.Value) for v in out), (type(val), out)
            return out
    assert False


def ir_constant(val: Any, canonicalize_types: bool = True) -> Sequence[ir.Value]:
    values = ir_constant_tuple(val, canonicalize_types=canonicalize_types)
    assert len(values) == 1
    return values[0]


def token_type() -> Sequence[ir.Type]:
    return [hlo.TokenType.get()]


def dummy_token_type_tuple() -> Sequence[ir.Type]:
    return make_ir_type_according_meta_tuple((0,), np.bool_)


def dummy_token() -> Sequence[ir.Value]:
    return ir_constant_tuple(np.zeros(0, np.bool_))


def i32_attr(i):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)


def i64_attr(i):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)


def ui64_attr(i):
    return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), i)


def f32_attr(i):
    return ir.FloatAttr.get(ir.F32Type.get(), i)


def bool_attr(i):
    return ir.BoolAttr.get(i)


def precision_attr(lhs_prec, rhs_prec) -> ir.ArrayAttr:
    lhs_prec = str(lhs_prec)
    rhs_prec = str(rhs_prec)

    assert lhs_prec == "float32"
    assert rhs_prec == "float32"

    dtype_to_precision = {
        "float32": "DEFAULT",
    }
    precision = (dtype_to_precision[lhs_prec], dtype_to_precision[rhs_prec])
    return ir.ArrayAttr.get([hlo.PrecisionAttr.get(p) for p in precision])


def dense_int_elements(xs) -> ir.DenseIntElementsAttr:
    return ir.DenseIntElementsAttr.get(np.asarray(xs, np.int64))


def dense_bool_elements(xs: Sequence[bool]) -> ir.DenseElementsAttr:
    a = np.packbits(np.array(xs, np.bool_), bitorder="little")
    if len(xs) == 1:
        a = np.array(0 if a.item() == 0 else 0xFF, np.uint8)
    return ir.DenseElementsAttr.get(
        a, type=ir.IntegerType.get_signless(1), shape=[len(xs)]
    )


def get_irnode_shape(irnode):
    if isinstance(irnode, (list, tuple, ir.OpResultList)):
        assert len(irnode) == 1
        irnode = irnode[0]
    assert isinstance(irnode, (ir.RankedTensorType, ir.BlockArgument, ir.OpResult))
    if not isinstance(irnode, ir.RankedTensorType):
        irnode = ir.RankedTensorType(irnode.type)
    return tuple(irnode.shape)


def get_irnode_dtype(irnode):
    if isinstance(irnode, (list, tuple, ir.OpResultList)):
        assert len(irnode) == 1
        irnode = irnode[0]
    assert isinstance(
        irnode, (ir.RankedTensorType, ir.BlockArgument, ir.OpResult)
    ), type(irnode)
    if not isinstance(irnode, ir.RankedTensorType):
        irnode = ir.RankedTensorType(irnode.type)
    etype = irnode.element_type

    for k, v in _dtype_to_ir_type.items():
        if etype == v():
            return k

    assert False, f"unknown irnode {irnode}"


def module_to_string(module: ir.Module) -> str:
    output = io.StringIO()
    module.operation.print(
        file=output, enable_debug_info=True, print_generic_op_form=False
    )
    return output.getvalue()


def module_to_bytecode(module: ir.Module) -> bytes:
    output = io.BytesIO()
    module.operation.write_bytecode(file=output)
    return output.getvalue()
