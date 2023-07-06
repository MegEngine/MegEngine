import dataclasses
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core._imperative_rt.core2 import OpInfo, VarInfo
from . import utils
from .device import xb
from .ir_utils import TraceResult, ir_constant_tuple, mge_varinfo_to_ir_type_tuple
from .lib import xla_client as xc
from .lib.mlir import dialects, ir
from .lib.mlir.dialects import func as func_dialect
from .rules import get_rule
from .rules.hlotensor import HLOTensor
from .rules.utils import _shape_equal
from .sharding import sharded_val


def make_ir_context() -> ir.Context:
    context = ir.Context()
    dialects.mhlo.register_mhlo_dialect(context)
    dialects.chlo.register_dialect(context)
    dialects.stablehlo.register_dialect(context)
    return context


@dataclasses.dataclass
class ModuleContext:
    context: ir.Context
    module: ir.Module
    ip: ir.InsertionPoint
    symbol_table: ir.SymbolTable
    backend_or_name: Optional[Union[str, xb.XlaBackend]]
    platform: str
    keepalives: List[Any]
    channel_iterator: Iterator[int]
    host_callbacks: List[Any]

    # Stores the value of varinfo that can be inferred in lowering process
    inferred_values: Dict[VarInfo, np.ndarray]

    def __init__(
        self,
        backend_or_name: Optional[Union[str, xb.XlaBackend]],
        platform: str,
        keepalives: List[Any] = [],
        host_callbacks: List[Any] = [],
        context: Optional[ir.Context] = None,
        module: Optional[ir.Module] = None,
        ip: Optional[ir.InsertionPoint] = None,
        symbol_table: Optional[ir.SymbolTable] = None,
    ):
        assert platform is not None
        self.context = context or make_ir_context()
        self.module = module or ir.Module.create(loc=ir.Location.unknown(self.context))
        self.ip = ip or ir.InsertionPoint(self.module.body)
        self.symbol_table = symbol_table or ir.SymbolTable(self.module.operation)
        self.backend_or_name = backend_or_name
        self.platform = platform
        self.keepalives = keepalives
        self.host_callbacks = host_callbacks
        self.inferred_values = {}

    @property
    def backend(self) -> xb.XlaBackend:
        if self.backend_or_name is None or isinstance(self.backend_or_name, str):
            return xb.get_backend(self.backend_or_name)
        return self.backend_or_name

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def get_value(self, varinfo):
        assert varinfo in self.inferred_values
        return self.inferred_values[varinfo]

    def set_value(self, varinfo, value):
        self.inferred_values[varinfo] = value


@dataclasses.dataclass
class LoweringRuleContext:
    module_context: ModuleContext
    op: OpInfo
    vars_in: Sequence[VarInfo]
    vars_out: Sequence[VarInfo]
    param: Dict = None

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def _unwrap_singleton_ir_values(x):
    return x[0] if len(x) == 1 else x


def _wrap_singleton_ir_values(
    x: Union[ir.Value, Sequence[ir.Value]]
) -> Sequence[ir.Value]:
    return (x,) if isinstance(x, ir.Value) else tuple(x)


def lowering_ops(
    ctx: ModuleContext, trace_result: TraceResult, *args: Sequence[ir.Value],
):
    # var_id -> ir.Value
    env: Dict[int, Tuple[ir.Value, ...]] = {}
    consts = list(map(ir_constant_tuple, trace_result._var_consts))

    # read ir.Values from env according to var_ids
    def read(var_ids):
        assert isinstance(var_ids, (list, tuple))
        ret = []
        for vid in var_ids:
            assert isinstance(vid, int)
            ret.append(env[vid])
        return ret

    # update env with var_ids and ir.Values
    def write(var_ids, hlo_nodes):
        assert isinstance(var_ids, (list, tuple))
        assert isinstance(hlo_nodes, (map, list, tuple))
        hlo_nodes = list(hlo_nodes)
        assert len(var_ids) == len(hlo_nodes), (len(var_ids), len(hlo_nodes))
        for vid, node in zip(var_ids, hlo_nodes):
            assert vid not in env
            env[vid] = node

    assert len(args) == len(trace_result.inputs)
    assert len(consts) == len(trace_result.consts)
    assert all(isinstance(v, ir.Value) for vs in consts for v in vs)

    # initialize env with inputs and consts
    write(trace_result.inputs, args)
    write(trace_result.consts, consts)

    for eqn in trace_result.eqns:
        rule_ctx = LoweringRuleContext(
            module_context=ctx,
            op=eqn.op,
            vars_in=[trace_result.vars[inp] for inp in eqn.inputs],
            vars_out=[trace_result.vars[oup] for oup in eqn.outputs],
            param=eqn.param,
        )
        rule = get_rule(eqn.op, use_fake_rule_for_debug=False)

        in_nodes = read(eqn.inputs)
        hinps = [
            HLOTensor(irval, var.shape, var.dtype)
            for var, irval in zip(
                rule_ctx.vars_in, map(_unwrap_singleton_ir_values, in_nodes)
            )
        ]
        houps = rule(rule_ctx, *hinps)
        if isinstance(houps, HLOTensor):
            houps = [houps]

        out_nodes = []
        for out_id, hlo_out in zip(eqn.outputs, houps):
            var_out = trace_result.vars[out_id]
            assert _shape_equal(
                var_out.shape, hlo_out.shape
            ), f"{eqn.op}: {var_out.shape} != {hlo_out.shape}"
            out_nodes.append(hlo_out.tensor)
        out_nodes = tuple(map(_wrap_singleton_ir_values, out_nodes))
        write(eqn.outputs, out_nodes)
    return read(trace_result.outputs)


def make_xla_graph(
    ctx: ModuleContext,
    name: str,
    trace_result: TraceResult,
    public: bool = True,
    in_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    out_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    input_output_aliases: Optional[Sequence[Optional[int]]] = None,
) -> func_dialect.FuncOp:
    assert public is True, "do not process the visibitity of function"
    assert (
        in_shardings is None and out_shardings is None
    ), "sharding when lowering is not supported yet"
    assert (
        input_output_aliases is None or input_output_aliases == []
    ), "donated inputs are not supported yet"

    input_types = [
        mge_varinfo_to_ir_type_tuple(trace_result.vars[idx])
        for idx in trace_result.inputs
    ]
    output_types = [
        mge_varinfo_to_ir_type_tuple(trace_result.vars[idx])
        for idx in trace_result.outputs
    ]

    flat_input_types = utils.flatten_list(input_types)
    flat_output_types = utils.flatten_list(output_types)
    assert len(flat_input_types) == len(trace_result.inputs)
    assert len(flat_output_types) == len(trace_result.outputs)

    ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
    func_op = func_dialect.FuncOp(name, ftype, ip=ctx.ip)
    func_op.attributes["sym_visibility"] = ir.StringAttr.get(
        "public" if public else "private"
    )
    ctx.symbol_table.insert(func_op)

    entry_block = func_op.add_entry_block()
    with ir.InsertionPoint(entry_block):
        flat_args = entry_block.arguments
        unflattened_args = utils.unflatten_list(flat_args, map(len, input_types))
        outs = lowering_ops(ctx, trace_result, *unflattened_args)
        flat_oups = utils.flatten_list(outs)
        func_dialect.ReturnOp(flat_oups)
    return func_op


def lower(
    trace_result: TraceResult,
    backend,
    platform,
    in_shardings=None,
    out_shardings=None,
    donated_invars=None,
):
    assert donated_invars is None, "donated inputs are not supported yet"
    assert trace_result.effects == [], "effect of trace is not supported"

    if in_shardings is not None:
        trace_result.inputs = [
            sharded_val(inp, in_sharding)
            for inp, in_sharding in zip(trace_result.inputs, in_shardings)
        ]
    if out_shardings is not None:
        trace_result.outputs = [
            sharded_val(outp, out_sharding)
            for outp, out_sharding in zip(trace_result.outputs, out_shardings)
        ]

    ctx = ModuleContext(backend, platform)

    with ctx.context, ir.Location.unknown(ctx.context):
        module_name = trace_result.func_name
        ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(module_name)
        assert trace_result.effects == [], "effect of trace is not supported"
        make_xla_graph(
            ctx,
            "main",
            trace_result,
            public=True,
            in_shardings=None,
            out_shardings=None,
            input_output_aliases=[],
        )
    return ctx.module, ctx.keepalives, ctx.host_callbacks
