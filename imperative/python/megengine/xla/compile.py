import dataclasses
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Union

import numpy as np

from .. import tensor
from ..distributed import is_distributed
from ..utils.dlpack import from_dlpack, to_dlpack
from . import ir_utils
from .lib import xla_bridge as xb
from .lib import xla_client as xc
from .lib.mlir import ir
from .sharding import (
    _get_normalized_avals_and_shardings,
    _get_op_sharding_shardings_from_executable,
    _get_pmap_sharding,
    _is_unspecified,
    _pmap_sharding_spec,
    is_op_sharding_replicated,
    pmap_lib,
    shard_args,
)
from .utils import safe_zip, unzip2

xla_extension = xc._xla
xe = xla_extension


def compile_impl(backend, computation: ir.Module, compile_options, host_callbacks):
    sym_name = computation.operation.attributes["sym_name"]
    module_name = ir.StringAttr(sym_name).value

    serialized_computation: Union[str, bytes, ir.Module]
    if getattr(backend, "needs_str_ir", True):
        serialized_computation = ir_utils.module_to_bytecode(computation)
    else:
        serialized_computation = computation

    supported_platforms = ["gpu"]
    if "--xla_cpu_use_xla_runtime=true" in os.environ.get("XLA_FLAGS", ""):
        supported_platforms.append("cpu")

    def backend_compile(backend, built_c, options, host_callbacks):
        if host_callbacks:
            return backend.compile(
                built_c, compile_options=options, host_callbacks=host_callbacks
            )
        return backend.compile(built_c, compile_options=options)

    return backend_compile(
        backend, serialized_computation, compile_options, host_callbacks
    )


class InputsHandler:
    __slots__ = ("handler", "local_devices", "in_shardings", "input_indices")

    def __init__(self, local_devices, in_shardings, input_indices):
        self.handler = shard_args
        self.local_devices = local_devices
        self.in_shardings = in_shardings
        self.input_indices = input_indices

    def from_dlpack(self, dlpack):
        return xe.dlpack_managed_tensor_to_buffer(
            dlpack, None, self.local_devices[0].client
        )

    def __call__(self, input_buffers):
        rst = []
        for idx, i in enumerate(input_buffers):
            if i._is_external_value():
                rst.append([i._external_obj()])
            else:
                capsule = to_dlpack(i)
                xla_array = self.from_dlpack(capsule)
                rst.append([xla_array])
        return rst

    def __str__(self):
        return (
            "InputsHandler(\n"
            f"local_devices={self.local_devices},\n"
            f"in_shardings={self.in_shardings},\n"
            f"input_indices={self.input_indices})"
        )


class ResultsHandler:
    __slots__ = ("handlers", "out_shardings", "out_avals", "return_device_array")

    def __init__(
        self,
        handlers=None,
        out_shardings=None,
        out_avals=None,
        return_device_array=False,
    ):
        self.return_device_array = return_device_array
        if handlers is None:

            def out_handler(bufs):
                assert isinstance(bufs, list) and len(bufs) == 1
                assert isinstance(bufs[0], xe.ArrayImpl)
                if not self.return_device_array:
                    return np.asarray(bufs[0])
                else:
                    return bufs[0]

        self.handlers = out_handler
        self.out_shardings = out_shardings
        self.out_avals = out_avals

    def __call__(self, out_bufs):
        if isinstance(self.handlers, list):
            return [h(bufs) for h, bufs in safe_zip(self.handlers, out_bufs)]
        else:
            return [self.handlers(bufs) for bufs in out_bufs]


class Executable(Protocol):
    def call(self, *args_flat):
        raise NotImplementedError

    def input_shardings(self):
        raise NotImplementedError

    def output_shardings(self):
        raise NotImplementedError

    def as_text(self) -> str:
        raise NotImplementedError

    def cost_analysis(self) -> Any:
        raise NotImplementedError

    def memory_analysis(self) -> Any:
        raise NotImplementedError

    def runtime_executable(self) -> Any:
        raise NotImplementedError

    def create_cpp_call(self, no_kwargs, in_tree, out_tree) -> Any:
        return None


class XlaExecutable(Executable):
    def xla_extension_executable(self):
        raise NotImplementedError("should be overrided")

    def call(self, *args_flat):
        raise NotImplementedError("should be overrided")

    def input_shardings(self):
        raise NotImplementedError("should be overrided")

    def output_shardings(self):
        raise NotImplementedError("should be overrided")

    def as_text(self) -> str:
        xla_ext_exe = self.xla_extension_executable()
        err_msg = (
            "text view unsupported on current XLA backend: " f"{type(xla_ext_exe)}"
        )
        if not hasattr(xla_ext_exe, "hlo_modules"):
            raise NotImplementedError(err_msg)
        try:
            return "\n\n".join([m.to_string() for m in xla_ext_exe.hlo_modules()])
        except xla_extension.XlaRuntimeError as e:
            msg, *_ = e.args
            if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
                raise NotImplementedError(err_msg) from e
            else:
                raise

    def cost_analysis(self) -> List[Dict[str, float]]:
        xla_ext_exe = self.xla_extension_executable()
        err_msg = (
            "cost analysis unsupported on current XLA backend: " f"{type(xla_ext_exe)}"
        )

        if hasattr(xla_ext_exe, "client"):
            try:
                return [
                    xla_extension.hlo_module_cost_analysis(xla_ext_exe.client, m)
                    for m in xla_ext_exe.hlo_modules()
                ]
            except xla_extension.XlaRuntimeError as e:
                msg, *_ = e.args
                if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
                    raise NotImplementedError(err_msg) from e
                else:
                    raise
        elif hasattr(xla_ext_exe, "cost_analysis"):
            try:
                return xla_ext_exe.cost_analysis()
            except xla_extension.XlaRuntimeError as e:
                msg, *_ = e.args
                if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
                    raise NotImplementedError(err_msg) from e
                else:
                    raise
        else:
            raise NotImplementedError(err_msg)

    def memory_analysis(self) -> Any:
        xla_ext_exe = self.xla_extension_executable()
        err_msg = (
            "memory analysis unsupported on current XLA backend: "
            f"{type(xla_ext_exe)}"
        )
        if not hasattr(xla_ext_exe, "get_compiled_memory_stats"):
            raise NotImplementedError(err_msg)
        try:
            return xla_ext_exe.get_compiled_memory_stats()
        except xla_extension.XlaRuntimeError as e:
            msg, *_ = e.args
            if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
                raise NotImplementedError(err_msg) from e
            else:
                raise

    def runtime_executable(self) -> Any:
        return self.xla_extension_executable()


# The logic to shard inputs, execute a replicated model, returning outputs
class ExecuteReplicated:
    __slots__ = [
        "xla_executable",
        "name",
        "backend",
        "in_handler",
        "out_handler",
        "has_unordered_effects",
        "ordered_effects",
        "keepalive",
        "has_host_callbacks",
        "_local_devices",
        "kept_var_idx",
        "__weakref__",
    ]

    def __init__(
        self,
        xla_executable,
        name,
        backend,
        in_handler: InputsHandler,
        out_handler: ResultsHandler,
        unordered_effects: Any,
        ordered_effects: Any,
        keepalive: Any,
        has_host_callbacks: bool,
        kept_var_idx: Set[int],
    ):
        self.xla_executable = xla_executable
        self.name = name
        self.backend = backend
        self.in_handler = in_handler
        self.out_handler = out_handler
        self.has_unordered_effects = bool(unordered_effects)
        self.ordered_effects = ordered_effects
        self._local_devices = self.xla_executable.local_devices()
        if ordered_effects:
            assert len(self._local_devices) == 1
        self.keepalive = keepalive
        self.has_host_callbacks = has_host_callbacks
        self.kept_var_idx = kept_var_idx

    def __call__(self, *args):
        args = [x for i, x in enumerate(args) if i in self.kept_var_idx]
        input_bufs = self.in_handler(args)
        assert not (
            self.ordered_effects
            or self.has_unordered_effects
            or self.has_host_callbacks
        )

        if True or not is_distributed():
            out_bufs = self.xla_executable.execute_sharded_on_local_devices(input_bufs)
            return self.out_handler(out_bufs)
        else:
            results = self.xla_executable.execute_sharded(input_bufs)
            outputs = results.disassemble_into_single_device_arrays()
            assert isinstance(outputs, list)
            out_bufs = []
            for oup in outputs:
                assert isinstance(oup, list) and len(oup) == 1
                out_bufs.append(oup[0].device_buffers)
            return self.out_handler(out_bufs)


@dataclasses.dataclass
class UnloadedMeshExecutable:
    xla_executable: Any
    trace_result: ir_utils.TraceResult
    device_assignment: Sequence[xc.Device]
    backend: xb.XlaBackend
    input_shardings: Sequence[Any]
    output_shardings: Sequence[Any]
    committed: bool
    are_out_shardings_from_xla: Sequence[bool]
    pmap_nreps: int
    name: str
    unordered_effects: List[Any]
    ordered_effects: List[Any]
    keepalive: Sequence[Any]
    host_callbacks: Sequence[Any]
    kept_var_idx: Set[int]
    auto_spmd_lowering: bool
    return_device_array: bool = False

    def load(self):
        def _get_input_indices(avals, shardings):
            input_indices = []
            for aval, sharding in zip(avals, shardings):
                proto = sharding._to_xla_op_sharding(len(aval.shape))
                if is_op_sharding_replicated(proto):
                    index = tuple(
                        (slice(None),) * len(aval.shape)
                        for _ in range(len(sharding.addressable_devices))
                    )
                else:
                    assert False
                input_indices.append(index)
            return input_indices

        input_indices = _get_input_indices(
            self.trace_result._var_inputs, self.input_shardings
        )
        handle_inps = InputsHandler(
            self.xla_executable.local_devices(), self.input_shardings, input_indices
        )
        handle_oups = ResultsHandler(return_device_array=self.return_device_array)

        if self.pmap_nreps > 1:
            assert False
        else:
            unsafe_call = ExecuteReplicated(
                self.xla_executable,
                self.name,
                self.backend,
                handle_inps,
                handle_oups,
                self.unordered_effects,
                self.ordered_effects,
                self.keepalive,
                bool(self.host_callbacks),
                self.kept_var_idx,
            )

        return MeshExecutable(
            self.xla_executable,
            unsafe_call,
            self.trace_result,
            self.input_shardings,
            self.output_shardings,
            self.auto_spmd_lowering,
            self.kept_var_idx,
            self.device_assignment,
        )

    @staticmethod
    def from_hlo(
        name: str,
        computation,
        mesh,
        trace_result: ir_utils.TraceResult,
        in_shardings,
        out_shardings,
        spmd_lowering: bool,
        tuple_args: bool,
        in_is_global: Sequence[bool],
        auto_spmd_lowering: bool,
        _allow_propagation_to_outputs: bool,
        _allow_compile_replicated: bool,
        unordered_effects,
        ordered_effects,
        host_callbacks,
        keepalive,
        kept_var_idx,
        backend: xb.XlaBackend,
        device_assignment: Sequence[xc.Device],
        committed: bool,
        pmap_nreps: int = 1,
        return_device_array: bool = False,
    ):
        assert mesh == None
        assert spmd_lowering == False
        assert tuple_args == False
        assert in_is_global == (True,) * len(trace_result.inputs)
        assert auto_spmd_lowering == False
        assert _allow_propagation_to_outputs == False
        assert _allow_compile_replicated == True
        assert unordered_effects == []
        assert ordered_effects == []
        assert host_callbacks == []
        assert keepalive == []
        assert committed == False
        assert pmap_nreps == 1

        dev: np.ndarray
        if auto_spmd_lowering:
            assert mesh is not None and spmd_lowering
            dev = mesh.devices
            num_replicas, num_partitions = 1, mesh.size
        else:
            dev = np.array(device_assignment)
            if pmap_nreps > 1:
                num_replicas, num_partitions = pmap_nreps, 1
            elif spmd_lowering:
                num_replicas, num_partitions = 1, dev.size
            else:
                num_replicas, num_partitions = dev.size, 1

        if pmap_nreps > 1:
            xla_device_assignment = None
        else:
            xla_device_assignment = dev.reshape((num_replicas, num_partitions))

        assert num_replicas == 1 and num_partitions == 1
        compile_options = xb.get_compile_options(
            num_replicas=num_replicas,
            num_partitions=num_partitions,
            device_assignment=xla_device_assignment,
            use_spmd_partitioning=spmd_lowering,
            use_auto_spmd_partitioning=auto_spmd_lowering,
        )
        if auto_spmd_lowering:
            assert False
        # tuple_args is only tpu related, so in mge we close it
        compile_options.parameter_is_tupled_arguments = False
        allow_propagation = [_allow_propagation_to_outputs]
        compile_options.executable_build_options.allow_spmd_sharding_propagation_to_output = (
            allow_propagation
        )
        assert hasattr(backend, "compile_replicated") == False
        if _allow_compile_replicated and hasattr(backend, "compile_replicated"):
            assert False
        else:
            xla_executable = compile_impl(
                backend, computation, compile_options, host_callbacks
            )

            if auto_spmd_lowering:
                assert False
            elif out_shardings and any(_is_unspecified(o) for o in out_shardings):
                assert mesh is None
                _, out_shardings_xla = _get_op_sharding_shardings_from_executable(  # type: ignore
                    xla_executable,
                    device_assignment,
                    len(trace_result.inputs),
                    len(trace_result.outputs),
                )
                out_shardings_tuple = [
                    (x, True) if _is_unspecified(o) else (o, False)
                    for x, o in safe_zip(out_shardings_xla, out_shardings)
                ]
                out_shardings, are_out_shardings_from_xla = unzip2(out_shardings_tuple)
            else:
                are_out_shardings_from_xla = (False,) * len(trace_result.outputs)

            input_avals, input_shardings = _get_normalized_avals_and_shardings(
                trace_result._var_inputs, in_shardings, in_is_global
            )

            return UnloadedMeshExecutable(
                xla_executable=xla_executable,
                trace_result=trace_result,
                device_assignment=device_assignment,
                backend=backend,
                input_shardings=input_shardings,
                output_shardings=out_shardings,
                committed=committed,
                are_out_shardings_from_xla=are_out_shardings_from_xla,
                pmap_nreps=pmap_nreps,
                name=name,
                unordered_effects=unordered_effects,
                ordered_effects=ordered_effects,
                keepalive=keepalive,
                host_callbacks=host_callbacks,
                kept_var_idx=kept_var_idx,
                auto_spmd_lowering=auto_spmd_lowering,
                return_device_array=return_device_array,
            )


class MeshExecutable(XlaExecutable):
    __slots__ = [
        "xla_executable",
        "unsafe_call",
        "trace_result",
        "_in_shardings",
        "_out_shardings",
        "_auto_spmd_lowering",
        "_kept_var_idx",
        "_device_assignment",
    ]

    def __init__(
        self,
        xla_executable,
        unsafe_call,
        trace_result,
        in_shardings,
        out_shardings,
        auto_spmd_lowering,
        kept_var_idx,
        device_assignment,
    ):
        self.xla_executable = xla_executable
        self.unsafe_call = unsafe_call
        self.trace_result = trace_result
        self._in_shardings = in_shardings
        self._out_shardings = out_shardings
        self._auto_spmd_lowering = auto_spmd_lowering
        self._kept_var_idx = kept_var_idx
        self._device_assignment = device_assignment

    def xla_extension_executable(self):
        return self.xla_executable

    def call(self, *args):
        return self.unsafe_call(*args)

    def input_shardings(self):
        return self._in_shardings

    def output_shardings(self):
        return self._out_shardings


class Lowering(Protocol):
    def compile(self) -> Executable:
        raise NotImplementedError

    def as_text(self, dialect: Optional[str] = None) -> str:
        raise NotImplementedError

    def compiler_ir(self, dialect: Optional[str] = None) -> Any:
        raise NotImplementedError


class XlaLowering(Lowering):
    def hlo(self) -> xc.XlaComputation:
        raise NotImplementedError("must override")

    # Return an MHLO IR of computation
    def mhlo(self) -> ir.Module:
        module_str = xla_extension.mlir.stablehlo_to_mhlo(
            ir_utils.module_to_bytecode(self.stablehlo())
        )
        with self.stablehlo().context:
            return ir.Module.parse(module_str)

    # Return a StableHLO IR of computation
    def stablehlo(self) -> ir.Module:
        raise NotImplementedError("must override")

    def compile(self) -> Executable:
        raise NotImplementedError("must override")

    def as_text(self, dialect: Optional[str] = None) -> str:
        if dialect is None:
            dialect = "stablehlo"
        if dialect == "mhlo":
            return str(self.mhlo())
        elif dialect == "stablehlo":
            return str(self.stablehlo())
        elif dialect == "hlo":
            return self.hlo().as_hlo_text()
        else:
            raise ValueError(f"unknown dialect: {dialect}")

    def compiler_ir(self, dialect: Optional[str] = None) -> Any:
        if dialect is None:
            dialect = "stablehlo"
        if dialect == "mhlo":
            return self.mhlo()
        elif dialect == "stablehlo":
            return self.stablehlo()
        elif dialect == "hlo":
            return self.hlo()
        else:
            raise ValueError(f"unknown dialect: {dialect}")


class MeshComputation(XlaLowering):
    _hlo: Optional[ir.Module]
    _executable: Optional[MeshExecutable]

    def __init__(
        self,
        name: str,
        hlo: Optional[ir.Module],
        donated_invars: Sequence[bool],
        **compile_args
    ):
        self._name = name
        self._hlo = hlo
        self._donated_invars = donated_invars
        self.compile_args = compile_args
        self._executable = None

    def _compile_unloaded(
        self,
        _allow_propagation_to_outputs: bool = False,
        _allow_compile_replicated: bool = True,
    ) -> Union[UnloadedMeshExecutable, MeshExecutable]:
        return UnloadedMeshExecutable.from_hlo(
            self._name,
            self._hlo,
            **self.compile_args,
            _allow_propagation_to_outputs=_allow_propagation_to_outputs,
            _allow_compile_replicated=_allow_compile_replicated,
        )

    def hlo(self) -> xc.XlaComputation:
        return xe.mlir.mlir_module_to_xla_computation(
            ir_utils.module_to_string(self._hlo),
            use_tuple_args=self.compile_args["tuple_args"],
        )

    def mhlo(self) -> ir.Module:
        return super().mhlo()

    def stablehlo(self) -> ir.Module:
        return self._hlo

    def compile(
        self,
        _allow_propagation_to_outputs: bool = False,
        _allow_compile_replicated: bool = True,
    ) -> MeshExecutable:
        if self._executable is None:
            executable = self._compile_unloaded(
                _allow_propagation_to_outputs, _allow_compile_replicated
            )
            if isinstance(executable, UnloadedMeshExecutable):
                executable = executable.load()
            self._executable = executable
        return self._executable


class PmapExecutable(XlaExecutable):
    __slots__ = [
        "xla_executable",
        "_unsafe_call",
        "build_unsafe_call",
        "trace_result",
        "_unloaded_executable",
    ]

    def __init__(
        self, xla_executable, build_unsafe_call, trace_result, unloaded_executable,
    ):
        self.xla_executable = xla_executable
        self._unsafe_call = None
        self.build_unsafe_call = build_unsafe_call
        self.trace_result = trace_result
        self._unloaded_executable = unloaded_executable

    @property
    def unsafe_call(self) -> Callable[..., Any]:
        if self._unsafe_call is None:
            self._unsafe_call = self.build_unsafe_call()
        return self._unsafe_call

    def xla_extension_executable(self):
        return self.xla_executable

    def call(self, *args):
        return self.unsafe_call(*args)


@dataclasses.dataclass
class UnloadedPmapExecutable:
    compiled: Any
    trace_result: ir_utils.TraceResult
    backend: xb.XlaBackend
    input_shardings: Sequence[Any]
    output_shardings: Sequence[Any]
    unordered_effects: List[Any]
    ordered_effects: List[Any]
    keepalive: Sequence[Any]
    host_callbacks: Sequence[Any]
    kept_var_idx: Set[int]
    rank: int
    return_device_array: bool = False

    @staticmethod
    def from_hlo(
        computation,
        trace_result: ir_utils.TraceResult,
        unordered_effects,
        ordered_effects,
        tuple_args,  # for tpu
        in_is_global,
        host_callbacks,
        keepalive,
        kept_var_idx,
        backend,
        devices,
        return_device_array,
        world_size,
        rank,
    ):
        assert unordered_effects == []
        assert ordered_effects == []
        assert host_callbacks == []
        assert keepalive == []
        assert tuple_args == False
        assert in_is_global == (True,) * len(trace_result.inputs)
        assert devices is None

        if devices is None:
            if world_size > xb.device_count(backend):
                assert (
                    False
                ), f"world_size={world_size} is bigger than device_count={xb.device_count(backend)}"

            devices = [
                d
                for process_index in range(xb.process_count(backend))
                for d in xb.local_devices(process_index, backend)
            ]
        else:
            assert False, "impossible"

        device_assignment: np.ndarray = np.array(devices).reshape((world_size, 1))

        use_spmd_partitioning = False
        compile_options = xb.get_compile_options(
            num_replicas=world_size,
            num_partitions=1,
            device_assignment=device_assignment,
            use_spmd_partitioning=use_spmd_partitioning,
        )
        compile_options.parameter_is_tupled_arguments = tuple_args
        compiled = compile_impl(backend, computation, compile_options, host_callbacks)

        process_index = xb.process_index(backend)
        local_device_assignment = np.array(
            [d for d in device_assignment.flat if d.process_index == process_index]
        )

        ishapes = [inp.shape for inp in trace_result._var_inputs]
        input_sharding_specs = [
            _pmap_sharding_spec(1, 1, 1, None, ishape, 0) for ishape in ishapes
        ]
        in_shardings = _get_pmap_sharding(local_device_assignment, input_sharding_specs)

        oshapes = [out.shape for out in trace_result._var_outputs]
        out_specs = [
            _pmap_sharding_spec(1, 1, 1, None, oshape, 0) for oshape in oshapes
        ]
        out_shardings = _get_pmap_sharding(local_device_assignment, out_specs)

        return UnloadedPmapExecutable(
            compiled=compiled,
            trace_result=trace_result,
            backend=backend,
            input_shardings=in_shardings,
            output_shardings=out_shardings,
            unordered_effects=unordered_effects,
            ordered_effects=ordered_effects,
            keepalive=keepalive,
            host_callbacks=host_callbacks,
            kept_var_idx=kept_var_idx,
            rank=rank,
            return_device_array=return_device_array,
        ).load()

    def build_execute_fun(self):
        input_indices = []
        ishapes = [inp.shape for inp in self.trace_result._var_inputs]
        for ishape, isharding in safe_zip(ishapes, self.input_shardings):
            spec = isharding.sharding_spec
            assert len(spec.sharding) == len(ishape) + 1
            assert spec.sharding[0] == pmap_lib.Unstacked(1)
            assert all(isinstance(s, pmap_lib.NoSharding) for s in spec.sharding[1:])
            input_indices.append(
                ((tuple(slice(None, None, None) for _ in range(len(ishape)))),)
            )
        handle_inps = InputsHandler(
            self.compiled.local_devices(), self.input_shardings, input_indices
        )
        handle_oups = ResultsHandler(return_device_array=self.return_device_array)

        execute_fun = ExecuteReplicated(
            self.compiled,
            "parallel computation",
            self.backend,
            handle_inps,
            handle_oups,
            self.unordered_effects,
            self.ordered_effects,
            self.keepalive,
            bool(self.host_callbacks),
            set(range(len(input_indices))),
        )
        return execute_fun

    def load(self) -> PmapExecutable:
        return PmapExecutable(
            self.compiled, self.build_execute_fun, self.trace_result, self,
        )


class PmapComputation(XlaLowering):
    _name: str
    _hlo: ir.Module
    _executable: Optional[PmapExecutable]

    def __init__(self, name, hlo: ir.Module, **compile_args):
        self._name = name
        self._executable = None
        self._hlo = hlo
        self.compile_args = compile_args

    def hlo(self) -> xc.XlaComputation:
        return xe.mlir.mlir_module_to_xla_computation(
            ir_utils.module_to_string(self._hlo),
            use_tuple_args=self.compile_args["tuple_args"],
        )

    def mhlo(self) -> ir.Module:
        return super().mhlo()

    def stablehlo(self) -> ir.Module:
        return self._hlo

    def compile(self) -> PmapExecutable:
        if self._executable is None:
            self._executable = UnloadedPmapExecutable.from_hlo(
                self._hlo, **self.compile_args
            )
        return self._executable
