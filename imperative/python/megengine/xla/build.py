import os

from ..distributed import get_rank, get_world_size, is_distributed
from .compile import MeshComputation, PmapComputation
from .device import get_xla_backend_and_device
from .distribute import initialize
from .ir_utils import DropoutMaskCanonicalizer, RngKeyAdder, TraceResult
from .lib import xla_client as xc
from .lower import lower
from .sharding import OpShardingSharding, _is_unspecified, make_unspec_sharding

xla_extention = xc._xla
xe = xla_extention

Backend = xe.Client


def build_xla(
    mge_traced,
    func_name=None,
    device=None,
    keep_unused=True,
    donate_invars=None,
    verbose=int(os.environ.get("MGE_VERBOSE_XLA_IR", "0")),
    return_with_io=False,
    return_device_array=False,
    ip: str = None,
    port: int = None,
):
    assert device == None, "cannot specify device now"
    assert keep_unused == True, "keep_unused error"
    assert donate_invars == None, "donate_invars error"

    # normalize megengine trace result for lowering
    tr = TraceResult(mge_traced, func_name)
    tr = RngKeyAdder()(tr)
    tr = DropoutMaskCanonicalizer()(tr)

    if verbose and get_rank() == 0:
        print("================ Mge Trace Result ================")
        print(tr)

    in_is_global = (True,) * len(tr.inputs)
    kept_var_idx = set(range(len(tr.inputs))) if keep_unused else set()

    # init for xla distributed and setup device
    if is_distributed():
        initialize(ip, port, get_world_size(), get_rank(), [get_rank()])
    backend, device_assignment, platform = get_xla_backend_and_device(device)

    module, keepalive, host_callbacks = lower(
        tr, backend, platform, None, None, donate_invars,
    )

    if not is_distributed():
        # setup sharding information
        in_shardings = make_unspec_sharding(tr.inputs)
        out_shardings = make_unspec_sharding(tr.outputs)

        in_shardings = tuple(
            OpShardingSharding.get_replicated(device_assignment)
            if _is_unspecified(i)
            else i
            for i in in_shardings
        )

        computation = MeshComputation(
            tr.func_name,
            module,
            donated_invars=donate_invars,
            trace_result=tr,
            mesh=None,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            spmd_lowering=False,
            tuple_args=False,  # for tpu
            in_is_global=in_is_global,
            auto_spmd_lowering=False,
            unordered_effects=[],
            ordered_effects=[],
            host_callbacks=host_callbacks,
            keepalive=keepalive,
            kept_var_idx=kept_var_idx,
            backend=backend,
            device_assignment=device_assignment,
            committed=False,  # unknown
            pmap_nreps=1,
            return_device_array=return_device_array,
        )
    else:
        computation = PmapComputation(
            tr.func_name,
            module,
            trace_result=tr,
            unordered_effects=[],
            ordered_effects=[],
            tuple_args=False,  # for tpu
            in_is_global=in_is_global,
            host_callbacks=host_callbacks,
            keepalive=keepalive,
            kept_var_idx=kept_var_idx,
            backend=backend,
            devices=None,
            return_device_array=return_device_array,
            world_size=get_world_size(),
            rank=get_rank(),
        )

    if verbose and get_rank() == 0:
        print("================ XLA HLO IR ================")
        print(computation.as_text())
    compiled = computation.compile()
    if verbose and get_rank() == 0:
        print("================ XLA Execute Plan ================")
        print(compiled.as_text())

    ret = compiled.unsafe_call
    if return_with_io:
        return ret, tr.inputs, tr.outputs
    return ret
