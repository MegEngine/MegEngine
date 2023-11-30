from collections import OrderedDict
from typing import Sequence

from ..core._imperative_rt.core2 import (
    add_backward_callback as insert_callback_as_grad_tape,
)
from ..core._imperative_rt.core2 import get_grad_slot, get_handle_id
from ..logger import get_logger
from ..tensor import Tensor
from .tracing import trace
from .xla_backend import xla_trace

logger = get_logger(__name__)


def _process_fwd_bwd_trace_result(fwd, bwd, inp_grad_map, out_grad_map):
    # partial_trace will record op sequences for forward/backward respectively, and get two TraceResult objects after tracing.
    # But the inputs/outputs of backward graph are unknown. This function will determine the inputs and outputs of the backward graph
    # var.handle_id is id of value ref. It's used to find the tensors  used in both forward and backward calculation.
    # inp_grad_map, key: handle id of forward inputs, value: handle id of grads of forward inputs.
    # out_grad_map, key: handle id of foward outputs, value: handle id of grads of forward outputs.
    fwd_features = set([t.handle_id for t in fwd._trace.vars])
    bwd_features = set([t.handle_id for t in bwd._trace.vars])
    keep_vars = fwd_features.intersection(
        bwd_features
    )  # some intermediate vars produced by forward, and will be used in backward.
    current = fwd.output_num
    saved_feature_map = OrderedDict()
    saved_featrues = []
    # mark keep_vars as forward outputs
    for var in fwd._trace.vars:
        if (
            var.handle_id in keep_vars
            and var.data_required
            and len(var.out_mark) == 0
            and var.kind not in ["const", "external"]
        ):
            keep_vars.remove(var.handle_id)
            fwd._trace.mark_output(current, var.id)
            saved_feature_map[var.handle_id] = current
            saved_featrues.append(current)
            current += 1
    fwd.keeped_activation = saved_featrues

    bwd_inp_idx = 0
    bwd_out_idx = 0
    bwd_dys = []
    bwd_inps = [-1] * len(saved_feature_map)
    saved_feature_handle_id = list(saved_feature_map.keys())
    dy_ids = list(out_grad_map.values())  # handle_id of grad of forward output
    inp_grad_ids = list(inp_grad_map.values())  # handle_id of grad of forward input
    bwd_dys = [-1] * len(dy_ids)
    bwd_outputs = [-1] * len(inp_grad_ids)
    # dy_ids + saved_feature_map are backward inputs
    # inp_grad_ids are backward outputs
    # mark inputs/outputs for backward
    for var in bwd._trace.vars:
        if var.handle_id in dy_ids and var.kind == "external":
            bwd._trace.mark_input(bwd_inp_idx, var.id)
            idx = dy_ids.index(var.handle_id)
            bwd_dys[idx] = bwd_inp_idx
            bwd_inp_idx += 1
        elif var.handle_id in saved_feature_map and var.kind == "external":
            bwd._trace.mark_input(bwd_inp_idx, var.id)
            bwd_inps[saved_feature_handle_id.index(var.handle_id)] = bwd_inp_idx
            bwd_inp_idx += 1
        if var.handle_id in inp_grad_ids and var.data_required:
            bwd_outputs[inp_grad_ids.index(var.handle_id)] = bwd_out_idx
            bwd._trace.mark_output(bwd_out_idx, var.id)
            bwd_out_idx += 1
    # assert -1 not in bwd_dys
    assert -1 not in bwd_inps
    fwd._trace._remove_unused_data_required()
    # assert -1 not in bwd_outputs
    bwd.setup_io_without_trace(bwd_dys + bwd_inps, bwd_outputs)
    bwd.setup_without_host()

    bwd._trace._remove_unused_data_required()

    def check_external(trace_obj):
        for var in trace_obj.vars:
            if var.kind == "external" and not var.inp_mark:
                raise RuntimeError(
                    "have unknown input in trace result, maybe you can set `capture_as_const=True` when trace"
                )

    check_external(fwd)
    check_external(bwd)


JIT_BACKEND = {"default": trace, "xla": xla_trace}


def partial_trace(
    func=None,
    *,
    backend="default",
    without_host=True,
    trace_gmcallback=True,
    **trace_options
):

    assert backend in JIT_BACKEND
    assert without_host, "partial_trace only support without_host mode currently!"

    def wrapper(func):
        trace_obj = JIT_BACKEND[backend](
            func, without_host=without_host, **trace_options
        )
        trace_options["capture_as_const"] = False
        backward_trace_obj = JIT_BACKEND[backend](
            None, without_host=without_host, **trace_options
        )
        backward_trace_obj.check_external = (
            False  # check if there are unknown external vars after tracing.
        )
        trace_obj.overall = False  # if trace overall train step
        backward_trace_obj.overall = False
        trace_obj._trace.remove_unused_data_required = False
        backward_trace_obj._trace.remove_unused_data_required = False
        inp_grad_maps = OrderedDict()  # x, dx map
        out_grad_maps = OrderedDict()  # y, dy map
        traced = False  # if wrapped function has been traced
        custom_autodiff = None
        outdef = None  # treedef of forward return value
        check_shape = backend == "xla"
        shape_hash = None
        unexpected_shape_hash = set()
        from ..core.autodiff.grad import Function

        class CustomAutodiff(Function):
            def __init__(self, fwd, bwd):
                self.fwd = fwd
                self.bwd = bwd
                del fwd.outdef
                self.keeped_features = []

            def forward(self, *args):
                rst = self.fwd(*args)
                keeped_features = rst[-1]
                if not isinstance(keeped_features, Sequence):
                    keeped_features = tuple([keeped_features])
                else:
                    keeped_features = tuple(keeped_features)
                self.keeped_features = keeped_features
                return rst[0]

            def get_keeped_features(self):
                rst = self.keeped_features
                del self.keeped_features
                return rst

            def backward(self, *output_grads):
                output_grads = tuple([i for i in output_grads if i is not None])
                return self.bwd(*(output_grads + self.get_keeped_features()))

        class CustomFwd:
            def __init__(self, fwd, bwd):
                self.fwd = fwd
                self.bwd = bwd

            def __call__(self, *args):
                rst = self.fwd(*args)
                if self.fwd.keeped_activation:
                    keeped_features = rst[-1]
                    if not isinstance(keeped_features, Sequence):
                        keeped_features = tuple([keeped_features])
                    else:
                        keeped_features = tuple(keeped_features)
                    self.keeped_features = keeped_features
                    return rst[0]
                else:
                    return rst

        def get_shape_hash(*tensors):
            def map_scalar_to_tuple(ishape):
                return (1,) if ishape == tuple() else ishape

            return hash(tuple([map_scalar_to_tuple(t._tuple_shape) for t in tensors]))

        def wrapped_func(*args, **kwargs):
            from ..traced_module.pytree import tree_flatten
            from ..core.autodiff.grad import Grad
            from ..autodiff.grad_manager import get_backwarding_grad_manager

            nonlocal traced
            nonlocal custom_autodiff
            nonlocal outdef
            nonlocal shape_hash
            nonlocal unexpected_shape_hash
            nonlocal inp_grad_maps, out_grad_maps
            trace_obj.convert_optimizer_state_to_tensor(*args, **kwargs)
            if not traced:
                traced = True
                fargs = trace_obj.flatten_inputs(*args, **kwargs)
                shape_hash = get_shape_hash(*fargs) if check_shape else None
                # we want to construct map like {inp_hid: inp_grad_hid, ...}. but the
                # backward() is not called now and the grad is unknown. so we record the
                # grad_slot here and then use grad_slot to get the grad hid when exiting
                # trace
                # **The grad grad_slot returned is the orginal grad, that means it is
                # not been processed by gradmanager attach callback, eg., the grad
                # before allreduce when distributed training**
                inp_grad_maps = {get_handle_id(t): get_grad_slot(t) for t in fargs}
                del fargs

                def exit_trace():
                    backward_trace_obj._trace.exit()
                    backward_trace_obj.unset_env()
                    for k, v in inp_grad_maps.items():
                        inp_grad_maps[k] = (
                            get_handle_id(v.grad) if v is not None else -1
                        )

                    # if we want to trace the gradmanager attach callback, we will replace
                    # the original grad by the grad processed the gradmanger attach callback
                    if trace_gmcallback:

                        def cb_wrapper(cb):
                            def wrapper(param, grad):
                                return (
                                    grad
                                    if grad._is_external_value()
                                    else cb(param, grad)
                                )

                            return wrapper

                        current_gm = get_backwarding_grad_manager()
                        for _attached_tensor_id, grad in current_gm._gradients.items():
                            spec = current_gm._attach_specs.get(_attached_tensor_id)
                            _attached_tensor = spec and spec.tensor()
                            if _attached_tensor is None:
                                continue
                            inp_hid = get_handle_id(_attached_tensor)
                            if inp_hid in inp_grad_maps:
                                spec.callbacks = list(map(cb_wrapper, spec.callbacks))
                                inp_grad_maps[inp_hid] = get_handle_id(grad)

                insert_callback_as_grad_tape(exit_trace)

                ret = trace_obj(*args)
                rlist, outdef = tree_flatten(ret)
                out_grad_maps = {get_handle_id(t): get_grad_slot(t) for t in rlist}

                def enter_trace():
                    for k, v in out_grad_maps.items():
                        out_grad_maps[k] = (
                            get_handle_id(v.grad) if v is not None else -1
                        )
                    backward_trace_obj.setup_env()
                    backward_trace_obj._trace.enter()

                insert_callback_as_grad_tape(enter_trace)
                return ret
            else:
                if custom_autodiff is None:
                    _process_fwd_bwd_trace_result(
                        trace_obj, backward_trace_obj, inp_grad_maps, out_grad_maps
                    )
                    if len(backward_trace_obj._trace.ops) > 0:
                        assert (
                            len(Grad.key2grad) == 1
                        ), "derivatives of higher order not supported in partial trace"
                        assert (
                            Grad.grouping is False
                        ), "group gradmanager is not supported"
                        custom_autodiff = CustomAutodiff(trace_obj, backward_trace_obj)
                    else:
                        # we always construct two xla_trace obj in partial_trace,
                        # one for forward and another for backward. so the value of the
                        # _expect_xlacompile_cnt is updated by "+= 2". but in some times
                        # there is no backward, so we should minus one at this time for
                        # _expect_xlacompile_cnt
                        from .xla_backend import _expect_xlacompile_cnt_minus_one

                        if backend == "xla":
                            _expect_xlacompile_cnt_minus_one()
                        custom_autodiff = CustomFwd(trace_obj, backward_trace_obj)
                fargs = trace_obj.flatten_inputs(*args, **kwargs)
                if check_shape and get_shape_hash(*fargs) != shape_hash:
                    if get_shape_hash(*fargs) not in unexpected_shape_hash:
                        unexpected_shape_hash.add(get_shape_hash(*fargs))
                        logger.warning("XLA shape mismatch, fallback to python")
                    return trace_obj.__wrapped__(*args, **kwargs)
                del args
                del kwargs
                if outdef is None:
                    return custom_autodiff(*fargs)
                else:
                    rst = custom_autodiff(*fargs)
                    rst = [rst,] if not isinstance(rst, Sequence) else rst
                    return outdef.unflatten(rst)

        return wrapped_func

    if func is None:
        return wrapper
    else:
        return wrapper(func)
