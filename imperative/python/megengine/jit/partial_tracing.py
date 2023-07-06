from collections import OrderedDict
from typing import Sequence

from ..core._imperative_rt.core2 import add_backward_callback as _add_backward_callback
from ..core._imperative_rt.core2 import get_grad_slot, get_handle_id
from ..tensor import Tensor
from .tracing import trace
from .xla_backend import xla_trace


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
    current = max(fwd.out_list) + 1
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
    for var in fwd._trace.vars:
        if not var.out_mark:
            var.data_required = False
    # assert -1 not in bwd_outputs
    bwd.setup_io_without_trace(bwd_dys + bwd_inps, bwd_outputs)
    bwd.setup_without_host()

    def check_external(trace_obj):
        for var in trace_obj.vars:
            if var.kind == "external" and not var.inp_mark:
                raise RuntimeError(
                    "have unknown input in trace result, maybe you can set `capture_as_const=True` when trace"
                )

    check_external(fwd)
    check_external(bwd)


JIT_BACKEND = {"default": trace, "xla": xla_trace}


def partial_trace(func=None, *, backend="default", without_host=True, **trace_options):
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
        compiled = False  # if wrapped function has been compiled
        custom_autodiff = None
        outdef = None  # treedef of forward return value
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

        def wrapped_func(*args, **kwargs):
            from ..traced_module.pytree import tree_flatten
            from ..module import Module

            nonlocal traced
            nonlocal compiled
            nonlocal custom_autodiff
            nonlocal outdef

            if not traced:
                traced = True
                fargs = trace_obj.flatten_inputs(*args, **kwargs)
                for t in fargs:
                    inp_grad_maps[t] = get_grad_slot(t)
                del fargs

                def exit_trace():
                    backward_trace_obj._trace.exit()
                    new_dict = {}
                    for k, v in inp_grad_maps.items():
                        if v is not None:
                            new_dict[get_handle_id(k)] = get_handle_id(v.grad)
                        else:
                            new_dict[get_handle_id(k)] = -1
                    inp_grad_maps.clear()
                    inp_grad_maps.update(new_dict)

                _add_backward_callback(exit_trace)
                ret = trace_obj(*args)
                rlist, outdef = tree_flatten(ret)
                for t in rlist:
                    out_grad_maps[t] = get_grad_slot(t)

                def enter_trace():
                    new_dict = {}
                    for k, v in out_grad_maps.items():
                        if v is not None:
                            new_dict[get_handle_id(k)] = get_handle_id(v.grad)
                    out_grad_maps.clear()
                    out_grad_maps.update(new_dict)
                    backward_trace_obj._trace.enter()

                _add_backward_callback(enter_trace)
                return ret
            elif not compiled:
                if custom_autodiff is None:
                    _process_fwd_bwd_trace_result(
                        trace_obj, backward_trace_obj, inp_grad_maps, out_grad_maps
                    )
                    if len(backward_trace_obj._trace.ops) > 0:
                        custom_autodiff = CustomAutodiff(trace_obj, backward_trace_obj)
                    else:
                        custom_autodiff = CustomFwd(trace_obj, backward_trace_obj)
                fargs = trace_obj.flatten_inputs(*args, **kwargs)
                del args
                del kwargs
                if outdef is None:
                    return custom_autodiff(*fargs)
                else:
                    return outdef.unflatten(custom_autodiff(*fargs))

        return wrapped_func

    if func is None:
        return wrapper
    else:
        return wrapper(func)
