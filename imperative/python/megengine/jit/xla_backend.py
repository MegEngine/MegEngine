from collections import OrderedDict, defaultdict

from ..core._imperative_rt import CompNode
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._trace_option import set_use_xla_backend
from ..device import get_default_device
from ..utils.dlpack import from_dlpack, to_dlpack
from .tracing import trace

try:
    from ..xla.lib import xla_client as xc
except ImportError:
    pass


class xla_trace(trace):
    r"""Wraps a callable, and provides accelerated evaluation compiled by xla.
        Currently it is an experimental feature.
        Refer to :class:`~.jit.tracing.trace` for more information.


    Examples:

        .. code-block:: python

            import numpy as np
            from basecls.models.resnet import resnet18
            from megengine.autodiff.grad_manager import GradManager
            from megengine.jit import xla_trace
            from megengine.optimizer import Adam

            model = resnet18()
            gm = GradManager()
            opt = Adam(model.parameters(), lr=1e-4)
            gm.attach(model.parameters())

            # Only tensors in wrapped func args/kwargs will be treated as graph inputs,
            # and other tensors will be captured as const value.
            # Module, optimizer, and train data/label should be arguments of the wrapped function.
            @xla_trace(capture_as_const=True)
            def train_step(model, opt, data, label):
                with gm:
                    pred = model(data)
                    loss = F.loss.cross_entropy(pred, label)
                    gm.backward(loss)
                opt.step().clear_grad()
                return loss

    """

    third_party_backend = True

    def __init__(self, function, *, without_host=True, symbolic_shape=False, **kwargs):
        assert without_host, "xla trace only support without host mode"
        assert not symbolic_shape, "xla doesn't support dynamic shape currently"
        super().__init__(
            function, without_host=without_host, symbolic_shape=symbolic_shape, **kwargs
        )

    def setup_env(self):
        self.orig_use_xla = set_use_xla_backend(True)

    def unset_env(self):
        set_use_xla_backend(self.orig_use_xla)

    def compile(self):
        from ..xla import build_xla
        from ..traced_module.pytree import SUPPORTED_LEAF_TYPE, register_supported_type
        from ..utils.module_utils import get_expand_structure
        from ..xla.device import get_xla_backend_and_device
        from ..tensor import Tensor

        assert self.traced
        if self.overall:
            for attr, _ in self.attr_to_key.items():
                param = get_expand_structure(attr[0], attr[1])
                param._reset(param.to("cpux"))

            for tensor, _ in self.opt_param_dict.items():
                tensor._reset(tensor.to("cpux"))
        self.xla_exec, self.inp_ids, self.out_ids = build_xla(
            self, return_with_io=True, return_device_array=True
        )
        id2inpidx = defaultdict(list)
        id2outidx = defaultdict(list)
        for idx, id in enumerate(self.inp_ids):
            id2inpidx[id].append(idx)
        for idx, id in enumerate(self.out_ids):
            id2outidx[id].append(idx)
        self.inpkey2idx = {}
        self.outkey2idx = {}
        if self.input_num == len(set(self.inp_ids)) - 1:
            self.has_randomstate = True
            self.random_seed = Tensor([[1, 2], [3, 4]], dtype="int32")
        else:
            assert self.input_num == len(set(self.inp_ids))
            self.has_randomstate = False
        inpmark2id = dict()
        outmark2id = dict()
        for var in self.vars:
            if var.kind == "external":
                for mark in var.inp_mark:
                    inpmark2id[mark] = var.id
            elif var.data_required and var.out_mark:
                for mark in var.out_mark:
                    outmark2id[mark] = var.id
        for k, v in inpmark2id.items():
            for idx in id2inpidx[v]:
                self.inpkey2idx[k] = idx

        for k, v in outmark2id.items():
            for idx in id2outidx[v]:
                self.outkey2idx[k] = idx

    def prepare_xla_inputs(self, tensors):
        from ..utils.module_utils import get_expand_structure

        inp_count = 0
        inp_list = [0] * self.input_num
        for idx, t in enumerate(tensors):
            inp = self.inpkey2idx[self.arg_list[idx]]
            inp_list[inp] = t
            inp_count += 1
        if self.overall:
            for attr, key in self.attr_to_key.items():
                param = get_expand_structure(attr[0], attr[1])
                inp = self.inpkey2idx[key]
                inp_list[inp] = param
                inp_count += 1
            for tensor, k in self.opt_param_dict.items():
                inp = self.inpkey2idx[k]
                inp_list[inp] = tensor
                inp_count += 1
        assert inp_count == self.input_num
        if self.has_randomstate:
            inp_list.append(self.random_seed)
        return inp_list

    def to_dlpack(self, x, take_ownership: bool = True):
        return xc._xla.buffer_to_dlpack_managed_tensor(x, take_ownership=take_ownership)

    def execute(self, *args, **kwargs):
        from ..traced_module.pytree import tree_flatten
        from ..tensor import Tensor
        from ..utils.module_utils import get_expand_structure

        inputs, _ = tree_flatten((args, kwargs))
        arrays = []
        cn = CompNode(get_default_device())
        stream = dict(self.xla_exec.backend.get_compute_compnode())
        device_kind, device_id, stream_id = cn.physical_locator

        xla_stream = stream[device_id]
        xla_comp_cn = "gpu{}:{}".format(device_id, xla_stream)
        for t in inputs:
            if isinstance(t, RawTensor):
                assert cn == t.device
                arrays.append(t.to(xla_comp_cn, _borrow=True))

        arrays = self.prepare_xla_inputs(arrays)
        outputs = self.xla_exec(*arrays)
        return_vals = []
        for i in self.out_list:
            if i == -1:
                if not hasattr(self, "outdef"):
                    return_vals.append(None)
            else:
                return_vals.append(outputs[self.outkey2idx[i]])
        keeped_features = []
        for i in self.keeped_activation:
            capsule = self.to_dlpack(outputs[self.outkey2idx[i]])
            t = from_dlpack(capsule, xla_stream).to(cn, _borrow=True)
            keeped_features.append(t)
        out_tensors = []
        for array in return_vals:
            if array is not None:
                capsule = self.to_dlpack(array)
                t = from_dlpack(capsule, xla_stream)
                out_tensors.append(t.to(cn, _borrow=True))
            else:
                out_tensors.append(array)
        if self.overall:
            for attr, key in self.update_param_dict.items():
                param = get_expand_structure(attr[0], attr[1])
                xla_array = outputs[self.outkey2idx[key]]
                capsule = self.to_dlpack(xla_array)
                param._reset(from_dlpack(capsule).to(cn, _borrow=True))

            for state, key in self.update_opt_param_dict.items():
                xla_array = outputs[self.outkey2idx[key]]
                capsule = self.to_dlpack(xla_array)
                state._reset(from_dlpack(capsule).to(cn, _borrow=True))
        rst = (
            self.outdef.unflatten(out_tensors)
            if hasattr(self, "outdef")
            else out_tensors
        )
        if keeped_features:
            return rst, keeped_features
        else:
            return rst
