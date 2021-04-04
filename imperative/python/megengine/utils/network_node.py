# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import abc
import json
import sys
from typing import Callable, Sequence

import numpy as np

from ..core import _imperative_rt as rt
from ..core._imperative_rt.core2 import SymbolVar
from ..core._wrap import Device
from ..core.ops import builtin
from ..core.tensor.array_method import ArrayMethodMixin
from ..core.tensor.indexing import getitem as _getitem
from ..core.tensor.indexing import setitem as _setitem
from ..core.tensor.megbrain_graph import InputNode, OutputNode
from ..tensor import Tensor
from .comp_graph_tools import replace_vars
from .module_stats import (
    preprocess_receptive_field,
    register_flops,
    register_receptive_field,
)


class NetworkNode:
    pass


class VarNodeMeta(type(SymbolVar), type(ArrayMethodMixin)):
    pass


class VarNode(NetworkNode, SymbolVar, ArrayMethodMixin, metaclass=VarNodeMeta):
    def __init__(self, var=None, *, owner_opr=None, name=None):
        SymbolVar.__init__(self, var)
        self.owner = owner_opr
        self.name = name
        self.id = id(self)

    @classmethod
    def load(cls, sym_var, owner_opr):
        obj = cls()
        obj.var = sym_var  # mgb varnode
        obj.name = sym_var.name
        obj.owner = owner_opr
        return obj

    @property
    def shape(self):
        rst = None
        if self.var:
            try:
                rst = self.var.shape
            except:
                rst = None
        return rst

    @property
    def dtype(self):
        return self.var.dtype if self.var else None

    def __bool__(self):
        return False

    __index__ = None
    __int__ = None
    __float__ = None
    __complex__ = None

    def __hash__(self):
        return id(self)

    @property
    def _tuple_shape(self):
        return self.var.shape

    def numpy(self):
        o = OutputNode(self.var)
        self.graph.compile(o.outputs).execute()
        return o.get_value().numpy()

    def __getitem__(self, index):
        return _getitem(self, index)

    def __setitem__(self, index, value):
        if index is not Ellipsis:
            value = _setitem(self, index, value)
        if self.owner is not None:
            idx = self.owner.outputs.index(self)
            self.owner.outputs[idx] = VarNode(
                self.var, owner_opr=self.owner, name=self.var.name
            )
        self.var = value.var
        self.owner = None

    def set_owner_opr(self, owner_opr):
        self.owner = owner_opr


class OpNode(NetworkNode):

    opdef = None
    type = None

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.params = {}
        self._opr = None  # mgb opnode
        self.id = id(self)

    @classmethod
    def load(cls, opr):
        obj = cls()
        obj.params = json.loads(opr.params)
        obj.name = opr.name
        obj._opr = opr
        return obj

    def compile(self, graph=None):
        op = self.opdef(**self.params)
        args = [i.var for i in self.inputs]
        outputs = rt.invoke_op(op, args)
        assert len(outputs) == len(self.outputs)
        self._opr = outputs[0].owner
        for i in range(len(self.outputs)):
            self.outputs[i].var = outputs[i]
            self.outputs[i].var.name = self.outputs[i].name
            assert self.outputs[i].owner is self

    def add_inp_var(self, x):
        self.inputs.append(x)

    def add_out_var(self, x):
        self.outputs.append(x)

    def __repr__(self):
        return "%s{%s}" % (self.name, self.type)


def str_to_mge_class(classname):
    # TODO: use megbrain C++ RTTI to replace type string
    if classname == "RNGOpr<MegDNNOpr>":
        classname = "RNGOpr"
    oprcls = getattr(sys.modules[__name__], classname, None)
    return oprcls if oprcls else ReadOnlyOpNode


class Host2DeviceCopy(OpNode):
    type = "Host2DeviceCopy"

    def __init__(self, shape=None, dtype=None, name=None, device=None):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.device = Device(device).to_c() if device else Device("xpux").to_c()
        self.outputs = []

    @classmethod
    def load(cls, opr):
        self = cls()
        self.outputs = []
        assert len(opr.outputs) == 1, "wrong number of outputs"
        self.shape = opr.outputs[0].shape
        self.dtype = opr.outputs[0].dtype
        self.name = opr.outputs[0].name
        self.device = opr.outputs[0].comp_node
        self._opr = opr
        return self

    def compile(self, graph):
        outputs = rt.make_h2d(graph, self.device, self.dtype, self.shape, self.name)
        self._opr = outputs.owner
        if len(self.outputs) == 0:
            self.outputs.append(VarNode(owner_opr=self, name=self.name))
        self.outputs[0].var = outputs
        assert self.outputs[0].owner is self


class ImmutableTensor(OpNode):
    type = "ImmutableTensor"

    def __init__(self, data=None, name=None, device=None, graph=None):
        super().__init__()
        self.name = name
        self.outputs = []
        self.graph = graph
        if data is not None:
            self.set_value(data, device)

    @property
    def device(self):
        return self._opr.outputs[0].comp_node if self._opr else None

    @device.setter
    def device(self, device):
        self.set_value(self.numpy(), device)

    @property
    def shape(self):
        return self.outputs[0].shape

    @property
    def dtype(self):
        return self._opr.outputs[0].dtype if self._opr else None

    def numpy(self):
        return self._opr.outputs[0].value if self._opr else None

    def set_value(self, data, device=None):
        assert self.graph is not None
        cn = device if device else self.device
        assert isinstance(data, (int, float, Sequence, np.ndarray))
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        elif data.dtype == np.int64:
            data = data.astype(np.int32)
        varnode = rt.make_const(self.graph, data, cn, data.dtype, self.name)
        if len(self.outputs) == 0:
            self.outputs.append(VarNode(owner_opr=self, name=self.name))
        self.outputs[0].var = varnode
        self._opr = varnode.owner

    @classmethod
    def load(cls, opr):
        self = cls()
        self.outputs = []
        self._opr = opr
        self.name = opr.outputs[0].name
        self.graph = opr.graph
        return self

    def compile(self, graph):
        assert self.outputs[0].var is self._opr.outputs[0]
        assert self.outputs[0].owner is self
        if self.graph != graph:
            self.graph = graph
            self.set_value(self.numpy())
        if self.name is not None:
            self.outputs[0].var.name = self.name


class ReadOnlyOpNode(OpNode):
    @classmethod
    def load(cls, opr):
        obj = super(ReadOnlyOpNode, cls).load(opr)
        obj.type = opr.type
        return obj

    def compile(self):
        assert self._opr is not None
        assert len(self.inputs) == len(self._opr.inputs)
        assert len(self.outputs) == len(self._opr.outputs)
        repl_dict = {}
        for ind, i in enumerate(self.inputs):
            if i.var != self._opr.inputs[ind]:
                repl_dict[self._opr.inputs[ind]] = i.var
        if bool(repl_dict):
            out_vars = replace_vars(self._opr.outputs, repl_dict)
            for ind, o in enumerate(self.outputs):
                o.var = out_vars[ind]


class Elemwise(OpNode):
    type = "Elemwise"
    opdef = builtin.Elemwise

    def __repr__(self):
        return "%s{Elemwise:%s}" % (self.name, self.params["mode"])


class ElemwiseMultiType(OpNode):
    type = "ElemwiseMultiType"
    opdef = builtin.ElemwiseMultiType

    def __repr__(self):
        return "%s{ElemwiseMultiType:%s}" % (self.name, self.params["mode"])

    @classmethod
    def load(cls, opr):
        obj = super(ElemwiseMultiType, cls).load(opr)
        obj.params["dtype"] = opr.outputs[0].dtype
        return obj


@register_flops(Elemwise, ElemwiseMultiType)
def flops_elemwise(opnode: Elemwise, inputs, outputs):
    return np.prod(outputs[0].shape)


class Reduce(OpNode):
    type = "Reduce"
    opdef = builtin.Reduce


class TypeCvt(OpNode):
    type = "TypeCvt"
    opdef = builtin.TypeCvt

    @classmethod
    def load(cls, opr):
        obj = super(TypeCvt, cls).load(opr)
        t_dtype = opr.outputs[0].dtype
        obj.params["dtype"] = t_dtype
        return obj


class MatrixInverse(OpNode):
    type = "MatrixInverse"
    opdef = builtin.MatrixInverse


class MatrixMul(OpNode):
    type = "MatrixMul"
    opdef = builtin.MatrixMul


@register_flops(MatrixMul)
def flops_matmul(opnode: MatrixMul, inputs, outputs):
    assert len(inputs[0].shape) == 2 and len(outputs[0].shape) == 2
    mid_shape = inputs[0].shape[1]
    return np.prod(outputs[0].shape) * mid_shape


class BatchedMatrixMul(OpNode):
    type = "BatchedMatmul"
    opdef = builtin.BatchedMatrixMul


@register_flops(BatchedMatrixMul)
def flops_batchmatmul(opnode: BatchedMatrixMul, inputs, outputs):
    assert len(inputs[0].shape) == 3 and len(outputs[0].shape) == 3
    mid_shape = inputs[0].shape[2]
    return np.prod(outputs[0].shape) * mid_shape


class Dot(OpNode):
    type = "Dot"
    opdef = builtin.Dot


class SVD(OpNode):
    type = "SVD"
    opdef = builtin.SVD


class ConvolutionForward(OpNode):
    type = "Convolution"
    opdef = builtin.Convolution


class ConvolutionBackwardData(OpNode):
    type = "ConvTranspose"
    opdef = builtin.ConvolutionBackwardData


class DeformableConvForward(OpNode):
    type = "DeformableConv"
    opdef = builtin.DeformableConv


class GroupLocalForward(OpNode):
    type = "GroupLocal"
    opdef = builtin.GroupLocal


class PoolingForward(OpNode):
    type = "Pooling"
    opdef = builtin.Pooling


class AdaptivePoolingForward(OpNode):
    type = "AdaptivePooling"
    opdef = builtin.AdaptivePooling


class ROIPoolingForward(OpNode):
    type = "ROIPooling"
    opdef = builtin.ROIPooling


class DeformablePSROIPoolingForward(OpNode):
    type = "DeformablePSROIPooling"
    opdef = builtin.DeformablePSROIPooling


class ConvBiasForward(OpNode):
    type = "ConvBias"
    opdef = builtin.ConvBias

    @classmethod
    def load(cls, opr):
        obj = super(ConvBiasForward, cls).load(opr)
        obj.params["dtype"] = opr.outputs[0].dtype
        return obj


@register_flops(
    ConvolutionForward, ConvBiasForward,
)
def flops_conv(opnode: ConvolutionForward, inputs, outputs):
    param_W_shape = inputs[1].shape
    kh = param_W_shape[-2]
    kw = param_W_shape[-1]
    if len(param_W_shape) == 5:
        num_input = param_W_shape[2]
    else:
        num_input = param_W_shape[1]
    NCHW = np.prod(outputs[0].shape)
    bias = 1 if isinstance(opnode, ConvBiasForward) else 0
    # N x Cout x H x W x  (Cin x Kw x Kh)
    return NCHW * (num_input * kw * kh + bias)


@register_receptive_field(ConvolutionForward, ConvBiasForward)
def receptive_field(opnode: ConvolutionForward, inputs, outputs):
    pre_rf, pre_stride = preprocess_receptive_field(opnode, inputs, outputs)
    param_W_shape = inputs[1].shape
    kh = param_W_shape[-2]
    kw = param_W_shape[-1]
    rf = (
        kh * pre_stride[0] + pre_rf[0] - pre_stride[0],
        kw * pre_stride[1] + pre_rf[1] - pre_stride[1],
    )
    stride = (
        opnode.params["stride_h"] * pre_stride[0],
        opnode.params["stride_w"] * pre_stride[1],
    )
    opnode._rf = rf
    opnode._stride = stride
    return rf, stride


class BatchConvBiasForward(OpNode):
    type = "BatchConvBias"
    opdef = builtin.BatchConvBias

    @classmethod
    def load(cls, opr):
        obj = super(BatchConvBiasForward, cls).load(opr)
        obj.params["dtype"] = opr.outputs[0].dtype
        return obj


class BatchNormForward(OpNode):
    type = "BatchNorm"
    opdef = builtin.BatchNorm
    output_idx = -1


class ROIAlignForward(OpNode):
    type = "ROIAlign"
    opdef = builtin.ROIAlign


class WarpPerspectiveForward(OpNode):
    type = "WarpPerspective"
    opdef = builtin.WarpPerspective


class WarpAffineForward(OpNode):
    type = "WarpAffine"
    opdef = builtin.WarpAffine


class RemapForward(OpNode):
    type = "Remap"
    opdef = builtin.Remap


class ResizeForward(OpNode):
    type = "Resize"
    opdef = builtin.Resize


class IndexingOneHot(OpNode):
    type = "IndexingOneHot"
    opdef = builtin.IndexingOneHot


class IndexingSetOneHot(OpNode):
    type = "IndexingSetOneHot"
    opdef = builtin.IndexingSetOneHot


class Copy(OpNode):
    type = "Copy"
    opdef = builtin.Copy

    @classmethod
    def load(cls, opr):
        obj = super(Copy, cls).load(opr)
        obj.params["comp_node"] = opr.outputs[0].comp_node
        return obj


class ArgsortForward(OpNode):
    type = "Argsort"
    opdef = builtin.Argsort


class Argmax(OpNode):
    type = "Argmax"
    opdef = builtin.Argmax


class Argmin(OpNode):
    type = "Argmin"
    opdef = builtin.Argmin


class CondTake(OpNode):
    type = "CondTake"
    opdef = builtin.CondTake


class TopK(OpNode):
    type = "TopK"
    opdef = builtin.TopK


class NvOf(OpNode):
    type = "NvOf"
    opdef = builtin.NvOf


class RNGOpr(OpNode):
    @classmethod
    def load(cls, opr):
        obj = super(RNGOpr, cls).load(opr)
        if len(obj.params) == 3:
            obj.opdef = builtin.GaussianRNG
            obj.type = "GaussianRNG"
        else:
            obj.opdef = builtin.UniformRNG
            obj.type = "UniformRNG"
        return obj


class Linspace(OpNode):
    type = "Linspace"
    opdef = builtin.Linspace

    @classmethod
    def load(cls, opr):
        obj = super(Linspace, cls).load(opr)
        obj.params["comp_node"] = opr.outputs[0].comp_node
        return obj


class Eye(OpNode):
    type = "Eye"
    opdef = builtin.Eye

    @classmethod
    def load(cls, opr):
        obj = super(Eye, cls).load(opr)
        obj.params["dtype"] = opr.outputs[0].dtype
        obj.params["comp_node"] = opr.outputs[0].comp_node
        return obj


class GetVarShape(OpNode):
    type = "GetVarShape"
    opdef = builtin.GetVarShape


class Concat(OpNode):
    type = "Concat"
    opdef = builtin.Concat

    @classmethod
    def load(cls, opr):
        obj = super(Concat, cls).load(opr)
        obj.params["comp_node"] = Device("xpux").to_c()
        return obj


class Broadcast(OpNode):
    type = "Broadcast"
    opdef = builtin.Broadcast


class Identity(OpNode):
    type = "Identity"
    opdef = builtin.Identity


class NMSKeep(OpNode):
    type = "NMSKeep"
    opdef = builtin.NMSKeep


# class ParamPackSplit
# class ParamPackConcat


class Dimshuffle(OpNode):
    type = "Dimshuffle"
    opdef = builtin.Dimshuffle

    @classmethod
    def load(cls, opr):
        obj = super(Dimshuffle, cls).load(opr)
        del obj.params["ndim"]
        return obj


class Reshape(OpNode):
    type = "Reshape"
    opdef = builtin.Reshape


class AxisAddRemove(OpNode):
    type = "AxisAddRemove"

    @classmethod
    def load(cls, opr):
        obj = cls()
        obj.name = opr.name
        obj._opr = opr
        params = json.loads(opr.params)
        desc = params["desc"]
        method = None
        axis = []
        for i in desc:
            if method is None:
                method = i["method"]
            assert method == i["method"]
            axis.append(i["axisnum"])
        obj.params = {"axis": axis}
        obj.opdef = builtin.AddAxis if desc[0]["method"] == 0 else builtin.RemoveAxis
        return obj


class IndexingBase(OpNode):
    @classmethod
    def load(cls, opr):
        obj = cls()
        obj.name = opr.name
        obj._opr = opr
        params = json.loads(opr.params)
        items = [
            [
                p["axis"],
                bool(p["begin"]),
                bool(p["end"]),
                bool(p["step"]),
                bool(p["idx"]),
            ]
            for p in params
        ]
        obj.params["items"] = items
        return obj


class Subtensor(IndexingBase):
    type = "Subtensor"
    opdef = builtin.Subtensor


class SetSubtensor(IndexingBase):
    type = "SetSubtensor"
    opdef = builtin.SetSubtensor


class IncrSubtensor(IndexingBase):
    type = "IncrSubtensor"
    opdef = builtin.IncrSubtensor


class IndexingMultiAxisVec(IndexingBase):
    type = "IndexingMultiAxisVec"
    opdef = builtin.IndexingMultiAxisVec


class IndexingSetMultiAxisVec(IndexingBase):
    type = "IndexingSetMultiAxisVec"
    opdef = builtin.IndexingSetMultiAxisVec


class IndexingIncrMultiAxisVec(IndexingBase):
    type = "IndexingIncrMultiAxisVec"
    opdef = builtin.IndexingIncrMultiAxisVec


class MeshIndexing(IndexingBase):
    type = "MeshIndexing"
    opdef = builtin.MeshIndexing


class SetMeshIndexing(IndexingBase):
    type = "SetMeshIndexing"
    opdef = builtin.SetMeshIndexing


class IncrMeshIndexing(IndexingBase):
    type = "IncrMeshIndexing"
    opdef = builtin.IncrMeshIndexing


class BatchedMeshIndexing(IndexingBase):
    type = "BatchedMeshIndexing"
    opdef = builtin.BatchedMeshIndexing


class BatchedSetMeshIndexing(IndexingBase):
    type = "BatchedSetMeshIndexing"
    opdef = builtin.BatchedSetMeshIndexing


class BatchedIncrMeshIndexing(IndexingBase):
    type = "BatchedIncrMeshIndexing"
    opdef = builtin.BatchedIncrMeshIndexing


# class CollectiveComm
# class RemoteSend
# class RemoteRecv
# class TQT
# class FakeQuant
# class InplaceAdd


class AssertEqual(OpNode):
    type = "AssertEqual"
    opdef = builtin.AssertEqual


class CvtColorForward(OpNode):
    type = "CvtColor"
    opdef = builtin.CvtColor
