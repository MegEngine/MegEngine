/**
 * \file src/core/include/megbrain/ir/ops.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_OPS
#define MGB_OPS

include "base.td"
include "param_defs.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def Elemwise : MgbHashableOp<"Elemwise", [ElemwiseParam], [NoSideEffect]> {
  let inputs = (ins Variadic<AnyType>:$input);
  let results = (outs AnyType);
}

def Reduce: MgbHashableOp<"Reduce", [ReduceParam]>;

def TypeCvt: MgbHashableOp<"TypeCvt", [], [NoSideEffect]> {
  let inputs = (ins AnyType:$inputs);
  let extraArguments = (ins
    TypeAttr:$idtype,
    MgbDTypeAttr:$dtype
  );
  let results = (outs AnyType);
}

def MatrixMul: MgbHashableOp<"MatrixMul", [MatrixMulParam]>;

def BatchedMatrixMul: MgbHashableOp<"BatchedMatmul", [MatrixMulParam]>;

def Dot: MgbHashableOp<"Dot", [EmptyParam]>;

def SVD: MgbHashableOp<"SVD", [SVDParam]>;

def Convolution : MgbHashableOp<"Convolution", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]>;

def ConvolutionBackwardData: MgbHashableOp<"ConvolutionBackwardData", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]>;

def GroupLocal: MgbHashableOp<"GroupLocal", [ConvolutionParam]>;

def Pooling: MgbHashableOp<"Pooling", [PoolingParam]>;

def AdaptivePooling : MgbHashableOp<"AdaptivePooling", [AdaptivePoolingParam]>;

def ROIPooling: MgbHashableOp<"ROIPooling", [ROIPoolingParam]>;

def ConvBias : MgbHashableOp<"ConvBias", [ConvBiasParam, ExecutionPolicyParamBase<"policy">]> {
  let extraArguments = (ins
    MgbDTypeAttr:$dtype
  );
}

def BatchConvBias : MgbHashableOp<"BatchConvBias", [BatchConvBiasParam, ExecutionPolicyParamBase<"policy">]> {
  let extraArguments = (ins
    MgbDTypeAttr:$dtype
  );
}

def BatchNorm : MgbHashableOp<"BatchNorm", [BNParam]>;

def ROIAlign: MgbHashableOp<"ROIAlign", [ROIAlignParam]>;

def WarpPerspective: MgbHashableOp<"WarpPerspective", [WarpPerspectiveParam]>;

def Remap: MgbHashableOp<"Remap", [RemapParam]>;

def IndexingOneHot: MgbHashableOp<"IndexingOneHot", [AxisParam]>;

def IndexingSetOneHot: MgbHashableOp<"IndexingSetOneHot", [AxisParam]>;

def Copy: MgbHashableOp<"Copy"> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Argsort: MgbHashableOp<"Argsort", [ArgsortParam]>;

def Argmax : MgbHashableOp<"Argmax", [AxisParam]>;

def Argmin : MgbHashableOp<"Argmin", [AxisParam]>;

def CondTake : MgbHashableOp<"CondTake">;

def TopK: MgbHashableOp<"TopK", [TopKParam]>;

def NvOf: MgbHashableOp<"NvOf", [NvOfParam]>;

def UniformRNG: MgbHashableOp<"UniformRNG", [UniformRNGParam]> {
  let hashFunction = [{return mgb::hash($_self.dyn_typeinfo());}];
  let cmpFunction = [{return true;}];
}

def GaussianRNG: MgbHashableOp<"GaussianRNG", [GaussianRNGParam]> {
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(mgb::hash($_self.mean), mgb::hash($_self.std)));
  }];
  let cmpFunction = [{return $0.mean == $1.mean && $0.std == $1.std;}];
}

def Linspace: MgbHashableOp<"Linspace", [LinspaceParam]> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Eye: MgbHashableOp<"Eye", [EyeParam]> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def GetVarShape : MgbHashableOp<"GetVarShape", [OptionalAxisV1Param]>;

def Concat: MgbHashableOp<"Concat", [AxisParam]> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Broadcast : MgbHashableOp<"Broadcast", [EmptyParam]>;

def Identity: MgbHashableOp<"Identity">;

def CollectiveComm : MgbHashableOp<"CollectiveComm", [CollectiveCommParam]> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbUI32Attr:$nr_devices,
    MgbUI32Attr:$rank,
    MgbBoolAttr:$is_root,
    MgbBoolAttr:$local_grad,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbDTypeAttr:$dtype,
    MgbStringAttr:$backend,
    MgbStringAttr:$comp_node
  );
}

def RemoteSend : MgbHashableOp<"RemoteSend"> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbUI32Attr:$rank_to
  );
}

def RemoteRecv : MgbHashableOp<"RemoteRecv"> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbUI32Attr:$rank_from,
    MgbCompNodeAttr:$cn,
    MgbTensorShapeAttr:$shape,
    MgbDTypeAttr:$dtype
  );
}

def NMSKeep : MgbHashableOp<"NMSKeep"> {
  let extraArguments = (ins
    MgbF32Attr:$iou_thresh,
    MgbUI32Attr:$max_output
  );
}

def ParamPackSplit : MgbHashableOp<"ParamPackSplit"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$offsets,
    MgbArrayAttr<MgbArrayAttr<MgbSizeTAddr>>:$shapes
  );
}

def ParamPackConcat : MgbHashableOp<"ParamPackConcat"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$offsets
  );
}

def Dimshuffle: MgbHashableOp<"Dimshuffle"> {
  let inputs = (ins AnyMemRef:$input);
  let extraArguments = (ins MgbArrayAttr<MgbI32Attr>:$pattern);
  let results = (outs AnyMemRef);
}

def Reshape: MgbHashableOp<"Reshape", [OptionalAxisV1Param]>;

// TODO: merge Add/Remove Axis into AxisAddRemove as megbrain?
def AddAxis: MgbHashableOp<"AddAxis"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$axis
  );
}
def RemoveAxis: MgbHashableOp<"RemoveAxis"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$axis
  );
}

class FancyIndexingBase<string name>: MgbHashableOp<name> {
  let extraArguments = (ins
    MgbArrayAttr<MgbTupleAttr<
      [MgbI8Attr, MgbBoolAttr, MgbBoolAttr, MgbBoolAttr, MgbBoolAttr]>>:$items
  );
}

def Subtensor: FancyIndexingBase<"Subtensor">;
def SetSubtensor: FancyIndexingBase<"SetSubtensor">;
def IncrSubtensor: FancyIndexingBase<"IncrSubtensor">;
def IndexingMultiAxisVec: FancyIndexingBase<"IndexingMultiAxisVec">;
def IndexingSetMultiAxisVec: FancyIndexingBase<"IndexingSetMultiAxisVec">;
def IndexingIncrMultiAxisVec: FancyIndexingBase<"IndexingIncrMultiAxisVec">;
def MeshIndexing: FancyIndexingBase<"MeshIndexing">;
def IncrMeshIndexing: FancyIndexingBase<"IncrMeshIndexing">;
def SetMeshIndexing: FancyIndexingBase<"SetMeshIndexing">;
def BatchedMeshIndexing: FancyIndexingBase<"BatchedMeshIndexing">;
def BatchedIncrMeshIndexing: FancyIndexingBase<"BatchedIncrMeshIndexing">;
def BatchedSetMeshIndexing: FancyIndexingBase<"BatchedSetMeshIndexing">;

def FakeQuant: MgbHashableOp<"FakeQuant", [FakeQuantParam]>;
def ElemwiseMultiType: MgbHashableOp<"ElemwiseMultiType", [ElemwiseMultiTypeParam]> {
  let extraArguments = (ins
    MgbDTypeAttr:$dtype
  );
}

#endif // MGB_OPS
