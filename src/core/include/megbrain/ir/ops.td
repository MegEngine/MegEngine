/**
 * \file src/core/include/megbrain/ir/ops.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
  let nameFunction = [{
    return to_string($_self.mode);
  }];
}

def Reduce: MgbHashableOp<"Reduce", [ReduceParam]>{
  let extraArguments = (ins
    MgbBoolAttr:$keepdim
  );
}

def TypeCvt: MgbHashableOp<"TypeCvt", [], [NoSideEffect]> {
  let inputs = (ins AnyType:$inputs);
  let extraArguments = (ins
    TypeAttr:$idtype,
    MgbDTypeAttr:$dtype
  );
  let results = (outs AnyType);
}

def MatrixInverse: MgbHashableOp<"MatrixInverse", [EmptyParam]>;

def MatrixMul: MgbHashableOp<"MatrixMul", [MatrixMulParam, ExecutionPolicyParamBase<"policy">]>;

def BatchedMatrixMul: MgbHashableOp<"BatchedMatmul", [MatrixMulParam, ExecutionPolicyParamBase<"policy">]>;

def Dot: MgbHashableOp<"Dot", [EmptyParam]>;

def SVD: MgbHashableOp<"SVD", [SVDParam]>;

def Convolution : MgbHashableOp<"Convolution", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]>;

def ConvolutionBackwardData: MgbHashableOp<"ConvolutionBackwardData", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]> {
  let extraArguments = (ins
    MgbDTypeAttr:$dtype
  );
}

def Convolution3D: MgbHashableOp<"Convolution3D", [Convolution3DParam, ExecutionPolicyParamBase<"policy">]>;

def Convolution3DBackwardData: MgbHashableOp<"Convolution3DBackwardData", [Convolution3DParam, ExecutionPolicyParamBase<"policy">]>;

def DeformableConv : MgbHashableOp<"DeformableConv", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]>;

def GroupLocal: MgbHashableOp<"GroupLocal", [ConvolutionParam]>;

def Pooling: MgbHashableOp<"Pooling", [PoolingParam, ExecutionPolicyParamBase<"policy">]>;

def AdaptivePooling : MgbHashableOp<"AdaptivePooling", [AdaptivePoolingParam]> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$shape
  );
}

def ROIPooling: MgbHashableOp<"ROIPooling", [ROIPoolingParam]>;

def DeformablePSROIPooling : MgbHashableOp<"DeformablePSROIPooling", [DeformablePSROIPoolingParam]>;

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

def Images2Neibs : MgbHashableOp<"Images2Neibs", [Images2NeibsParam]>;

def SlidingWindowTranspose : MgbHashableOp<"SlidingWindowTranspose", [SlidingWindowTransposeParam]>;

def BatchNorm : MgbHashableOp<"BatchNorm", [BNParam]>;

def BatchNormBackward : MgbHashableOp<"BatchNormBackward", [BNParam]>;

def ROIAlign: MgbHashableOp<"ROIAlign", [ROIAlignParam]>;
def Correlation: MgbHashableOp<"Correlation", [CorrelationParam]>;

def WarpPerspective: MgbHashableOp<"WarpPerspective", [WarpPerspectiveParam]>;

def WarpAffine: MgbHashableOp<"WarpAffine", [WarpAffineParam]>;

def Remap: MgbHashableOp<"Remap", [RemapParam]>;

def Resize: MgbHashableOp<"Resize", [ResizeParam]>;

def IndexingOneHot: MgbHashableOp<"IndexingOneHot", [AxisParam]>;

def IndexingSetOneHot: MgbHashableOp<"IndexingSetOneHot", [AxisParam]>;

def Copy: MgbHashableOp<"Copy"> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Borrow: MgbHashableOp<"Borrow"> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Barrier: MgbHashableOp<"Barrier"> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node,
    MgbUI32Attr:$nr_outputs
  );
}

def Argsort: MgbHashableOp<"Argsort", [ArgsortParam]>;

def Argmax : MgbHashableOp<"Argmax", [AxisParam]>;

def Argmin : MgbHashableOp<"Argmin", [AxisParam]>;

def CondTake : MgbHashableOp<"CondTake">;

def TopK: MgbHashableOp<"TopK", [TopKParam]>;

def NvOf: MgbHashableOp<"NvOf", [NvOfParam]>;

def UniformRNG: MgbHashableOp<"UniformRNG", [UniformRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash($_self.handle),
        mgb::hash($_self.dtype.enumv())
      )
    );
  }];
  let cmpFunction = [{return $0.handle == $1.handle && $0.dtype == $1.dtype;}];
}

def GaussianRNG: MgbHashableOp<"GaussianRNG", [GaussianRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash($_self.handle),
        mgb::hash_pair_combine(
          mgb::hash($_self.mean),
          mgb::hash_pair_combine(
            mgb::hash($_self.std),
            mgb::hash($_self.dtype.enumv())
          )
        )
      )
    );
  }];
  let cmpFunction = [{return $0.handle == $1.handle && $0.mean == $1.mean && $0.std == $1.std && $0.dtype == $1.dtype;}];
}

def GammaRNG: MgbHashableOp<"GammaRNG", [GammaRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash($_self.handle)
      );
  }];
  let cmpFunction = [{return $0.handle == $1.handle;}];
}

def PoissonRNG: MgbHashableOp<"PoissonRNG", [PoissonRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash($_self.handle)
      );
  }];
  let cmpFunction = [{return $0.handle == $1.handle;}];
}

def BetaRNG: MgbHashableOp<"BetaRNG", [BetaRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash($_self.handle)
      );
  }];
  let cmpFunction = [{return $0.handle == $1.handle;}];
}

def PermutationRNG: MgbHashableOp<"PermutationRNG", [PermutationRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash($_self.handle),
        mgb::hash($_self.dtype.enumv())
      )
    );
  }];
  let cmpFunction = [{return $0.handle == $1.handle && $0.dtype == $1.dtype;}];
}

def ShuffleRNG: MgbHashableOp<"ShuffleRNG", [ShuffleRNGParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash($_self.handle)
      );
  }];
  let cmpFunction = [{return $0.handle == $1.handle;}];
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

def Diag: MgbHashableOp<"Diag", [DiagParam]>;

def GetVarShape : MgbHashableOp<"GetVarShape", [OptionalAxisV1Param]>;

def Concat: MgbHashableOp<"Concat", [AxisParam]> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Broadcast : MgbHashableOp<"Broadcast", [EmptyParam]> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$shape
  );
}

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
    MgbUI32Attr:$rank_to,
    MgbStringAttr:$backend
  );
}

def RemoteRecv : MgbHashableOp<"RemoteRecv"> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbUI32Attr:$rank_from,
    MgbCompNodeAttr:$cn,
    MgbArrayAttr<MgbI32Attr>:$shape,
    MgbDTypeAttr:$dtype,
    MgbStringAttr:$backend
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

def Reshape: MgbHashableOp<"Reshape", [OptionalAxisV1Param]> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$shape
  );
}

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
def AssertEqual: MgbHashableOp<"AssertEqual",[AssertEqualParam]>;
def TQT: MgbHashableOp<"TQT", [TQTParam]>;
def LSQ: MgbHashableOp<"LSQ", [LSQParam]>;
def Softmax: MgbHashableOp<"Softmax", [SoftmaxParam]>;
def ElemwiseMultiType: MgbHashableOp<"ElemwiseMultiType", [ElemwiseMultiTypeParam]> {
  let extraArguments = (ins
    MgbDTypeAttr:$dtype
  );
  let nameFunction = [{
    return to_string($_self.mode);
  }];
}

def InplaceAdd: MgbHashableOp<"InplaceAdd", [EmptyParam]>;

def TensorRTRuntime: MgbHashableOp<"TensorRTRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size
  );
}

def AtlasRuntime: MgbHashableOp<"AtlasRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size
  );
}

def CambriconRuntime: MgbHashableOp<"CambriconRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size,
    MgbStringAttr:$symbol,
    MgbBoolAttr:$tensor_dim_mutable
  );
}

def MagicMindRuntime: MgbHashableOp<"MagicMindRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size
  );
}

def CvtColor: MgbHashableOp<"CvtColor", [CvtColorParam]>;

def CheckNonFinite: MgbHashableOp<"CheckNonFinite", [CheckNonFiniteParam]>;

def FastpathCopy: MgbHashableOp<"FastpathCopy">;

def PixelShuffle: MgbHashableOp<"PixelShuffle"> {
  let extraArguments = (ins
    MgbI32Attr:$factor
  );
}

def PixelShuffleBackward: MgbHashableOp<"PixelShuffleBackward"> {
  let extraArguments = (ins
    MgbI32Attr:$factor
  );
}

def ExternOpr: MgbHashableOp<"ExternOpr"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbArrayAttr<MgbSizeTAddr>>:$output_shapes,
    MgbStringAttr:$name,
    MgbStringAttr:$data,
    MgbSizeTAddr:$data_len,
    MgbArrayAttr<MgbDTypeAttr>:$output_dtypes
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash($_self.name),
        mgb::hash($_self.data))
      );
  }];
}

def Cumsum: MgbHashableOp<"Cumsum", [CumsumParam]>;

def Split: MgbHashableOp<"Split", [EmptyParam]> {
  let extraArguments = (ins
    MgbI32Attr:$axis,
    MgbI32Attr:$nsections
  );
}

def Padding: MgbHashableOp<"Padding", [PaddingParam]>;

def LRN: MgbHashableOp<"LRN", [LRNParam]>;

def LayerNorm: MgbHashableOp<"LayerNorm", [LayerNormParam]>;
def RNNCell: MgbHashableOp<"RNNCell", [RNNCellParam]>;

def LSTMCell: MgbHashableOp<"LSTMCell", [EmptyParam]>;

def RNN: MgbHashableOp<"RNN", [RNNParam]>;

def LSTM: MgbHashableOp<"LSTM", [LSTMParam]>;

def Dropout: MgbHashableOp<"Dropout", [DropoutParam]> {
  let extraArguments = (ins
    MgbSizeTAddr:$handle
  );
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(
        mgb::hash($_self.drop_prob),
        mgb::hash($_self.handle))
      );
  }];
  let cmpFunction = [{return $0.handle == $1.handle && $0.drop_prob == $1.drop_prob;}];

}
#endif // MGB_OPS
