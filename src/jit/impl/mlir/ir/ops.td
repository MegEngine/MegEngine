/**
 * \file src/jit/impl/mlir/ir/ops.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_MLIR_OPS
#define MGB_MLIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "./shape_inference_interface.td"

def Mgb_Dialect : Dialect {
  let name = "mgb";
  let cppNamespace = "mgb::jit";
}

class ElemwiseOp<string mnemonic, list<OpTrait> traits = []> :
  Op<Mgb_Dialect, mnemonic, traits>;

class GenericOp<string mnemonic, list<OpTrait> traits = []> :
  Op<Mgb_Dialect, mnemonic, traits>;

def AddOp : ElemwiseOp<"add",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F32MemRef:$lhs, F32MemRef:$rhs);
  let results = (outs F32MemRef);

  // Specify a parser and printer method.
  let parser = [{ return ::parseBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<"OpBuilder &b, OperationState &state, Value lhs, Value rhs">
  ];
}

def ReturnOp : GenericOp<"return",
    [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an no tensor operand and produces no results.
  }];

}

def AssignOp :  GenericOp<"assign", []> {
  let summary = "assign op";
  let description = [{
    assign rhs to lhs without results
  }];

  let arguments = (ins F32MemRef:$lhs, F32MemRef:$rhs);
}

#endif
