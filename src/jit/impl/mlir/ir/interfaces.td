/**
 * \file src/jit/impl/mlir/ir/interfaces.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_MLIR_INTERFACES
#define MGB_MLIR_INTERFACES

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def GenericBuilderInterface : OpInterface<"GenericBuilder"> {
  let methods = [
  StaticInterfaceMethod<"TODO", "Type", "getResultType", (ins "ArrayRef<Value>":$operands)>,
  StaticInterfaceMethod<"TODO", "Operation*", "create", (ins
  "OpBuilder*":$builder,
  "Location":$loc,
  "ArrayRef<Value>":$operands
  )>,
  ];
}

def ElemwiseOpInterface : OpInterface<"ElemwiseOp">;

#endif
