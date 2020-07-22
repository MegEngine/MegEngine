/**
 * \file src/jit/impl/mlir/ir/shape_inference_interface.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_JIT_SHAPE_INFERENCE_INTERFACE
#define MGB_JIT_SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
  InterfaceMethod<"Infer and set the output shape for the current operation.",
  "void", "infer_shapes">
  ];
}

#endif // MGB_SHAPE_INFERENCE_INTERFACE
