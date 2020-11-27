/**
 * \file src/jit/include/megbrain/jit/mlir/ir/mgb_dialect.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_DIALECT
#define MGB_DIALECT

include "ops.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

class GenericOp<string mnemonic, list<OpTrait> traits = []> :
  Op<Mgb_Dialect, mnemonic, traits>;

def ReturnOp : GenericOp<"return",
    [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an no tensor operand and produces no results.
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "attr-dict";
}

def ConstantScalarOp: GenericOp<"sconst", [NoSideEffect]> {
  let summary = "scalar constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs F32:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, float value", [{
      result.addAttribute("value", builder->getF32FloatAttr(value));
      result.addTypes(builder->getF32Type());
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
    FloatAttr getFloatAttr() { return getAttrOfType<FloatAttr>("value"); }
  }];

}

def AssignOp :  GenericOp<"assign", []> {
  let summary = "assign op";
  let description = [{
    assign rhs to lhs without results
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs);
}

#endif // MGB_DIALECT
