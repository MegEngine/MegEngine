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

include "./interfaces.td"
include "./predicates.td"

def Mgb_Dialect : Dialect {
  let name = "mgb";
  let cppNamespace = "mgb::jit";
}

class ElemwiseBuilderImpl {
  code ElemwiseBuilderImpl_create = [{
    static Operation* create(OpBuilder* builder, Location loc, ValueRange operands) {
      OperationState state(loc, getOperationName());
      state.addOperands(operands);
      state.addTypes(getResultType(operands));
      return builder->createOperation(state);
    }
  }];
}

class ElemwiseOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
  Op<Mgb_Dialect, mnemonic, !listconcat(traits, [ElemwiseOpInterface,
  GenericBuilderInterface])>, ElemwiseBuilderImpl;

class GenericOp<string mnemonic, list<OpTrait> traits = []> :
  Op<Mgb_Dialect, mnemonic, traits>;

class ElemwiseUnaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    ElemwiseOp<mnemonic, traits> {
  let arguments = (ins F32MemRef:$lhs);
  let results = (outs F32MemRef);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ValueRange operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]>, OpBuilder <
    "OpBuilder& builder, OperationState& result, Value lhs", [{
      result.addOperands(lhs);
      result.addTypes(getResultType({lhs}));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ValueRange operands) {
      return deduce_result_type(operands);
    }
  }] # ElemwiseBuilderImpl_create;
}

def ReluOp : ElemwiseUnaryOp<"relu", [NoSideEffect]>;
def AbsOp : ElemwiseUnaryOp<"abs", [NoSideEffect]>;
def NegOp : ElemwiseUnaryOp<"negate", [NoSideEffect]>;
/* ACOS */
/* ASIN */
def CeilOp : ElemwiseUnaryOp<"ceil", [NoSideEffect]>;
def CosOp : ElemwiseUnaryOp<"cos", [NoSideEffect]>;
def ExpOp : ElemwiseUnaryOp<"exp", [NoSideEffect]>;
def ExpM1Op : ElemwiseUnaryOp<"expm1", [NoSideEffect]>;
def FloorOp : ElemwiseUnaryOp<"floor", [NoSideEffect]>;
def LogOp : ElemwiseUnaryOp<"log", [NoSideEffect]>;
def Log1POp : ElemwiseUnaryOp<"log1p", [NoSideEffect]>;
def SigmoidOp: ElemwiseUnaryOp<"sigmoid", [NoSideEffect]>;
def SinOp : ElemwiseUnaryOp<"sin", [NoSideEffect]>;
def TanhOp : ElemwiseUnaryOp<"tanh", [NoSideEffect]>;
def FastTanhOp : ElemwiseUnaryOp<"fast_tanh", [NoSideEffect]>;
def HswishOp : ElemwiseUnaryOp<"hswish", [NoSideEffect]>;
def RoundOp : ElemwiseUnaryOp<"round", [NoSideEffect]>;
/* ERF */
/* ERFINV */
/* ERFC */
/* ERFCINV */

class ElemwiseBinaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
  ElemwiseOp<mnemonic, traits> {

  let arguments = (ins ElemwiseFloatAny:$lhs, ElemwiseFloatAny:$rhs);
  let results = (outs F32MemRef);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ValueRange operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >, OpBuilder <
    "OpBuilder& builder, OperationState& result, Value lhs, Value rhs", [{
      result.addOperands(lhs);
      result.addOperands(rhs);
      result.addTypes(getResultType({lhs, rhs}));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ValueRange operands) {
      return deduce_result_type(operands);
    }
  }] # ElemwiseBuilderImpl_create;
}

def AbsGradOp : ElemwiseBinaryOp<"abs_grad", [NoSideEffect]>;
def AddOp : ElemwiseBinaryOp<"add", [Commutative, NoSideEffect]>;
def FloorDivOp : ElemwiseBinaryOp<"floor_div", [NoSideEffect]>;
def MaxOp : ElemwiseBinaryOp<"max", [Commutative, NoSideEffect]>;
def MinOp : ElemwiseBinaryOp<"min", [Commutative, NoSideEffect]>;
def ModOp : ElemwiseBinaryOp<"mod", [NoSideEffect]>;
def MulOp : ElemwiseBinaryOp<"mul", [Commutative, NoSideEffect]>;
def SubOp : ElemwiseBinaryOp<"sub", [NoSideEffect]>;
def SigmoidGradOp :  ElemwiseBinaryOp<"sigmoid_grad", [NoSideEffect]>;
def SwishGt0Op :  ElemwiseBinaryOp<"switch_gt0", [NoSideEffect]>;
def TanhGradOp :  ElemwiseBinaryOp<"tanh_grad", [NoSideEffect]>;
def LtOp :  ElemwiseBinaryOp<"lt", [NoSideEffect]>;
def LeqOp :  ElemwiseBinaryOp<"leq", [NoSideEffect]>;
def EqOp :  ElemwiseBinaryOp<"eq", [Commutative, NoSideEffect]>;
def FuseAddReluOp :  ElemwiseBinaryOp<"fuse_add_relu", [NoSideEffect]>;
def TrueDivOp : ElemwiseBinaryOp<"true_div", [NoSideEffect]>;
/* POW */
def LogSumExpOp : ElemwiseBinaryOp<"log_sum_exp", [Commutative, NoSideEffect]>;
def FuseAddTanhOp :  ElemwiseBinaryOp<"fuse_add_tanh", [NoSideEffect]>;
def FastTanhGradOp :  ElemwiseBinaryOp<"fast_tanh_grad", [NoSideEffect]>;
def FuseAddSigmoidOp :  ElemwiseBinaryOp<"fuse_add_sigmoid", [NoSideEffect]>;
def HswishGradOp :  ElemwiseBinaryOp<"hswish_grad", [NoSideEffect]>;
def FuseAddHswishOp : ElemwiseBinaryOp<"fuse_add_hswish", [NoSideEffect]>;
/* ATAN2 */

class ElemwiseTernaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
  ElemwiseOp<mnemonic, traits> {

  let arguments = (ins ElemwiseFloatAny:$x, ElemwiseFloatAny:$y, ElemwiseFloatAny:$z);
  let results = (outs F32MemRef);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ValueRange operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >, OpBuilder <
    "OpBuilder& builder, OperationState& result, Value x, Value y, Value z", [{
      result.addOperands(x);
      result.addOperands(y);
      result.addOperands(z);
      result.addTypes(getResultType({x, y, z}));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ValueRange operands) {
      return deduce_result_type(operands);
    }
  }] # ElemwiseBuilderImpl_create;
}

def CondLeqMovOp: ElemwiseTernaryOp<"cond_leq_mov", [NoSideEffect]>;
def FuseMulAdd3Op: ElemwiseTernaryOp<"fuse_mul_add3", [NoSideEffect]>;

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

  let arguments = (ins F32MemRef:$lhs, F32MemRef:$rhs);
}

#endif
