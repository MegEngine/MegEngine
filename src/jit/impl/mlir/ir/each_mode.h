/**
 * \file src/jit/impl/mlir/ir/each_mode.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megdnn/opr_param_defs.h"

#include <mlir/IR/Builders.h>

// clang-format off
#define MLIR_MGB_FOREACH_ELEMWISE_MODE_UNARY(cb) \
    cb(AbsOp, ABS) \
    cb(AcosOp, ACOS) \
    cb(AsinOp, ASIN) \
    cb(CeilOp, CEIL) \
    cb(CosOp, COS) \
    cb(ErfCInvOp, ERFCINV) \
    cb(ErfCOp, ERFC) \
    cb(ErfInvOp, ERFINV) \
    cb(ErfOp, ERF) \
    cb(ExpM1Op, EXPM1) \
    cb(ExpOp, EXP) \
    cb(FastTanhOp, FAST_TANH) \
    cb(FloorOp, FLOOR) \
    cb(HswishOp, H_SWISH) \
    cb(Log1POp, LOG1P) \
    cb(LogOp, LOG) \
    cb(NegOp, NEGATE) \
    cb(ReluOp, RELU) \
    cb(RoundOp, ROUND) \
    cb(SigmoidOp, SIGMOID) \
    cb(SinOp, SIN) \
    cb(TanhOp, TANH)

#define MLIR_MGB_FOREACH_ELEMWISE_MODE_BINARY(cb) \
    cb(AbsGradOp, ABS_GRAD) \
    cb(AddOp, ADD) \
    cb(Atan2Op, ATAN2) \
    cb(EqOp, EQ) \
    cb(FastTanhGradOp, FAST_TANH_GRAD) \
    cb(FloorDivOp, FLOOR_DIV) \
    cb(FuseAddHswishOp, FUSE_ADD_H_SWISH) \
    cb(FuseAddReluOp, FUSE_ADD_RELU) \
    cb(FuseAddSigmoidOp, FUSE_ADD_SIGMOID) \
    cb(FuseAddTanhOp, FUSE_ADD_TANH) \
    cb(HswishGradOp, H_SWISH_GRAD) \
    cb(LeqOp, LEQ) \
    cb(LogSumExpOp, LOG_SUM_EXP) \
    cb(LtOp, LT) \
    cb(MaxOp, MAX) \
    cb(MinOp, MIN) \
    cb(ModOp, MOD) \
    cb(MulOp, MUL) \
    cb(PowOp, POW) \
    cb(SigmoidGradOp, SIGMOID_GRAD) \
    cb(SubOp, SUB) \
    cb(SwishGt0Op, SWITCH_GT0) \
    cb(TanhGradOp, TANH_GRAD) \
    cb(TrueDivOp, TRUE_DIV)

#define MLIR_MGB_FOREACH_ELEMWISE_MODE_TERNARY(cb) \
    cb(CondLeqMovOp, COND_LEQ_MOV) \
    cb(FuseMulAdd3Op, FUSE_MUL_ADD3)
// clang-format on

namespace mgb {
namespace jit {

mlir::Value lower_elemwise_to_std(mlir::Operation* op,
                                  mlir::OpBuilder& builder,
                                  mlir::Location loc,
                                  mlir::ValueRange operands);

mlir::Value lower_typecvt_to_std(mlir::Operation* op,
                                 mlir::OpBuilder& builder,
                                 mlir::Location loc,
                                 mlir::Value input);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
