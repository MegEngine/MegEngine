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

#include "megbrain/jit/mlir/ir/dialect.h"

#include "./common.h"
#include "./numerical.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

// clang-format off
#define MLIR_MGB_FOREACH_ELEMWISE_MODE_UNARY(cb) \
    cb(ReluOp, RELU) \
    cb(AbsOp, ABS) \
    cb(NegOp, NEGATE) \
    cb(AcosOp, ACOS) \
    cb(AsinOp, ASIN) \
    cb(CeilOp, CEIL) \
    cb(CosOp, COS) \
    cb(ExpOp, EXP) \
    cb(FloorOp, FLOOR) \
    cb(LogOp, LOG) \
    cb(Log1POp, LOG1P) \
    cb(SigmoidOp, SIGMOID) \
    cb(SinOp, SIN) \
    cb(TanhOp, TANH) \
    cb(FastTanhOp, FAST_TANH) \
    cb(HswishOp, H_SWISH) \
    cb(ExpM1Op, EXPM1) \
    cb(RoundOp, ROUND) \
    cb(ErfOp, ERF) \
    cb(ErfInvOp, ERFINV) \
    cb(ErfCOp, ERFC) \
    cb(ErfCInvOp, ERFCINV)

#define MLIR_MGB_FOREACH_ELEMWISE_MODE_BINARY(cb) \
    cb(AbsGradOp, ABS_GRAD) \
    cb(AddOp, ADD) \
    cb(FloorDivOp, FLOOR_DIV) \
    cb(MaxOp, MAX) \
    cb(MinOp, MIN) \
    cb(ModOp, MOD) \
    cb(SubOp, SUB) \
    cb(MulOp, MUL) \
    cb(TrueDivOp, TRUE_DIV) \
    cb(PowOp, POW) \
    cb(SigmoidGradOp, SIGMOID_GRAD) \
    cb(SwishGt0Op, SWITCH_GT0) \
    cb(TanhGradOp, TANH_GRAD) \
    cb(LtOp, LT) \
    cb(LeqOp, LEQ) \
    cb(EqOp, EQ) \
    cb(FuseAddReluOp, FUSE_ADD_RELU) \
    cb(LogSumExpOp, LOG_SUM_EXP) \
    cb(FuseAddTanhOp, FUSE_ADD_TANH) \
    cb(FastTanhGradOp, FAST_TANH_GRAD) \
    cb(FuseAddSigmoidOp, FUSE_ADD_SIGMOID) \
    cb(HswishGradOp, H_SWISH_GRAD) \
    cb(FuseAddHswishOp, FUSE_ADD_H_SWISH) \
    cb(Atan2Op, ATAN2)

#define MLIR_MGB_FOREACH_ELEMWISE_MODE_TERNARY(cb) \
    cb(CondLeqMovOp, COND_LEQ_MOV) \
    cb(FuseMulAdd3Op, FUSE_MUL_ADD3)
// clang-format on

namespace mgb {
namespace jit {

template <typename mgb_op>
struct StandardOp;

#define cb(mgb_op, fun)                                                      \
    template <>                                                              \
    struct StandardOp<jit::mgb_op> {                                         \
        mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc, \
                               ValueRange operands) {                        \
            ValueBuilderHelper helper(builder, loc);                         \
            return helper.fun(operands);                                     \
        }                                                                    \
    }

//! unary
cb(AbsOp, abs);
cb(NegOp, neg);
cb(ExpOp, exp);
cb(CosOp, cos);
cb(CeilOp, ceil);
cb(FloorOp, floor);
cb(LogOp, log);
cb(SinOp, sin);
cb(TanhOp, tanh);

//! binary
cb(AddOp, add);
cb(MaxOp, max);
cb(MinOp, min);
cb(SubOp, sub);
cb(MulOp, mul);
cb(ModOp, mod);
cb(TrueDivOp, div);

#undef cb

/////////////////////////// unary op ///////////////////////////
//! max(x, 0)
template <>
struct StandardOp<jit::ReluOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.max(operands[0], helper.const_val(0.f));
    }
};

//! x * (27.f + x * x) / (27.f + 9.f * x * x);
template <>
struct StandardOp<jit::FastTanhOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto square = helper.mul(operands[0], operands[0]);
        return helper.div(
                helper.mul(operands[0],
                           helper.add(helper.const_val(27.f), square)),
                helper.add(helper.const_val(27.f),
                           helper.mul(helper.const_val(9.f), square)));
    }
};

//! x * clip(x + 3, 0, 6) / 6
template <>
struct StandardOp<jit::HswishOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);

        auto const_3 = helper.const_val(3.f);
        auto const_0 = helper.const_val(0.f);
        auto const_6 = helper.const_val(6.f);
        auto tmp = helper.add(operands[0], const_3);
        return helper.div(
                helper.mul(operands[0],
                           helper.min(helper.max(tmp, const_0), const_6)),
                const_6);
    }
};

//! log(1 + p)
template <>
struct StandardOp<jit::Log1POp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.log(helper.add(operands[0], helper.const_val(1.f)));
    }
};

//! 1.f / (expf(-y) + 1.f))
template <>
struct StandardOp<jit::SigmoidOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.div(helper.const_val(1.f),
                          helper.add(helper.exp(helper.neg(operands[0])),
                                     helper.const_val(1.f)));
    }
};

//! exp(x) - 1
template <>
struct StandardOp<jit::ExpM1Op> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.sub(helper.exp(operands[0]), helper.const_val(1.f));
    }
};

template <>
struct StandardOp<jit::RoundOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.select(
                helper.gt(operands[0], helper.const_val(0.f)),
                helper.floor(helper.add(operands[0], helper.const_val(0.5f))),
                helper.ceil(helper.sub(operands[0], helper.const_val(0.5f))));
    }
};

//! pi / 2 - arctan2(x, sqrt(1 - x * x))
template <>
struct StandardOp<jit::AcosOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto x = operands[0];
        auto one_minus_x_2 = helper.sub(helper.const_val(1.f), helper.mul(x, x));
        auto asin = atan2_approx(helper, x, helper.sqrt(one_minus_x_2));
        auto pi_over_2 = helper.const_val(1.57079637f);
        return helper.sub(pi_over_2, asin);
    }
};

//! arctan2(x, sqrt(1 - x * x))
template <>
struct StandardOp<jit::AsinOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto x = operands[0];
        auto one_minus_x_2 = helper.sub(helper.const_val(1.f), helper.mul(x, x));
        return atan2_approx(helper, x, helper.sqrt(one_minus_x_2));
    }
};

//! gauss error function
template <>
struct StandardOp<jit::ErfOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return erf_approx(helper, operands[0]);
    }
};

//! inverse of gauss error function
//! https://github.com/scipy/scipy/blob/master/scipy/special/cephes/erfinv.c
template <>
struct StandardOp<jit::ErfInvOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto sqrt2 = helper.const_val(1.4142135623f);
        auto x = helper.mul(helper.const_val(0.5f),
                            helper.add(operands[0], helper.const_val(1.f)));
        return helper.div(ndtri_approx(helper, x), sqrt2);
    }
};

//! complementary error function
template <>
struct StandardOp<jit::ErfCOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.sub(helper.const_val(1.f), erf_approx(helper, operands[0]));
    }
};

//! inverse of complementary gauss error function
//! https://github.com/scipy/scipy/blob/master/scipy/special/cephes/erfinv.c
template <>
struct StandardOp<jit::ErfCInvOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto minus_sqrt2 = helper.const_val(-1.4142135623f);
        auto x = helper.mul(helper.const_val(0.5f), operands[0]);
        return helper.div(ndtri_approx(helper, x), minus_sqrt2);
    }
};

/////////////////////////// binary op ///////////////////////////

//! binary: x > 0 ? y : -y
template <>
struct StandardOp<jit::AbsGradOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.select(helper.gt(operands[0], helper.const_val(0.f)),
                             operands[1], helper.neg(operands[1]));
    }
};

//! x^y = exp(y * log(x))
template <>
struct StandardOp<jit::PowOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.exp(helper.mul(operands[1], helper.log(operands[0])));
    }
};

//! x * (1 - x) * y
template <>
struct StandardOp<jit::SigmoidGradOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.mul(
                helper.mul(operands[0],
                           helper.sub(helper.const_val(1.f), operands[0])),
                operands[1]);
    }
};

//! (x > 0) * y
template <>
struct StandardOp<jit::SwishGt0Op> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.select(helper.gt(operands[0], helper.const_val(0.f)),
                             operands[1], helper.const_val(0.f));
    }
};

//! (1 - x * x) * y
template <>
struct StandardOp<jit::TanhGradOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.mul(helper.sub(helper.const_val(1.0f),
                                     helper.mul(operands[0], operands[0])),
                          operands[1]);
    }
};

#define cb(op, fun)                                                          \
    template <>                                                              \
    struct StandardOp<jit::op> {                                             \
        mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc, \
                               ValueRange operands) {                        \
            ValueBuilderHelper helper(builder, loc);                         \
            return helper.select(helper.fun(operands[0], operands[1]),       \
                                 helper.const_val(1.f),                      \
                                 helper.const_val(0.f));                     \
        }                                                                    \
    }

cb(LtOp, lt);
cb(LeqOp, le);
cb(EqOp, eq);
#undef cb

//! (x + y) <= ctype(0) ? ctype(0) : (x + y)
template <>
struct StandardOp<jit::FuseAddReluOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto sum = helper.add(operands[0], operands[1]);
        return helper.max(sum, helper.const_val(0.f));
    }
};

//! log(exp(x) + exp(y))
template <>
struct StandardOp<jit::LogSumExpOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.log(
                helper.add(helper.exp(operands[0]), helper.exp(operands[1])));
    }
};

//! floor(x/y)
template <>
struct StandardOp<jit::FloorDivOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.floor(helper.div(operands[0], operands[1]));
    }
};

//! tanh(x + y)
template <>
struct StandardOp<jit::FuseAddTanhOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.tanh(helper.add(operands[0], operands[1]));
    }
};

//! ((-48.f * x * x) / (3.f + x * x) + 27.f + x * x) / (3.f + x * x) * y
template <>
struct StandardOp<jit::FastTanhGradOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto x_pow2 = helper.mul(operands[0], operands[0]);
        auto deno = helper.add(helper.const_val(3.f), x_pow2);
        return helper.mul(
                helper.div(
                        helper.add(
                                helper.add(
                                        helper.div(helper.mul(helper.const_val(
                                                                      -48.f),
                                                              x_pow2),
                                                   deno),
                                        helper.const_val(27.f)),
                                x_pow2),
                        helper.mul(deno, helper.const_val(9.f))),
                operands[1]);
    }
};

//!  1.f / (expf(-(x+y)) + 1.f))
template <>
struct StandardOp<jit::FuseAddSigmoidOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.div(helper.const_val(1.f),
                          helper.add(helper.exp(helper.neg(helper.add(
                                             operands[0], operands[1]))),
                                     helper.const_val(1.f)));
    }
};

//! x < -3.f ? 0.f : (x > 3.f ? y : (2.f * x + 3.f) / 6.f * y)
template <>
struct StandardOp<jit::HswishGradOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.select(
                helper.lt(operands[0], helper.const_val(-3.f)),
                helper.const_val(0.f),
                helper.select(
                        helper.gt(operands[0], helper.const_val(3.f)),
                        operands[1],
                        helper.mul(
                                helper.div(
                                        helper.add(helper.mul(helper.const_val(
                                                                      2.f),
                                                              operands[0]),
                                                   helper.const_val(3.f)),
                                        helper.const_val(6.f)),
                                operands[1])));
    }
};

//! (x+y) * min(max(x + y + 3, 0), 6) * (1/6)
template <>
struct StandardOp<jit::FuseAddHswishOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        auto sum = helper.add(operands[0], operands[1]);

        auto const_3 = helper.const_val(3.f);
        auto const_0 = helper.const_val(0.f);
        auto const_6 = helper.const_val(6.f);
        auto tmp = helper.add(sum, const_3);
        return helper.div(
                helper.mul(sum, helper.min(helper.max(tmp, const_0), const_6)),
                const_6);
    }
};

//! arctan
template <>
struct StandardOp<jit::Atan2Op> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return atan2_approx(helper, operands[0], operands[1]);
    }
};

/////////////////////////// ternary op ///////////////////////////
//! x <= y ? z : ctype(0)
template <>
struct StandardOp<jit::CondLeqMovOp> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.select(helper.le(operands[0], operands[1]), operands[2],
                             helper.const_val(0.f));
    }
};

//!  x * y + z
template <>
struct StandardOp<jit::FuseMulAdd3Op> {
    mlir::Value operator()(mlir::OpBuilder& builder, mlir::Location loc,
                           ValueRange operands) {
        ValueBuilderHelper helper(builder, loc);
        return helper.add(helper.mul(operands[0], operands[1]), operands[2]);
    }
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
