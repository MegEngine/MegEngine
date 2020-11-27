/**
 * \file src/jit/impl/mlir/ir/each_mode.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "./common.h"
#include "./each_mode.h"
#include "./numerical.h"
#include "./types.h"

#include "megbrain/common.h"
#include "megbrain/exception.h"
#include "megbrain/jit/mlir/ir/dialect.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

namespace mgb {
namespace jit {

using Mode = megdnn::param::Elemwise::Mode;

template <Mode mode>
mlir::Value lower_mode(mlir::OpBuilder& builder, mlir::Location loc,
                       ValueRange operands);

/* ===================== trivial implementations ===================== */

#define cb(mode, fun)                                             \
    template <>                                                   \
    mlir::Value lower_mode<Mode::mode>(mlir::OpBuilder & builder, \
                                       mlir::Location loc,        \
                                       ValueRange operands) {     \
        ValueBuilderHelper helper(builder, loc);                  \
        return helper.fun(operands);                              \
    }

//! unary
cb(ABS, abs);
cb(CEIL, ceil);
cb(COS, cos);
cb(EXP, exp);
cb(FLOOR, floor);
cb(LOG, log);
cb(NEGATE, neg);
cb(SIN, sin);
cb(TANH, tanh);

//! binary
cb(ADD, add);
cb(MAX, max);
cb(MIN, min);
cb(MOD, mod);
cb(MUL, mul);
cb(SUB, sub);
cb(TRUE_DIV, div);

#undef cb

/* ===================== unary op ===================== */

//! ACOS: pi / 2 - arctan2(x, sqrt(1 - x * x))
template <>
mlir::Value lower_mode<Mode::ACOS>(mlir::OpBuilder& builder, mlir::Location loc,
                                   ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto x = operands[0];
    auto one_minus_x_2 = helper.sub(helper.const_f32(1.f), helper.mul(x, x));
    auto asin = atan2_approx(helper, x, helper.sqrt(one_minus_x_2));
    auto pi_over_2 = helper.const_f32(1.57079637f);
    return helper.sub(pi_over_2, asin);
}

//! ASIN: arctan2(x, sqrt(1 - x * x))
template <>
mlir::Value lower_mode<Mode::ASIN>(mlir::OpBuilder& builder, mlir::Location loc,
                                   ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto x = operands[0];
    auto one_minus_x_2 = helper.sub(helper.const_f32(1.f), helper.mul(x, x));
    return atan2_approx(helper, x, helper.sqrt(one_minus_x_2));
}

//! ERFCINV: inverse of complementary gauss error function
//! https://github.com/scipy/scipy/blob/master/scipy/special/cephes/erfinv.c
template <>
mlir::Value lower_mode<Mode::ERFCINV>(mlir::OpBuilder& builder,
                                      mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto minus_sqrt2 = helper.const_f32(-1.4142135623f);
    auto x = helper.mul(helper.const_f32(0.5f), operands[0]);
    return helper.div(ndtri_approx(helper, x), minus_sqrt2);
}

//! ERFC: complementary error function
template <>
mlir::Value lower_mode<Mode::ERFC>(mlir::OpBuilder& builder, mlir::Location loc,
                                   ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.sub(helper.const_f32(1.f), erf_approx(helper, operands[0]));
}

//! ERFINV: inverse of gauss error function
//! https://github.com/scipy/scipy/blob/master/scipy/special/cephes/erfinv.c
template <>
mlir::Value lower_mode<Mode::ERFINV>(mlir::OpBuilder& builder,
                                     mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto sqrt2 = helper.const_f32(1.4142135623f);
    auto x = helper.mul(helper.const_f32(0.5f),
                        helper.add(operands[0], helper.const_f32(1.f)));
    return helper.div(ndtri_approx(helper, x), sqrt2);
}

//! ERF: gauss error function
template <>
mlir::Value lower_mode<Mode::ERF>(mlir::OpBuilder& builder, mlir::Location loc,
                                  ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return erf_approx(helper, operands[0]);
}

//! EXPM1: exp(x) - 1
template <>
mlir::Value lower_mode<Mode::EXPM1>(mlir::OpBuilder& builder,
                                    mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.sub(helper.exp(operands[0]), helper.const_f32(1.f));
}

//! FAST_TANH: x * (27.f + x * x) / (27.f + 9.f * x * x);
template <>
mlir::Value lower_mode<Mode::FAST_TANH>(mlir::OpBuilder& builder,
                                        mlir::Location loc,
                                        ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto square = helper.mul(operands[0], operands[0]);
    return helper.div(
            helper.mul(operands[0], helper.add(helper.const_f32(27.f), square)),
            helper.add(helper.const_f32(27.f),
                       helper.mul(helper.const_f32(9.f), square)));
}

//! H_SWISH: x * clip(x + 3, 0, 6) / 6
template <>
mlir::Value lower_mode<Mode::H_SWISH>(mlir::OpBuilder& builder,
                                      mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);

    auto const_3 = helper.const_f32(3.f);
    auto const_0 = helper.const_f32(0.f);
    auto const_6 = helper.const_f32(6.f);
    auto tmp = helper.add(operands[0], const_3);
    return helper.div(helper.mul(operands[0],
                                 helper.min(helper.max(tmp, const_0), const_6)),
                      const_6);
}

//! LOG1P: log(1 + p)
template <>
mlir::Value lower_mode<Mode::LOG1P>(mlir::OpBuilder& builder,
                                    mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.log(helper.add(operands[0], helper.const_f32(1.f)));
}

//! RELU: max(x, 0)
template <>
mlir::Value lower_mode<Mode::RELU>(mlir::OpBuilder& builder, mlir::Location loc,
                                   ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.max(operands[0], helper.const_f32(0.f));
}

//! ROUND
template <>
mlir::Value lower_mode<Mode::ROUND>(mlir::OpBuilder& builder,
                                    mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(
            helper.gt(operands[0], helper.const_f32(0.f)),
            helper.floor(helper.add(operands[0], helper.const_f32(0.5f))),
            helper.ceil(helper.sub(operands[0], helper.const_f32(0.5f))));
}

//! SIGMOID: 1.f / (expf(-y) + 1.f))
template <>
mlir::Value lower_mode<Mode::SIGMOID>(mlir::OpBuilder& builder,
                                      mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.div(helper.const_f32(1.f),
                      helper.add(helper.exp(helper.neg(operands[0])),
                                 helper.const_f32(1.f)));
}

/* ===================== binary op ===================== */

//! ABS_GRAD: x > 0 ? y : -y
template <>
mlir::Value lower_mode<Mode::ABS_GRAD>(mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(helper.gt(operands[0], helper.const_f32(0.f)),
                         operands[1], helper.neg(operands[1]));
}

//! ATAN2
template <>
mlir::Value lower_mode<Mode::ATAN2>(mlir::OpBuilder& builder,
                                    mlir::Location loc, ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return atan2_approx(helper, operands[0], operands[1]);
}

//! EQ: x == y ? 1 : 0
template <>
mlir::Value lower_mode<Mode::EQ>(mlir::OpBuilder& builder, mlir::Location loc,
                                 ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(helper.eq(operands[0], operands[1]),
                         helper.const_f32(1.f), helper.const_f32(0.f));
}

//! FAST_TANH_GRAD: ((-48.f * x * x) / (3.f + x * x) + 27.f + x * x) / (3.f + x
//! * x) * y
template <>
mlir::Value lower_mode<Mode::FAST_TANH_GRAD>(mlir::OpBuilder& builder,
                                             mlir::Location loc,
                                             ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto x_pow2 = helper.mul(operands[0], operands[0]);
    auto deno = helper.add(helper.const_f32(3.f), x_pow2);
    return helper.mul(
            helper.div(
                    helper.add(
                            helper.add(helper.div(helper.mul(helper.const_f32(
                                                                     -48.f),
                                                             x_pow2),
                                                  deno),
                                       helper.const_f32(27.f)),
                            x_pow2),
                    helper.mul(deno, helper.const_f32(9.f))),
            operands[1]);
}

//! FLOOR_DIV: floor(x/y)
template <>
mlir::Value lower_mode<Mode::FLOOR_DIV>(mlir::OpBuilder& builder,
                                        mlir::Location loc,
                                        ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.floor(helper.div(operands[0], operands[1]));
}

//! FUSE_ADD_H_SWISH: (x+y) * min(max(x + y + 3, 0), 6) * (1/6)
template <>
mlir::Value lower_mode<Mode::FUSE_ADD_H_SWISH>(mlir::OpBuilder& builder,
                                               mlir::Location loc,
                                               ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto sum = helper.add(operands[0], operands[1]);

    auto const_3 = helper.const_f32(3.f);
    auto const_0 = helper.const_f32(0.f);
    auto const_6 = helper.const_f32(6.f);
    auto tmp = helper.add(sum, const_3);
    return helper.div(
            helper.mul(sum, helper.min(helper.max(tmp, const_0), const_6)),
            const_6);
}

//! FUSE_ADD_RELU: (x + y) <= ctype(0) ? ctype(0) : (x + y)
template <>
mlir::Value lower_mode<Mode::FUSE_ADD_RELU>(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    auto sum = helper.add(operands[0], operands[1]);
    return helper.max(sum, helper.const_f32(0.f));
}

//! FUSE_ADD_SIGMOID: 1.f / (expf(-(x+y)) + 1.f))
template <>
mlir::Value lower_mode<Mode::FUSE_ADD_SIGMOID>(mlir::OpBuilder& builder,
                                               mlir::Location loc,
                                               ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.div(helper.const_f32(1.f),
                      helper.add(helper.exp(helper.neg(
                                         helper.add(operands[0], operands[1]))),
                                 helper.const_f32(1.f)));
}

//! FUSE_ADD_TANH: tanh(x + y)
template <>
mlir::Value lower_mode<Mode::FUSE_ADD_TANH>(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.tanh(helper.add(operands[0], operands[1]));
}

//! H_SWISH_GRAD: x < -3.f ? 0.f : (x > 3.f ? y : (2.f * x + 3.f) / 6.f * y)
template <>
mlir::Value lower_mode<Mode::H_SWISH_GRAD>(mlir::OpBuilder& builder,
                                           mlir::Location loc,
                                           ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(
            helper.lt(operands[0], helper.const_f32(-3.f)),
            helper.const_f32(0.f),
            helper.select(
                    helper.gt(operands[0], helper.const_f32(3.f)), operands[1],
                    helper.mul(
                            helper.div(
                                    helper.add(helper.mul(helper.const_f32(2.f),
                                                          operands[0]),
                                               helper.const_f32(3.f)),
                                    helper.const_f32(6.f)),
                            operands[1])));
}

//! LEQ: x <= y ? 1 : 0
template <>
mlir::Value lower_mode<Mode::LEQ>(mlir::OpBuilder& builder, mlir::Location loc,
                                  ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(helper.le(operands[0], operands[1]),
                         helper.const_f32(1.f), helper.const_f32(0.f));
}

//! LOG_SUM_EXP: log(exp(x) + exp(y))
template <>
mlir::Value lower_mode<Mode::LOG_SUM_EXP>(mlir::OpBuilder& builder,
                                          mlir::Location loc,
                                          ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.log(
            helper.add(helper.exp(operands[0]), helper.exp(operands[1])));
}

//! LT: x < y ? 1 : 0
template <>
mlir::Value lower_mode<Mode::LT>(mlir::OpBuilder& builder, mlir::Location loc,
                                 ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(helper.lt(operands[0], operands[1]),
                         helper.const_f32(1.f), helper.const_f32(0.f));
}

//! POW: x^y = exp(y * log(x))
template <>
mlir::Value lower_mode<Mode::POW>(mlir::OpBuilder& builder, mlir::Location loc,
                                  ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.exp(helper.mul(operands[1], helper.log(operands[0])));
}

//! SIGMOID_GRAD: x * (1 - x) * y
template <>
mlir::Value lower_mode<Mode::SIGMOID_GRAD>(mlir::OpBuilder& builder,
                                           mlir::Location loc,
                                           ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.mul(helper.mul(operands[0], helper.sub(helper.const_f32(1.f),
                                                         operands[0])),
                      operands[1]);
}

//! SWITCH_GT0: (x > 0) * y
template <>
mlir::Value lower_mode<Mode::SWITCH_GT0>(mlir::OpBuilder& builder,
                                         mlir::Location loc,
                                         ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(helper.gt(operands[0], helper.const_f32(0.f)),
                         operands[1], helper.const_f32(0.f));
}

//! TANH_GRAD: (1 - x * x) * y
template <>
mlir::Value lower_mode<Mode::TANH_GRAD>(mlir::OpBuilder& builder,
                                        mlir::Location loc,
                                        ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.mul(helper.sub(helper.const_f32(1.0f),
                                 helper.mul(operands[0], operands[0])),
                      operands[1]);
}

/* ===================== ternary op ===================== */

//! COND_LEQ_MOV: x <= y ? z : ctype(0)
template <>
mlir::Value lower_mode<Mode::COND_LEQ_MOV>(mlir::OpBuilder& builder,
                                           mlir::Location loc,
                                           ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.select(helper.le(operands[0], operands[1]), operands[2],
                         helper.const_f32(0.f));
}

//! FUSE_MUL_ADD3: x * y + z
template <>
mlir::Value lower_mode<Mode::FUSE_MUL_ADD3>(mlir::OpBuilder& builder,
                                            mlir::Location loc,
                                            ValueRange operands) {
    ValueBuilderHelper helper(builder, loc);
    return helper.add(helper.mul(operands[0], operands[1]), operands[2]);
}

/* ===================== elemwise ===================== */

mlir::Value lower_elemwise_to_std(mlir::Operation* op, mlir::OpBuilder& builder,
                                  mlir::Location loc, ValueRange operands) {
    auto mode = llvm::dyn_cast<dialect::Elemwise>(op).mode();
    switch (mode) {
#define cb(_, _mode)  \
    case Mode::_mode: \
        return lower_mode<Mode::_mode>(builder, loc, operands);
        MLIR_MGB_FOREACH_ELEMWISE_MODE_UNARY(cb);
        MLIR_MGB_FOREACH_ELEMWISE_MODE_BINARY(cb);
        MLIR_MGB_FOREACH_ELEMWISE_MODE_TERNARY(cb);
        default:
            return nullptr;
    }
#undef cb
}

/* ===================== typecvt ===================== */

mlir::Value lower_typecvt_to_std(mlir::Operation* op, mlir::OpBuilder& builder,
                                 mlir::Location loc, mlir::Value input) {
    auto&& typecvt = llvm::dyn_cast<dialect::TypeCvt>(op);
    mlir::Type idtype = typecvt.idtype();
    mlir::Type odtype =
            megdnn_dtype_to_mlir_type(typecvt.dtype(), builder.getContext());

    mlir::Type itype = input.getType();
    mlir::Type otype = signless(odtype);
    mgb_assert(signless(idtype) == itype);

    if (mlir::FPExtOp::areCastCompatible(itype, otype)) {
        return builder.create<mlir::FPExtOp>(loc, otype, input);
    } else if (mlir::FPTruncOp::areCastCompatible(itype, otype)) {
        return builder.create<mlir::FPTruncOp>(loc, otype, input);
    } else if (mlir::FPToSIOp::areCastCompatible(itype, otype) and
               odtype.isSignedInteger()) {
        return builder.create<mlir::FPToSIOp>(loc, otype, input);
    } else if (mlir::FPToUIOp::areCastCompatible(itype, otype) and
               odtype.isUnsignedInteger()) {
        return builder.create<mlir::FPToUIOp>(loc, otype, input);
    } else if (mlir::SIToFPOp::areCastCompatible(itype, otype) and
               idtype.isSignedInteger()) {
        return builder.create<mlir::SIToFPOp>(loc, otype, input);
    } else if (mlir::UIToFPOp::areCastCompatible(itype, otype) and
               idtype.isUnsignedInteger()) {
        return builder.create<mlir::UIToFPOp>(loc, otype, input);
    } else {
        std::string tmp;
        llvm::raw_string_ostream os(tmp);
        os << "cannot convert from " << idtype << " to " << odtype;
        mgb_throw_raw(InternalError{tmp});
    }

    return nullptr;
}

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
