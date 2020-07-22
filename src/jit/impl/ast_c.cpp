/**
 * \file src/jit/impl/ast_c.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/jit/ast_c.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/opr/tensor_manip.h"

#if MGB_JIT

using namespace mgb;
using namespace jit;
using namespace ast_c;

namespace {
ASTPtr gen_powc(ASTPtr inp, float exp) {
    auto int_neg = [exp](ASTPtr x) {
        if (exp < 0) {
            return 1.f / x;
        }
        return x;
    };
    if (almost_equal(std::abs(exp), 0.f)) {
        return 1.f;
    }
    if (almost_equal(std::abs(exp), 1.f)) {
        return int_neg(inp);
    }
    if (almost_equal(std::abs(exp), 2.f)) {
        return int_neg(inp * inp);
    }
    if (almost_equal(std::abs(exp), 3.f)) {
        return int_neg(inp * inp * inp);
    }
    if (almost_equal(exp, 1.f / 3.f)) {
        return make_call("cbrtf", {inp});
    }
    if (almost_equal(exp, -1.f / 3.f)) {
        return make_call("rcbrtf", {inp});
    }
    if (almost_equal(exp, .5f)) {
        return make_call("sqrtf", {inp});
    }
    if (almost_equal(exp, -.5f)) {
        return make_call("rsqrtf", {inp});
    }
    int exp_i = std::round(exp);
    if (almost_equal(static_cast<float>(exp_i), exp)) {
        auto inp_abs = make_call("fabsf", {inp});
        if (exp_i & 1) {
            auto pow = make_call("powf", {inp_abs, exp});
            return make_call("copysign", {pow, inp});
        } else {
            return make_call("powf", {inp_abs, exp});
        }
    }

    return make_call("powf", {inp, exp});
}
}  // anonymous namespace

const ElemGeneratorMap& ast_c::elem_opr_generator() {
#define ENTRY(_mode, _impl)                                                \
    {                                                                      \
        ElemMode::_mode, {                                                 \
            [](const ASTPtrArray& inps) -> ASTPtrArray { return {_impl}; } \
        }                                                                  \
    }
    static ElemGeneratorMap map = {
            // unary
            ENTRY(RELU, make_call("fmaxf", {inps[0], 0.f})),
            ENTRY(ABS, make_call("fabsf", inps)),
            ENTRY(ACOS, make_call("acosf", inps)),
            ENTRY(ASIN, make_call("asinf", inps)),
            ENTRY(CEIL, make_call("ceilf", inps)),
            ENTRY(COS, make_call("cosf", inps)),
            ENTRY(EXP, make_call("expf", inps)),
            ENTRY(EXPM1, make_call("expm1f", inps)),
            ENTRY(FLOOR, make_call("floorf", inps)),
            ENTRY(LOG, make_call("logf", inps)),
            ENTRY(LOG1P, make_call("log1pf", inps)),
            ENTRY(NEGATE, make_call("-", inps)),
            ENTRY(SIGMOID, 1 / (1 + make_call("expf", {0 - inps[0]}))),
            ENTRY(SIN, make_call("sinf", inps)),
            ENTRY(TANH, make_call("tanhf", inps)),
            ENTRY(ERF, make_call("erff", inps)),
            ENTRY(ERFC, make_call("erfcf", inps)),
            ENTRY(H_SWISH,
                  inps[0] *
                          make_call("fmaxf",
                                    {make_call("fminf", {inps[0] + 3.f, 6.f}),
                                     0.f}) /
                          6.f),

            // binary
            ENTRY(ABS_GRAD,
                  ASTPtr::make<Cond3AST>(inps[0] > 0, inps[1], -inps[1])),
            ENTRY(ADD, inps[0] + inps[1]),
            ENTRY(FLOOR_DIV, make_call("floorf", {inps[0] / inps[1]})),
            ENTRY(MAX, make_call("fmaxf", inps)),
            ENTRY(MIN, make_call("fminf", inps)),
            ENTRY(MOD, make_call("fmodf", inps)),
            ENTRY(MUL, inps[0] * inps[1]),
            ENTRY(POW, make_call("powf", inps)),
            ENTRY(SIGMOID_GRAD, inps[0] * (1 - inps[0]) * inps[1]),
            ENTRY(SUB, inps[0] - inps[1]),
            ENTRY(SWITCH_GT0, ASTPtr::make<Cond3AST>(inps[0] > 0, inps[1], 0)),
            ENTRY(TANH_GRAD, (1 - inps[0] * inps[0]) * inps[1]),
            ENTRY(TRUE_DIV, inps[0] / inps[1]),
            ENTRY(LOG_SUM_EXP,
                  make_call("mgb_log_sum_exp", {inps[0], inps[1]})),
            ENTRY(LT, ASTPtr::make<BinaryAST>("<", inps[0], inps[1])),
            ENTRY(LEQ, ASTPtr::make<BinaryAST>("<=", inps[0], inps[1])),
            ENTRY(EQ, ASTPtr::make<BinaryAST>("==", inps[0], inps[1])),
            ENTRY(ATAN2, make_call("atan2f", inps)),
            ENTRY(H_SWISH_GRAD,
                  ASTPtr::make<Cond3AST>(
                          -inps[0] > 3.f, 0.f,
                          ASTPtr::make<Cond3AST>(
                                  inps[0] > 3.f, inps[1],
                                  (2.f * inps[0] + 3.f) * inps[1] / 6.f))),

            // misc
            ENTRY(COND_LEQ_MOV,
                  ASTPtr::make<BinaryAST>("<=", inps[0], inps[1]) * inps[2]),
            ENTRY(FUSE_MUL_ADD3, inps[0] * inps[1] + inps[2]),
            ENTRY(FUSE_MUL_ADD4, inps[0] * inps[1] + inps[2] * inps[3]),
            ENTRY(FUSE_ADD_RELU, make_call("fmaxf", {inps[0] + inps[1], 0})),
            ENTRY(FUSE_ADD_SIGMOID,
                  1 / (1 + make_call("expf", {-(inps[0] + inps[1])}))),
            ENTRY(FUSE_ADD_TANH, make_call("tanhf", {inps[0] + inps[1]})),
            ENTRY(FUSE_ADD_H_SWISH,
                  (inps[0] + inps[1]) *
                          make_call(
                                  "fmaxf",
                                  {make_call("fminf",
                                             {(inps[0] + inps[1]) + 3.f, 6.f}),
                                   0.f}) /
                          6.f),
    };
    mgb_assert(map.size() + 12 == opr::Elemwise::Param::MODE_NR_MEMBER);
    // unimplemented modes: SHL, SHR, FAST_TANH, FAST_TANH_GRAD, ROUND, RMULH,
    // ERFINV, ERFCINV, NOT, AND, OR, XOR
    return map;
#undef ADD_OPR
}

ASTPtrArray ast_c::opr2AST(cg::OperatorNodeBase* opr,
                           const ASTPtrArray& inputs) {
    using namespace opr;
    if (auto elem = gopt::try_cast_as_op<Elemwise>(opr)) {
        if (check_elem_mode(elem->param().mode)) {
            return elem_opr_generator()
                    .find(elem->param().mode)
                    ->second(inputs);
        }
    }

    if (auto powc = gopt::try_cast_as_op<PowC>(opr)) {
        mgb_assert(inputs.size() == 1);
        return {gen_powc(inputs[0], powc->param().exp)};
    }

    auto imm = SymbolVar{opr->output(0)}.as_immutable_scalar();
    if (imm.valid()) {
        auto dtype = imm->dtype();
        if (dtype == dtype::Int32{}) {
            return {ASTPtr::make<IntAST>(imm->get<int>())};
        }
        float scalar_value;
        if (dtype == dtype::Float32()) {
            scalar_value = imm->get<float>();
        } else if (dtype == dtype::Float16()) {
            scalar_value = imm->get<dt_float16>();
        } else {
            mgb_throw(InternalError,
                      "dtype(%s) is not any of [Float16, Float32, Int32]",
                      dtype.name());
        }
        return {ASTPtr::make<FloatAST>(scalar_value)};
    }

    if (opr->same_type<opr::TypeCvt>()) {
        // simply ignore TypeCvt oprs.
        mgb_assert(inputs.size() == 1);
        return inputs;
    }

    mgb_throw(InternalError, "unknown opr %s{%s}", opr->cname(),
              opr->dyn_typeinfo()->name);
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
