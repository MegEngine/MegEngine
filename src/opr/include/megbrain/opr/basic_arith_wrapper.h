/**
 * \file src/opr/include/megbrain/opr/basic_arith_wrapper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/basic_arith.h"

namespace mgb {
namespace opr {

#define EL1(_name, _mode) \
    static inline SymbolVar _name(SymbolVar x, \
            const OperatorNodeConfig &config = {}) { \
        return Elemwise::make({x}, Elemwise::Mode::_mode, config); \
    }

    EL1(negate, NEGATE)
    EL1(relu, RELU)
    EL1(sigmoid, SIGMOID)
    EL1(tanh, TANH)
    EL1(hswish, H_SWISH)
    EL1(sin, SIN)
    EL1(cos, COS)
    EL1(exp, EXP)
    EL1(log, LOG)
    EL1(abs, ABS)

#undef EL1

#define EL2(_name, _mode) \
    static inline SymbolVar _name(SymbolVar x, SymbolVar y, \
            const OperatorNodeConfig &config = {}) { \
        return Elemwise::make({x, y}, Elemwise::Mode::_mode, config); \
    }

    EL2(add, ADD)
    EL2(sub, SUB)
    EL2(mul, MUL)
    EL2(div, TRUE_DIV)
    EL2(floor_div, FLOOR_DIV)
    EL2(pow, POW)
    EL2(less_than, LT)
    EL2(less_equal, LEQ)
    EL2(max, MAX)
    EL2(min, MIN)
    EL2(switch_gt0, SWITCH_GT0)
    EL2(eq, EQ)

#undef EL2

#define REDUCE(_name, _mode)                                                \
    static inline SymbolVar reduce_##_name(                                 \
            SymbolVar x, SymbolVar tshape,                                  \
            const OperatorNodeConfig& config = {}) {                        \
        return Reduce::make(x, {Reduce::Mode::_mode}, tshape, config);      \
    }                                                                       \
    static inline SymbolVar reduce_ax_##_name(                              \
            SymbolVar x, int axis, const OperatorNodeConfig& config = {}) { \
        return Reduce::make(x, {Reduce::Mode::_mode, axis}, {}, config);    \
    }

    REDUCE(sum, SUM);
    REDUCE(sum_sqr, SUM_SQR);
    REDUCE(prod, PRODUCT);
    REDUCE(min, MIN);
    REDUCE(max, MAX);

#undef REDUCE

    static inline SymbolVar powf(SymbolVar x, float a,
            const OperatorNodeConfig &config = {}) {
        // Note: Elemwise mode POW can only work on float values, which means if
        // the dtype of `x` is INT, all of the inputs would be converted to Float32
        // before exec (see Elemwise::make for more details).
        return opr::pow(x, x.make_scalar_dt(a), config);
    }

} // namespace opr
} // namespace mgb



// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

