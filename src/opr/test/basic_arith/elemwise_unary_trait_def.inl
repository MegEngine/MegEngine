/**
 * \file src/opr/test/basic_arith/elemwise_unary_trait_def.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef DEF_TRAIT
#error "DEF_TRAIT must be defined"
#endif

/* ======================= unary ======================= */
#define _CUR_ARITY 1
#define _EXPAND_PARAMS \
    ctype x = inp[0][idx]

#define _ALLOW_BOOL true
#define _ALLOW_FLOAT false
#define _ALLOW_INT false
DEF_TRAIT(NOT, !x)
#undef _ALLOW_INT
#undef _ALLOW_FLOAT
#undef _ALLOW_BOOL

#define _ALLOW_BOOL false

#define _ALLOW_FLOAT true

#define _ALLOW_INT true
DEF_TRAIT(ABS, std::abs(x))
DEF_TRAIT(NEGATE, -x)
DEF_TRAIT(RELU, std::max<ctype>(x, 0))
#undef _ALLOW_INT

#define _ALLOW_INT false
DEF_TRAIT(ACOS, std::acos(x))
DEF_TRAIT(ASIN, std::asin(x))
DEF_TRAIT(CEIL, std::ceil(x))
DEF_TRAIT(COS, std::cos(x))
DEF_TRAIT(EXP, std::exp(x))
DEF_TRAIT(EXPM1, std::expm1(x))
DEF_TRAIT(FLOOR, std::floor(x))
DEF_TRAIT(LOG, std::log(x))
DEF_TRAIT(LOG1P, std::log1p(x))
DEF_TRAIT(SIGMOID, 1 / (1 + std::exp(-x)))
DEF_TRAIT(SIN, std::sin(x))
DEF_TRAIT(TANH, std::tanh(x))
DEF_TRAIT(FAST_TANH, do_fast_tanh(x))
DEF_TRAIT(ROUND, std::round(x))
DEF_TRAIT(ERF, std::erf(x))
DEF_TRAIT(ERFINV, do_erfinv(x))
DEF_TRAIT(ERFC, std::erfc(x))
DEF_TRAIT(ERFCINV, do_erfcinv(x))
DEF_TRAIT(H_SWISH, do_h_swish(x))
#undef _ALLOW_INT

#undef _ALLOW_FLOAT

#undef _ALLOW_BOOL

#undef _CUR_ARITY
#undef _EXPAND_PARAMS

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
