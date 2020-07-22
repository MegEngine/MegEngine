/**
 * \file dnn/src/common/cond_take/predicate.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/common/opr_param_defs_enumv.cuh"
#include "megdnn/arch.h"

#if MEGDNN_CC_HOST
#include "megdnn/opr_param_defs.h"
#endif

#ifndef __device__
#define __device__
#define __host__
#define def_device
#endif

#include <cmath>

namespace megdnn {
namespace cond_take {
    typedef param_enumv::CondTake::Mode PEnum;

    struct KParam {
        float val, eps;
#if MEGDNN_CC_HOST
        KParam(const param::CondTake &p):
            val(p.val), eps(p.eps)
        {}
#endif
    };

    template<uint32_t mode, typename ctype>
    struct Pred;

#define do_inst_eq_f(_ct) \
    template<> \
    struct Pred<PEnum::EQ, _ct> { \
        typedef _ct ctype; \
        ctype val, eps; \
        Pred(const KParam &p): val(p.val), eps(p.eps) {} \
        __device__ __host__ bool operator() (ctype x) const { \
            return fabsf(val - x) < eps; \
        } \
    };

#define do_inst_eq_i(_ct) \
    template<> \
    struct Pred<PEnum::EQ, _ct> { \
        typedef _ct ctype; \
        ctype val; \
        Pred(const KParam &p): val(p.val) {} \
        __device__ __host__ bool operator() (ctype x) const { \
            return val == x; \
        } \
    };

#define inst_eq_f(_dt) do_inst_eq_f(DTypeTrait<_dt>::ctype)
#define inst_eq_i(_dt) do_inst_eq_i(DTypeTrait<_dt>::ctype)
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(inst_eq_f)
    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(inst_eq_i)
    inst_eq_i(::megdnn::dtype::Bool)
#undef inst_eq_f
#undef inst_eq_i

    template<typename ctype_>
    struct Pred<PEnum::NEQ, ctype_> {
        typedef ctype_ ctype;
        Pred<PEnum::EQ, ctype> eq;

        Pred(const KParam &p): eq(p) {}

        __device__ __host__ bool operator() (ctype x) const {
            return !this->eq(x);
        }
    };

#define DEF_OP(_name, _op) \
    template<typename ctype_> \
    struct Pred<PEnum::_name, ctype_> { \
        typedef ctype_ ctype; \
        ctype val; \
        Pred(const KParam &p): val(p.val) {} \
        __device__ __host__ bool operator() (ctype x) const { \
            return x _op val; \
        } \
    }

    DEF_OP(LT, < );
    DEF_OP(LEQ, <= );
    DEF_OP(GT, > );
    DEF_OP(GEQ, >= );

#undef DEF_OP

#define MEGDNN_FOREACH_COND_TAKE_MODE(cb) \
    cb(EQ) cb(NEQ) cb(LT) cb(LEQ) cb(GT) cb(GEQ)

} // namespace cond_take
} // namespace megdnn

#ifdef def_device
#undef __device__
#undef __host__
#endif

// vim: ft=cpp syntax=cpp.doxygen
