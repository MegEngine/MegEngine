/**
 * \file dnn/src/common/argmxx_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megdnn/dtype.h"

#if MEGDNN_CC_HOST
#include "megdnn/basic_types.h"
#endif

namespace megdnn {
namespace argmxx {

template <typename stype_, bool is_max>
struct ArgmxxOp {
    struct wtype {
        stype_ key;
        dt_int32 val;
        MEGDNN_HOST MEGDNN_DEVICE wtype()
        {}
        MEGDNN_HOST MEGDNN_DEVICE wtype(stype_ key, dt_int32 val):
            key(key), val(val)
        {}
        MEGDNN_HOST MEGDNN_DEVICE wtype(wtype &rhs):
            key(rhs.key),
            val(rhs.val)
        {}
        MEGDNN_HOST MEGDNN_DEVICE wtype(volatile wtype &rhs):
            key(rhs.key),
            val(rhs.val)
        {}
        MEGDNN_HOST MEGDNN_DEVICE wtype(const wtype &rhs):
            key(rhs.key),
            val(rhs.val)
        {}
        MEGDNN_HOST MEGDNN_DEVICE wtype(const volatile wtype &rhs):
            key(rhs.key),
            val(rhs.val)
        {}
        MEGDNN_HOST MEGDNN_DEVICE volatile wtype &operator=(const wtype &rhs) volatile
        {
            this->key = rhs.key;
            this->val = rhs.val;
            return *this;
        }
    };
    MEGDNN_HOST MEGDNN_DEVICE
    ArgmxxOp(stype_ *src, dt_int32 *dst, uint32_t A, uint32_t B, uint32_t C):
        src(src), dst(dst), A(A), B(B), C(C),
        INIT(wtype(is_max ? DTypeTrait<stype_>::min() :
                    DTypeTrait<stype_>::max(), 0))
    {
    }
    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx)
    {
        wtype res;
        res.key = src[idx];
        res.val = idx / C % B;
        return res;
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val)
    {
        dst[idx] = val.val;
    }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs)
    {
        if (is_max) {
            if (lhs.key > rhs.key) return lhs; else return rhs;
        } else {
            if (lhs.key < rhs.key) return lhs; else return rhs;
        }
    }
    stype_ *src;
    dt_int32 *dst;
    uint32_t A, B, C;
    const wtype INIT;
};

} // namespace argmxx
} // namespace megdnn
// vim: syntax=cpp.doxygen
