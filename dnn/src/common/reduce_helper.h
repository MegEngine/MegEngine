/**
 * \file dnn/src/common/reduce_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/dtype.h"

#include "megdnn/basic_types.h"

namespace megdnn {
namespace reduce {

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct SumOp {
    typedef wtype_ wtype;

    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) { return lhs + rhs; }
    SumOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MeanOp {
    typedef wtype_ wtype;

    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) {
        dst.ptr<dst_ctype>()[idx] = val / static_cast<wtype>(B);
    }
    static wtype apply(wtype lhs, wtype rhs) { return lhs + rhs; }
    MeanOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct SumSqrOp {
    typedef wtype_ wtype;

    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) {
        return static_cast<wtype>(src.ptr<src_ctype>()[idx]) *
               static_cast<wtype>(src.ptr<src_ctype>()[idx]);
    }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) { return lhs + rhs; }
    SumSqrOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct ProdOp {
    typedef wtype_ wtype;
    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) { return lhs * rhs; }
    ProdOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(1)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MinOp {
    typedef wtype_ wtype;
    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) { return std::min(lhs, rhs); }
    MinOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::max())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype>
struct MinOp<src_ctype, dst_ctype, dt_float32> {
    typedef dt_float32 wtype;
    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) {
        return (std::isnan(lhs) || lhs < rhs) ? lhs : rhs;
    }
    MinOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::max())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MaxOp {
    typedef wtype_ wtype;
    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) { return std::max(lhs, rhs); }
    MaxOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::min())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype>
struct MaxOp<src_ctype, dst_ctype, dt_float32> {
    typedef dt_float32 wtype;
    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return src.ptr<src_ctype>()[idx]; }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) {
        return (std::isnan(lhs) || lhs > rhs) ? lhs : rhs;
    }
    MaxOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::min())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct CheckNonFiniteOp {
    typedef wtype_ wtype;
    const wtype INIT;

    RefPtr src;
    RefPtr dst;
    const size_t B;

    wtype read(uint32_t idx) { return !std::isfinite(src.ptr<src_ctype>()[idx]); }
    void write(uint32_t idx, wtype val) { dst.ptr<dst_ctype>()[idx] = val; }
    static wtype apply(wtype lhs, wtype rhs) { return lhs | rhs; }
    MEGDNN_HOST MEGDNN_DEVICE
    CheckNonFiniteOp(const RefPtr& src, const RefPtr& dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

void get_ABC(const TensorShape& shape, size_t& A, size_t& B, size_t& C, size_t axis);

}  // namespace reduce
}  // namespace megdnn
// vim: syntax=cpp.doxygen
