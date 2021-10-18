/**
 * \file dnn/src/fallback/relayout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/relayout/opr_impl.h"
#include "src/common/relayout_helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

using namespace megdnn;
using namespace fallback;

namespace {

bool is_lastdim_contig(const TensorLayout& layout) {
    return layout.ndim <= 3 && layout.stride[layout.ndim - 1] == 1;
}

template <size_t sz, typename T0 = char>
struct equiv_ctype_storage {
    T0 _[sz];
};

template <typename dtype>
struct equiv_ctype {
    using type = std::aligned_storage_t<
            sizeof(typename DTypeTrait<dtype>::ctype),
            alignof(typename DTypeTrait<dtype>::ctype)>;
};

typedef void (*memcpy_policy_t)(void* cont, void* non_cont, size_t);

void memcpy_cont2noncont(void* cont, void* non_cont, size_t size) {
    memcpy(non_cont, cont, size);
}

void memcpy_noncont2cont(void* cont, void* non_cont, size_t size) {
    memcpy(cont, non_cont, size);
}

template <typename T>
void call_transpose(
        size_t batch, size_t m, size_t n, size_t ch, void* src, void* dst,
        size_t stride_m) {
    megdnn_assert(ch == 1);
    relayout::transpose_fallback::transpose<T>(
            batch, m, n, static_cast<T*>(src), static_cast<T*>(dst), stride_m);
}

//! one operand contiguous, and the other non-contiguous
template <typename ctype>
void dispatch_on_dtype_cont(
        Handle* handle, const TensorND& cont, const TensorND& nonc,
        memcpy_policy_t mcp_pol) {
    auto ctptr = static_cast<uint8_t*>(cont.raw_ptr),
         ncptr = static_cast<uint8_t*>(nonc.raw_ptr);
    thin_function<void()> kern;
    switch (nonc.layout.ndim) {
        case 2: {
            auto shp0 = nonc.layout.shape[0], shp1 = nonc.layout.shape[1];
            auto strd0_n = nonc.layout.stride[0] * sizeof(ctype);
            auto strd0_c = shp1 * sizeof(ctype);
            kern = [=]() {
                auto cur_ctptr = ctptr;
                auto cur_ncptr = ncptr;
                for (size_t i = 0; i < shp0; ++i) {
                    mcp_pol(cur_ctptr, cur_ncptr, strd0_c);
                    cur_ctptr += strd0_c;
                    cur_ncptr += strd0_n;
                }
            };
            break;
        }
        case 3: {
            auto shp0 = nonc.layout.shape[0], shp1 = nonc.layout.shape[1],
                 shp2 = nonc.layout.shape[2];
            auto strd0_n = nonc.layout.stride[0] * sizeof(ctype),
                 strd1_n = nonc.layout.stride[1] * sizeof(ctype);
            auto strd1_c = shp2 * sizeof(ctype);
            kern = [=]() {
                auto cur_ctptr = ctptr;
                auto ncptr_row = ncptr;
                for (size_t i = 0; i < shp0; ++i) {
                    auto cur_ncptr = ncptr_row;
                    for (size_t j = 0; j < shp1; ++j) {
                        mcp_pol(cur_ctptr, cur_ncptr, strd1_c);
                        cur_ctptr += strd1_c;
                        cur_ncptr += strd1_n;
                    }
                    ncptr_row += strd0_n;
                }
            };
            break;
        }
        default:
            megdnn_assert(0);
    }

    static_cast<naive::HandleImpl*>(handle)->dispatch_kern(std::move(kern));
}

void dispatch_cont(
        Handle* handle, const TensorND& cont, const TensorND& nonc,
        memcpy_policy_t mcp_pol) {
    switch (cont.layout.dtype.enumv()) {
#define cb(_dt)                                                       \
    case DTypeTrait<dtype::_dt>::enumv:                               \
        return dispatch_on_dtype_cont<equiv_ctype<dtype::_dt>::type>( \
                handle, cont, nonc, mcp_pol);
        MEGDNN_FOREACH_DTYPE_NAME(cb)
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
        megdnn_assert(0);
    }
}

const size_t BLOCK_SIZE = 16,
             TRANSPOSE_CV_MAX_C = relayout::transpose_fallback::BLOCK_LINE_SIZE_BYTES;

/*!
 * \tparam ctype The type of the data
 */
template <typename ctype>
void transpose_cv_block(
        size_t m, size_t n, size_t ch, size_t i, size_t j, size_t h, size_t w,
        void* src, void* dst) {
    auto batch_src = static_cast<const ctype*>(src);
    auto batch_dst = static_cast<ctype*>(dst);

#define SET_VAL(dst, src)                      \
    switch (ch) {                              \
        case 3:                                \
            dst[2] = src[2];                   \
            MEGDNN_FALLTHRU                    \
        case 2:                                \
            dst[1] = src[1];                   \
            MEGDNN_FALLTHRU                    \
        case 1:                                \
            dst[0] = src[0];                   \
            break;                             \
        default:                               \
            for (size_t _c = 0; _c < ch; ++_c) \
                dst[_c] = src[_c];             \
            break;                             \
    }

    constexpr size_t B = BLOCK_SIZE;
    static_assert(TRANSPOSE_CV_MAX_C % sizeof(ctype) == 0, "bad ctype");
    ctype tmp[B][B][TRANSPOSE_CV_MAX_C / sizeof(ctype)];
    auto sptr = batch_src + i * n * ch + j * ch;
    for (size_t x = 0; x < h; ++x) {
        for (size_t y = 0; y < w; ++y) {
            SET_VAL(tmp[y][x], (sptr + y * ch))
        }
        sptr += n * ch;
    }

    auto dptr = batch_dst + j * m * ch + i * ch;
    for (size_t x = 0; x < w; ++x) {
        for (size_t y = 0; y < h; ++y) {
            SET_VAL((dptr + y * ch), tmp[x][y])
        }
        dptr += m * ch;
    }
#undef SET_VAL
}

template <typename ctype>
void transpose_cv_row(
        size_t m, size_t n, size_t ch, size_t i, size_t h, void* src, void* dst) {
    constexpr size_t B = BLOCK_SIZE;
    size_t j = 0;
    for (; j + B <= n; j += B) {
        transpose_cv_block<ctype>(m, n, ch, i, j, h, B, src, dst);
    }
    if (j < n) {
        transpose_cv_block<ctype>(m, n, ch, i, j, h, n - j, src, dst);
    }
}

template <typename ctype>
void transpose_cv(
        size_t batch, size_t m, size_t n, size_t ch, void* src, void* dst,
        size_t stride_m) {
    megdnn_assert(stride_m == 0);
    constexpr size_t B = BLOCK_SIZE;
    auto batch_src = static_cast<ctype*>(src);
    auto batch_dst = static_cast<ctype*>(dst);
    for (size_t b = 0; b < batch; ++b) {
        size_t i = 0;
        for (; i + B <= m; i += B) {
            transpose_cv_row<ctype>(m, n, ch, i, B, batch_src, batch_dst);
        }
        if (i < m) {
            transpose_cv_row<ctype>(m, n, ch, i, m - i, batch_src, batch_dst);
        }
        batch_src += m * n * ch;
        batch_dst += m * n * ch;
    }
}

}  // anonymous namespace

void RelayoutForwardImpl::exec(
        _megdnn_tensor_in src0, _megdnn_tensor_out dst0, Handle* src_handle) {
    check_cpu_handle(src_handle);
    TensorND src = src0, dst = dst0;
    check_layout_and_canonize(src.layout, dst.layout);

    bool has_neg_stride = false;
    for (size_t i = 0; i < src.layout.ndim; ++i) {
        if (src.layout.stride[i] < 0) {
            has_neg_stride = true;
            break;
        }
    }
    for (size_t i = 0; i < dst.layout.ndim; ++i) {
        if (dst.layout.stride[i] < 0) {
            has_neg_stride = true;
            break;
        }
    }
    if (has_neg_stride) {
        NaiveRelayoutForwardImpl::do_exec(src, dst);
        return;
    }

    // FIXME: optimize for lowbit cases
    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
        src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        NaiveRelayoutForwardImpl::do_exec(src, dst);
        return;
    }

    relayout::TransposeParam trans_param;
    bool trans = relayout::is_transpose(src.layout, dst.layout, trans_param, true);
    exec_after_preprocess(src, dst, trans ? &trans_param : nullptr);
}

void RelayoutForwardImpl::exec_after_preprocess(
        const TensorND& src, const TensorND& dst, relayout::TransposeParam* transpose) {
    if (transpose) {
        auto dsize = src.layout.dtype.size() * transpose->c;
        void (*kptr)(size_t, size_t, size_t, size_t, void*, void*, size_t) = nullptr;
        auto src_addr = reinterpret_cast<uintptr_t>(src.raw_ptr),
             dst_addr = reinterpret_cast<uintptr_t>(dst.raw_ptr);
        if (dsize == 1) {
            megdnn_assert(transpose->c == 1);
            kptr = call_transpose<uint8_t>;
        } else if (dsize == 2) {
            transpose->c = 1;
            if (!((src_addr | dst_addr) & (alignof(uint16_t) - 1))) {
                kptr = call_transpose<uint16_t>;
            } else {
                kptr = call_transpose<equiv_ctype_storage<2>>;
                megdnn_log_error("unaligned addr in relayout");
            }
        } else if (dsize == 3) {
            transpose->c = 1;
            kptr = call_transpose<equiv_ctype_storage<3>>;
        } else if (dsize == 4) {
            transpose->c = 1;
            if (!((src_addr | dst_addr) & (alignof(uint32_t) - 1))) {
                kptr = call_transpose<uint32_t>;
            } else {
                kptr = call_transpose<equiv_ctype_storage<4>>;
                megdnn_log_error("unaligned addr in relayout");
            }
        } else if (dsize == 12) {
            transpose->c = 1;
            if (!((src_addr | dst_addr) & (alignof(uint32_t) - 1))) {
                kptr = call_transpose<equiv_ctype_storage<3, uint32_t>>;
            } else {
                kptr = call_transpose<equiv_ctype_storage<12>>;
                megdnn_log_error("unaligned addr in relayout");
            }
        } else if (dsize <= TRANSPOSE_CV_MAX_C) {
            switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                             \
    case DTypeTrait<dtype::_dt>::enumv:                     \
        kptr = transpose_cv<equiv_ctype<dtype::_dt>::type>; \
        break;
                MEGDNN_FOREACH_DTYPE_NAME(cb)
                MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
            }
            megdnn_assert(kptr);
        }

        if (kptr) {
            auto kern = [t = *transpose, sptr = src.raw_ptr, dptr = dst.raw_ptr,
                         kptr]() {
                kptr(t.batch, t.m, t.n, t.c, sptr, dptr, t.stride_m);
            };
            static_cast<naive::HandleImpl*>(handle())->dispatch_kern(kern);
            return;
        } else {
            megdnn_assert(transpose->c != 1, "unsupported dtype size");
        }
    }

    using relayout::is_contig;

    if (is_contig(dst.layout) && is_contig(src.layout)) {
        auto sptr = src.raw_ptr, dptr = dst.raw_ptr;
        auto sz = src.layout.span().dist_byte();
        MEGDNN_DISPATCH_CPU_KERN_OPR(memcpy(dptr, sptr, sz));
        return;
    }

    if (is_contig(dst.layout) && is_lastdim_contig(src.layout)) {
        return dispatch_cont(handle(), dst, src, memcpy_noncont2cont);
    }

    if (is_contig(src.layout) && is_lastdim_contig(dst.layout)) {
        return dispatch_cont(handle(), src, dst, memcpy_cont2noncont);
    }
    NaiveRelayoutForwardImpl::do_exec(src, dst);
}

// vim: syntax=cpp.doxygen
