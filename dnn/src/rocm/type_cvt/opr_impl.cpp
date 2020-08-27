/**
 * \file dnn/src/rocm/type_cvt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./opr_impl.h"
#include "src/rocm/type_cvt/type_cvt.h.hip"

#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

namespace {
template <typename T>
void exec_src_quantized(const TensorND& dst, const TensorND& src,
                        const DTypeParam<T>& src_param, hipStream_t stream) {
    bool is_dst_quantized =
            dst.layout.dtype.category() == DTypeCategory::QUANTIZED;
    using ctype_src = typename DTypeTrait<T>::ctype;
    if (!is_dst_quantized) {
        switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                               \
    case DTypeTrait<_dt>::enumv: {                                            \
        using ctype_dest = typename DTypeTrait<_dt>::ctype;                   \
        typecvt_kern_q2n<ctype_src, ctype_dest>(dst, src, src_param, stream); \
        return;                                                               \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
            default:
                megdnn_assert_internal(0);
#undef cb
        }
    } else {
        switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                      \
    case DTypeTrait<_dt>::enumv: {                                   \
        auto dst_param = dst.layout.dtype.param<_dt>();              \
        using ctype_dest = typename DTypeTrait<_dt>::ctype;          \
        typecvt_kern_q2q<ctype_src, ctype_dest>(dst, src, src_param, \
                                                dst_param, stream);  \
        return;                                                      \
    }
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb);
            default:
                megdnn_assert_internal(0);
#undef cb
        }
    }
}

template <typename T>
void exec_src_normal(const TensorND& dst, const TensorND& src,
                     hipStream_t stream) {
    bool is_dst_quantized =
            dst.layout.dtype.category() == DTypeCategory::QUANTIZED;
    using ctype_src = typename DTypeTrait<T>::ctype;
    if (!is_dst_quantized) {
        switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                    \
    case DTypeTrait<_dt>::enumv: {                                 \
        using ctype_dest = typename DTypeTrait<_dt>::ctype;        \
        typecvt_kern_n2n<ctype_src, ctype_dest>(dst, src, stream); \
        return;                                                    \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    } else {
        switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                               \
    case DTypeTrait<_dt>::enumv: {                                            \
        auto dst_param = dst.layout.dtype.param<_dt>();                       \
        using ctype_dest = typename DTypeTrait<_dt>::ctype;                   \
        typecvt_kern_n2q<ctype_src, ctype_dest>(dst, src, dst_param, stream); \
        return;                                                               \
    }
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb);
            default:
                megdnn_assert_internal(0);
#undef cb
        }
    }
}
}  // namespace

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    check_exec(src.layout, dst.layout);
    bool is_src_quantized =
            src.layout.dtype.category() == DTypeCategory::QUANTIZED;
    auto stream = hip_stream(handle());
    if (!is_src_quantized)
        switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                 \
    case DTypeTrait<_dt>::enumv: {              \
        exec_src_normal<_dt>(dst, src, stream); \
        return;                                 \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    else {
        switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                           \
    case DTypeTrait<_dt>::enumv: {                        \
        auto param = src.layout.dtype.param<_dt>();       \
        exec_src_quantized<_dt>(dst, src, param, stream); \
        return;                                           \
    }
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    }
}

// vim: syntax=cpp.doxygen
