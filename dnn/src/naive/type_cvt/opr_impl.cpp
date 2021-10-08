/**
 * \file dnn/src/naive/type_cvt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"
#include "megdnn/tensor_iter.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/lowbit_utils.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_typecvt)

using namespace megdnn;
using namespace naive;

namespace {

template <typename T>
using QuantizedCType = std::enable_if_t<
        DTypeTrait<T>::category == DTypeCategory::QUANTIZED,
        typename DTypeTrait<T>::ctype>;
template <typename T>
using NormalCType = std::enable_if_t<
        DTypeTrait<T>::category != DTypeCategory::QUANTIZED,
        typename DTypeTrait<T>::ctype>;

template <typename T>
inline float from(QuantizedCType<T> in, DType dtype) {
    return dtype.param<T>().dequantize(in);
}

template <typename T>
inline typename DTypeTrait<T>::ctype from(NormalCType<T> in, DType) {
    return in;
}

template <typename T, typename P>
inline QuantizedCType<T> to(P in, DType dtype) {
    return dtype.param<T>().quantize(static_cast<float>(in));
}

template <typename T, typename P>
inline NormalCType<T> to(P in, DType) {
    return static_cast<typename DTypeTrait<T>::ctype>(in);
}

template <typename type_dest, typename type_src>
void do_cvt(const TensorND& dst, const TensorND& src) {
    auto dptr = tensor_iter_valonly<typename DTypeTrait<type_dest>::ctype>(dst).begin();
    auto iter = tensor_iter_valonly<typename DTypeTrait<type_src>::ctype>(src).begin();
    size_t nr_elems = src.layout.total_nr_elems();
    while (iter.logical_offset() < nr_elems) {
        *dptr = to<type_dest>(
                from<type_src>(*iter, src.layout.dtype), dst.layout.dtype);
        ++dptr;
        ++iter;
    }
}

template <typename type_dest>
void on_dest_ctype(HandleImpl* handle, const TensorND& dest, const TensorND& src) {
    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                                    \
    case DTypeTrait<_dt>::enumv: {                                                 \
        MIDOUT_BEGIN(megdnn_naive_typecvt, type_dest, _dt) {                       \
            MEGDNN_DISPATCH_CPU_KERN(handle, (do_cvt<type_dest, _dt>(dest, src))); \
        }                                                                          \
        MIDOUT_END();                                                              \
        return;                                                                    \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
        cb(::megdnn::dtype::Bool) cb(::megdnn::dtype::Uint16)
#undef cb
                default : megdnn_throw("bad dtype");
    }
}

}  // anonymous namespace

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    check_exec(src.layout, dst.layout);

    // exec
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                           \
    case DTypeTrait<_dt>::enumv: {                                        \
        on_dest_ctype<_dt>(static_cast<HandleImpl*>(handle()), dst, src); \
        break;                                                            \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
        cb(::megdnn::dtype::Bool) cb(::megdnn::dtype::Uint16)
#undef cb
                default : megdnn_throw("bad dtype");
    }
}

// vim: syntax=cpp.doxygen
