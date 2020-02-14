/**
 * \file dnn/src/x86/type_cvt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/x86/type_cvt/opr_impl.h"
#include <immintrin.h>
#include "src/x86/elemwise_helper/kimpl/typecvt.h"
#include "src/x86/elemwise_op.h"
#include "src/x86/utils.h"

using namespace megdnn;
using namespace x86;
#define DISPATCH_CONVERT_TYPE                                                \
    DISPATCH_QUANTIZED(QuantizedS32, dt_qint32, Quantized8Asymm, dt_quint8); \
    DISPATCH_QUANTIZED(Quantized8Asymm, dt_quint8, Quantized8Asymm,          \
                       dt_quint8);                                           \
    DISPATCH_QUANTIZED(QuantizedS8, dt_qint8, Quantized8Asymm, dt_quint8);   \
    DISPATCH_QUANTIZED(Float32, dt_float32, Quantized8Asymm, dt_quint8);     \
    DISPATCH_QUANTIZED(QuantizedS32, dt_qint32, QuantizedS8, dt_qint8);      \
    DISPATCH_QUANTIZED(QuantizedS8, dt_qint8, QuantizedS8, dt_qint8);        \
    DISPATCH_QUANTIZED(Quantized8Asymm, dt_quint8, QuantizedS8, dt_qint8);   \
    DISPATCH_QUANTIZED(Float32, dt_float32, QuantizedS8, dt_qint8);          \
    DISPATCH_QUANTIZED(QuantizedS8, dt_qint8, QuantizedS32, dt_qint32);      \
    DISPATCH_QUANTIZED(Quantized8Asymm, dt_quint8, QuantizedS32, dt_qint32); \
    DISPATCH_QUANTIZED(QuantizedS32, dt_qint32, QuantizedS32, dt_qint32);    \
    DISPATCH_QUANTIZED(Float32, dt_float32, QuantizedS32, dt_qint32);        \
    DISPATCH_QUANTIZED(QuantizedS8, dt_qint8, Float32, dt_float32);          \
    DISPATCH_QUANTIZED(Quantized8Asymm, dt_quint8, Float32, dt_float32);     \
    DISPATCH_QUANTIZED(QuantizedS32, dt_qint32, Float32, dt_float32);

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    DType src_dtype = src.layout.dtype;
    DType dst_dtype = dst.layout.dtype;
    size_t nr_elems = src.layout.total_nr_elems();
    bool execed = false;
    if (src.layout.is_contiguous() && dst.layout.is_contiguous()) {
        if (is_supported(SIMDType::SSE4_2)) {
            using namespace dtype;
#define DISPATCH_QUANTIZED(_stype_enumv, _stype, _dtype_enumv, _dtype)     \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&            \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {            \
        using op = TypeCvtOp<SIMDType::SSE4_2, _stype, _dtype>;            \
        thin_function<void(const _stype*, _dtype*, DType, DType, size_t)>  \
                run = OpCallerUnary<op, SIMDType::SSE4_2>::run;            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(run(src.compatible_ptr<_stype>(),     \
                                         dst.compatible_ptr<_dtype>(),     \
                                         src_dtype, dst_dtype, nr_elems)); \
        execed = true;                                                     \
    }
            DISPATCH_CONVERT_TYPE
#undef DISPATCH_QUANTIZED
        }
    }
    if (!execed) {
        fallback::TypeCvtImpl::exec(src, dst);
    }
}

#undef DISPATCH_CONVERT_TYPE

// vim: syntax=cpp.doxygen
