/**
 * \file dnn/src/naive/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/matrix_mul/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "./matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_matmul)

namespace megdnn {
namespace naive {

size_t MatrixMulForwardImpl::get_workspace_in_bytes(const TensorLayout& A,
                                                    const TensorLayout& B,
                                                    const TensorLayout&) {
    MIDOUT_BEGIN(
            megdnn_naive_matmul,
            midout_iv("MatrixMulForwardImpl::get_workspace_in_bytes"_hash)) {
        if (A.dtype.enumv() == DTypeEnum::Quantized4Asymm ||
            A.dtype.enumv() == DTypeEnum::QuantizedS4) {
            return (A.span().dist_elem() + B.span().dist_elem()) *
                   sizeof(uint8_t);
        }
        return 0;
    }
    MIDOUT_END();
}

template <bool TA, bool TB>
void dispatch_ta_tb(_megdnn_tensor_in A, _megdnn_tensor_in B,
                    _megdnn_tensor_out C, _megdnn_workspace workspace,
                    const MatrixMul::Param& param) {
    auto M = C.layout.shape[0], N = C.layout.shape[1];
    auto K = A.layout.shape[param.transposeA ? 0 : 1];
    auto LDA = A.layout.stride[0], LDB = B.layout.stride[0],
         LDC = C.layout.stride[0];

#define cb(_itype, _otype, _comp_type)                                         \
    if (param.format == param::MatrixMul::Format::DEFAULT) {                   \
        return run_matrix_mul_tpl<_itype, _otype, TA, TB, _comp_type>(         \
                A.compatible_ptr<_itype>(), B.compatible_ptr<_itype>(),        \
                C.compatible_ptr<_otype>(), M, N, K, LDA, LDB, LDC,            \
                A.layout.dtype, B.layout.dtype);                               \
    } else if (param.format == param::MatrixMul::Format::MK4) {                \
        return run_matrix_mul_mk4_tpl<_itype, _otype, TA, TB, _comp_type>(     \
                A.compatible_ptr<_itype>(), B.compatible_ptr<_itype>(),        \
                C.compatible_ptr<_otype>(), M, N, K, LDA, LDB, LDC,            \
                A.layout.dtype, B.layout.dtype);                               \
    } else if (param.format == param::MatrixMul::Format::MK4_DOT) {            \
        return run_matrix_mul_mk4_dot_tpl<_itype, _otype, TA, TB, _comp_type>( \
                A.compatible_ptr<_itype>(), B.compatible_ptr<_itype>(),        \
                C.compatible_ptr<_otype>(), M, N, K, LDA, LDB, LDC,            \
                A.layout.dtype, B.layout.dtype);                               \
    } else if (param.format == param::MatrixMul::Format::MK8) {                \
        return run_matrix_mul_mk8_tpl<_itype, _otype, TA, TB, _comp_type>(     \
                A.compatible_ptr<_itype>(), B.compatible_ptr<_itype>(),        \
                C.compatible_ptr<_otype>(), M, N, K, LDA, LDB, LDC,            \
                A.layout.dtype, B.layout.dtype);                               \
    }

    if (A.layout.dtype == dtype::Float32()) {
        cb(dt_float32, dt_float32, dt_float32);
#if !MEGDNN_DISABLE_FLOAT16
    } else if (A.layout.dtype == dtype::Float16()) {
        using Param = MatrixMul::Param;
        if (param.compute_mode == Param::ComputeMode::DEFAULT) {
            cb(dt_float16, dt_float16, dt_float16);
        } else if (param.compute_mode == Param::ComputeMode::FLOAT32) {
            cb(dt_float16, dt_float16, dt_float32);
        }
    } else if (A.layout.dtype == dtype::BFloat16()) {
        using Param = MatrixMul::Param;
        if (param.compute_mode == Param::ComputeMode::DEFAULT) {
            cb(dt_bfloat16, dt_bfloat16, dt_bfloat16);
        } else if (param.compute_mode == Param::ComputeMode::FLOAT32) {
            cb(dt_bfloat16, dt_bfloat16, dt_float32);
        }
#endif
    } else if (A.layout.dtype == dtype::Int8() &&
               C.layout.dtype == dtype::Int16()) {
        cb(dt_int8, dt_int16, dt_int16);
    } else if (A.layout.dtype == dtype::Int16() &&
               C.layout.dtype == dtype::Int32()) {
        cb(dt_int16, dt_int32, dt_int32);
    } else if ((A.layout.dtype == dtype::Int8() ||
                A.layout.dtype.enumv() == DTypeEnum::QuantizedS8) &&
               (C.layout.dtype == dtype::Int32() ||
                C.layout.dtype.enumv() == DTypeEnum::QuantizedS32)) {
        cb(dt_int8, dt_int32, dt_int32);
    } else if (A.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm &&
               C.layout.dtype.enumv() == DTypeEnum::QuantizedS32) {
        cb(uint8_t, dt_int32, dt_int32);
    } else if (A.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm &&
               C.layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&
               param.format == param::MatrixMul::Format::DEFAULT) {
        exec_matrix_mul_quint4x4x32_helper<TA, TB>(A, B, C, workspace, param);
        return;
    } else if (A.layout.dtype.enumv() == DTypeEnum::QuantizedS4 &&
               C.layout.dtype.enumv() == DTypeEnum::QuantizedS16 &&
               param.format == param::MatrixMul::Format::DEFAULT) {
        exec_matrix_mul_qint4x4x16_helper<TA, TB>(A, B, C, workspace, param);
        return;
    }
#undef cb
    megdnn_throw(ssprintf(
            "unsupported naive MatrixMul(%s, %s) -> %s (cmode = %d)",
            A.layout.dtype.name(), B.layout.dtype.name(), C.layout.dtype.name(),
            static_cast<int>(param.compute_mode)));
}

void MatrixMulForwardImpl::exec_internal(_megdnn_tensor_in A,
                                         _megdnn_tensor_in B,
                                         _megdnn_tensor_out C,
                                         _megdnn_workspace workspace,
                                         const Param& param) {
#define DISPATCH(TA, TB)                                    \
    if (param.transposeA == TA && param.transposeB == TB) { \
        dispatch_ta_tb<TA, TB>(A, B, C, workspace, param);  \
        return;                                             \
    }
    DISPATCH(true, true);
    DISPATCH(true, false);
    DISPATCH(false, true);
    DISPATCH(false, false);
#undef DISPATCH
    megdnn_assert_internal(0);
}

void MatrixMulForwardImpl::exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                                _megdnn_tensor_out C,
                                _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_matmul,
                 midout_iv("MatrixMulForwardImpl::exec"_hash)) {
        check_exec(A.layout, B.layout, C.layout, workspace.size);
        auto p = param();
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal(A, B, C, workspace, p));
    }
    MIDOUT_END();
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
