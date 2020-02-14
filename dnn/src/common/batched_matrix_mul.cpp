/**
 * \file dnn/src/common/batched_matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void BatchedMatrixMulForward::deduce_dtype(DType A, DType B, DType &C) {
    DType C_candi, C_candi2;
    if (A.category() == DTypeCategory::FLOAT) {
        C_candi = A;
    } else if (A.enumv() == DTypeEnum::Int8) {
        C_candi = dtype::Int32();
        C_candi2 = dtype::Int16();
    } else if (A.enumv() == DTypeEnum::QuantizedS8) {
        C_candi = dtype::QuantizedS32(mul_scale(A, B));
    } else if (A.enumv() == DTypeEnum::Quantized8Asymm) {
        C_candi = dtype::QuantizedS32(mul_scale(A, B));
    } else if (A.enumv() == DTypeEnum::Quantized4Asymm) {
        C_candi = dtype::QuantizedS32(mul_scale(A, B));
    }
    if (!C.valid()) {
        C = C_candi;
    }
    megdnn_assert(C.valid() && (C == C_candi || C == C_candi2),
                  "unsupported BatchedMatMul(%s, %s) -> %s", A.name(), B.name(),
                  C.name());
}
void BatchedMatrixMulForward::deduce_layout(const TensorLayout& A,
                                            const TensorLayout& B,
                                            TensorLayout& C) {
    auto errmsg = [&]() {
        std::string msg;
        msg.append(megdnn_mangle("A="));
        msg.append(A.to_string());
        msg.append(megdnn_mangle(", B="));
        msg.append(B.to_string());
        msg.append(megdnn_mangle(", C="));
        msg.append(C.to_string());
        msg.append(megdnn_mangle(", transposeA="));
        msg.append(std::to_string(m_param.transposeA));
        msg.append(megdnn_mangle(", transposeB="));
        msg.append(std::to_string(m_param.transposeB));
        return msg;
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    auto good_layout = [](const TensorLayout& l) {
        // l.stride[0] == 0 because im2col conv need batched matrixmul and
        // filter tensor need to be broadcasted. It's only implemented in
        // opencl.
        return l.ndim == 3 && l.stride[2] == 1 &&
               l.stride[1] >= static_cast<ptrdiff_t>(l.shape[2]) &&
               (l.shape[0] == 1 ||
                l.stride[0] >=
                        static_cast<ptrdiff_t>(l.shape[1]) * l.stride[1] ||
                l.stride[0] == 0);
    };
    size_t A0, A1, B0, B1;
    A0 = A.shape[1];
    A1 = A.shape[2];
    B0 = B.shape[1];
    B1 = B.shape[2];
    if (m_param.transposeA)
        std::swap(A0, A1);
    if (m_param.transposeB)
        std::swap(B0, B1);
    deduce_dtype(A.dtype, B.dtype, C.dtype);
    megdnn_assert(good_layout(A) && good_layout(B) && A1 == B0 &&
                          A[0] == B[0] && A.dtype.enumv() == B.dtype.enumv(),
                  "bad input layouts: %s", errmsg().c_str());
    C = TensorLayout(TensorShape({A[0], A0, B1}), C.dtype);
}

void BatchedMatrixMulForward::check_exec(const TensorLayout& A,
                                         const TensorLayout& B,
                                         const TensorLayout& C,
                                         size_t workspace_in_bytes) {
    TensorLayout C_expect;
    deduce_layout(A, B, C_expect);
    megdnn_assert(C_expect.eq_layout(C), "bad layout for C: expect=%s got=%s",
                  C_expect.to_string().c_str(), C.to_string().c_str());
    auto required_workspace_in_bytes = get_workspace_in_bytes(A, B, C);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes,
                  "needed workspace: %zu; got: %zu",
                  required_workspace_in_bytes, workspace_in_bytes);
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen
