/**
 * \file dnn/src/common/matrix_mul.cpp
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

void MatrixMulForward::deduce_dtype(DType A, DType B, DType& C) {
    // Expect that the user specifies output dtype (C), we then do sanity
    // check on the dtype supplied by the user. C_dtype and C_dtype2 are the
    // expected dtypes. If the user does not specify an output dtype by setting
    // C = {}, we deduce one (C_dtype) and return it to the user.
    DType C_candi, C_candi2;
    if (A.category() == DTypeCategory::FLOAT) {
        C_candi = A;
    } else if (A.enumv() == DTypeEnum::Int8) {
        C_candi = dtype::Int32();
        C_candi2 = dtype::Int16();
    } else if (A.enumv() == DTypeEnum::Int16) {
        C_candi = dtype::Int32();
    } else if (A.enumv() == DTypeEnum::QuantizedS8) {
        C_candi = dtype::QuantizedS32(mul_scale(A, B));
    } else if (A.enumv() == DTypeEnum::Quantized8Asymm) {
        C_candi = dtype::QuantizedS32(mul_scale(A, B));
    } else if (A.enumv() == DTypeEnum::Quantized4Asymm) {
        C_candi = dtype::QuantizedS32(mul_scale(A, B));
    } else if (A.enumv() == DTypeEnum::QuantizedS4) {
        C_candi = dtype::QuantizedS16(mul_scale(A, B));
    }
    if (!C.valid()) {
        C = C_candi;
    }
    megdnn_assert(C.valid() && (C == C_candi || C == C_candi2),
                  "unsupported MatMul(%s, %s) -> %s", A.name(), B.name(),
                  C.name());
}

void MatrixMulForward::deduce_layout(const TensorLayout& A,
                                     const TensorLayout& B, TensorLayout& C) {
    megdnn_assert(A.dtype.enumv() == B.dtype.enumv(),
                  "matmul input should be of same dtype, got %s and %s",
                  A.dtype.name(), B.dtype.name());
    deduce_dtype(A.dtype, B.dtype, C.dtype);
    size_t A0, A1, B0, B1;
    if (param().format == param::MatrixMul::Format::DEFAULT) {
        megdnn_assert(A.ndim == 2 && B.ndim == 2,
                      "matmul requires input to be 2-dimensional; get: %s %s",
                      A.TensorShape::to_string().c_str(),
                      B.TensorShape::to_string().c_str());
        A0 = A.shape[0];
        A1 = A.shape[1];
        B0 = B.shape[0];
        B1 = B.shape[1];
        if (m_param.transposeA)
            std::swap(A0, A1);
        if (m_param.transposeB)
            std::swap(B0, B1);
        megdnn_assert(A1 == B0,
                      "shape mismatch in matmal: (transposed) A is (%zu,%zu), "
                      "(transposed) B is (%zu,%zu)",
                      A0, A1, B0, B1);
        C = TensorLayout(TensorShape({A0, B1}), C.dtype);
    } else {
        auto do_deduce = [&](size_t pack_size) {
            megdnn_assert(A.ndim == 4 && B.ndim == 3,
                          "matmul requires input dimension to be A(4), B(3); "
                          "get: %s %s",
                          A.TensorShape::to_string().c_str(),
                          B.TensorShape::to_string().c_str());
            A0 = A.shape[0];
            A1 = A.shape[1];
            B0 = B.shape[0];
            B1 = B.shape[1];
            if (m_param.transposeA)
                std::swap(A0, A1);
            if (m_param.transposeB)
                std::swap(B0, B1);
            megdnn_assert(A1 == B0,
                          "shape mismatch in matmal: (transposed) A is "
                          "(%zu,%zu,4,4), "
                          "(transposed) B is (%zu,%zu,4)",
                          A0, A1, B0, B1);
            C = TensorLayout(TensorShape({A0, B1, pack_size}), C.dtype);
        };
        do_deduce(pack_size(param().format));
    }
}

void MatrixMulForward::check_exec(const TensorLayout& A, const TensorLayout& B,
                                  const TensorLayout& C,
                                  size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        std::string msg;
        msg.append(megdnn_mangle("A="));
        msg.append(A.to_string());
        msg.append(megdnn_mangle(", B="));
        msg.append(B.to_string());
        msg.append(megdnn_mangle(", C="));
        msg.append(C.to_string());
        msg.append(megdnn_mangle(", transposeA="));
        msg.append(std::to_string(param().transposeA));
        msg.append(megdnn_mangle(", transposeB="));
        msg.append(std::to_string(param().transposeB));
        return msg;
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    if (param().format == param::MatrixMul::Format::DEFAULT) {
        megdnn_assert_eq_size_t(A.ndim, 2_z);
        megdnn_assert_eq_size_t(B.ndim, 2_z);
        megdnn_assert_eq_size_t(C.ndim, 2_z);

        megdnn_assert(A.stride[1] == 1);
        megdnn_assert(A.stride[0] >= static_cast<ptrdiff_t>(A.shape[1]));
        megdnn_assert(B.stride[1] == 1);
        megdnn_assert(B.stride[0] >= static_cast<ptrdiff_t>(B.shape[1]));
        megdnn_assert(C.stride[1] == 1);
        megdnn_assert(C.stride[0] >= static_cast<ptrdiff_t>(C.shape[1]));
        size_t A0, A1, B0, B1, C0, C1;
        A0 = A.shape[0];
        A1 = A.shape[1];
        B0 = B.shape[0];
        B1 = B.shape[1];
        C0 = C.shape[0];
        C1 = C.shape[1];
        if (m_param.transposeA)
            std::swap(A0, A1);
        if (m_param.transposeB)
            std::swap(B0, B1);
        megdnn_assert(A0 == C0, "%s", errmsg().c_str());
        megdnn_assert(B1 == C1, "%s", errmsg().c_str());
        megdnn_assert(A1 == B0, "%s", errmsg().c_str());
    } else {
        megdnn_assert_eq_size_t(A.ndim, 4_z);
        megdnn_assert_eq_size_t(B.ndim, 3_z);
        megdnn_assert_eq_size_t(C.ndim, 3_z);

        megdnn_assert_contiguous(A);
        megdnn_assert_contiguous(B);
        megdnn_assert_contiguous(C);
        size_t A0, A1, B0, B1, C0, C1;
        A0 = A.shape[0];
        A1 = A.shape[1];
        B0 = B.shape[0];
        B1 = B.shape[1];
        C0 = C.shape[0];
        C1 = C.shape[1];
        if (m_param.transposeA)
            std::swap(A0, A1);
        if (m_param.transposeB)
            std::swap(B0, B1);
        megdnn_assert(A0 == C0, "%s", errmsg().c_str());
        megdnn_assert(B1 == C1, "%s", errmsg().c_str());
        megdnn_assert(A1 == B0, "%s", errmsg().c_str());
    }

    megdnn_assert(A.dtype.enumv() == B.dtype.enumv());
    if (A.dtype.category() == DTypeCategory::FLOAT) {
        megdnn_assert(A.dtype == C.dtype);
    } else if (A.dtype == dtype::Int8()) {
        megdnn_assert(C.dtype == dtype::Int16() || C.dtype == dtype::Int32());
    } else if (A.dtype.enumv() == DTypeEnum::QuantizedS8 ||
               A.dtype.enumv() == DTypeEnum::Quantized8Asymm ||
               A.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        megdnn_assert(C.dtype.enumv() == DTypeEnum::QuantizedS32);
    } else if(A.dtype.enumv() == DTypeEnum::QuantizedS4){
        megdnn_assert(C.dtype.enumv() == DTypeEnum::QuantizedS16);
    }
    megdnn_assert(param().compute_mode !=
                          Param::ComputeMode::FLOAT32 MEGDNN_INC_FLOAT16(
                                  || A.dtype == dtype::Float16() ||
                                  A.dtype == dtype::BFloat16()),
                  "ComputeMode::FLOAT32 is only available for Float16/BFloat16 "
                  "input / output.");
    auto required_workspace_in_bytes = get_workspace_in_bytes(A, B, C);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

size_t MatrixMulForward::pack_size(const Param::Format format) {
    switch (format) {
        case Param::Format::DEFAULT:
            return 1;
        case Param::Format::MK4:
            return 4;
        case Param::Format::MK4_DOT:
            return 4;
        case Param::Format::MK8:
            return 8;
        default:
            megdnn_throw(megdnn_mangle("Unknown matmul format."));
    }
}

}  // namespace megdnn
   // vim: syntax=cpp.doxygen
