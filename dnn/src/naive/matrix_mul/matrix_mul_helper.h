/**
 * \file dnn/src/naive/matrix_mul/matrix_mul_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>
#include "megdnn/dtype.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

template <typename ctype, typename otype, typename enable = void>
struct Getter {
    Getter(const DType&){};
    otype operator()(ctype item) { return item; }
};

template <typename ctype, typename otype>
struct Getter<ctype, otype,
              typename std::enable_if_t<std::is_same<ctype, uint8_t>::value>> {
    otype zp;
    Getter(const DType& dtype) {
        zp = dtype.param<dtype::Quantized8Asymm>().zero_point;
    }
    otype operator()(ctype item) { return static_cast<otype>(item) - zp; }
};

template <typename itype, typename otype, bool transA, bool transB,
          typename comp_type = otype>
void run_matrix_mul_tpl(const itype* A, const itype* B, otype* C, size_t M,
                        size_t N, size_t K, size_t LDA, size_t LDB, size_t LDC,
                        const DType& A_type, const DType& B_type) {
    Getter<itype, comp_type> getterA(A_type), getterB(B_type);
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            comp_type res = comp_type(0);
            for (size_t k = 0; k < K; ++k) {
                comp_type av = transA ? getterA(A[k * LDA + m])
                                      : getterA(A[m * LDA + k]),
                          bv = transB ? getterB(B[n * LDB + k])
                                      : getterB(B[k * LDB + n]);
                res += av * bv;
            }
            C[m * LDC + n] = res;
        }
    }
}

template <typename itype, typename otype, bool transA, bool transB,
          typename comp_type = otype>
void run_matrix_mul_mk4_tpl(const itype* A, const itype* B, otype* C, size_t M,
                            size_t N, size_t K, size_t LDA, size_t LDB,
                            size_t LDC, const DType& A_type,
                            const DType& B_type) {
    Getter<itype, comp_type> getterA(A_type), getterB(B_type);
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            comp_type res[4] = {comp_type(0)};
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < 4; i++) {
                    comp_type av, bv;
                    for (size_t j = 0; j < 4; j++) {
                        av = transA ? getterA(A[k * LDA + m * 16 + 4 * j + i])
                                    : getterA(A[m * LDA + k * 16 + 4 * j + i]),
                        bv = transB ? getterB(B[n * LDB + k * 4 + j])
                                    : getterB(B[k * LDB + n * 4 + j]);
                        res[i] += av * bv;
                    }
                }
            }
            for (size_t i = 0; i < 4; i++) {
                C[m * LDC + n * 4 + i] = res[i];
            }
        }
    }
}

template <typename itype, typename otype, bool transA, bool transB,
          typename comp_type = otype>
void run_matrix_mul_mk4_dot_tpl(const itype* A, const itype* B, otype* C,
                                size_t M, size_t N, size_t K, size_t LDA,
                                size_t LDB, size_t LDC, const DType& A_type,
                                const DType& B_type) {
    Getter<itype, comp_type> getterA(A_type), getterB(B_type);
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            comp_type res[4] = {comp_type(0)};
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < 4; i++) {
                    comp_type av, bv;
                    for (size_t j = 0; j < 4; j++) {
                        av = transA ? getterA(A[k * LDA + m * 16 + 4 * i + j])
                                    : getterA(A[m * LDA + k * 16 + 4 * i + j]),
                        bv = transB ? getterB(B[n * LDB + k * 4 + j])
                                    : getterB(B[k * LDB + n * 4 + j]);
                        res[i] += av * bv;
                    }
                }
            }
            for (size_t i = 0; i < 4; i++) {
                C[m * LDC + n * 4 + i] = res[i];
            }
        }
    }
}

template <typename itype, typename otype, bool transA, bool transB,
          typename comp_type = otype>
void run_matrix_mul_mk8_tpl(const itype* A, const itype* B, otype* C, size_t M,
                            size_t N, size_t K, size_t LDA, size_t LDB,
                            size_t LDC, const DType& A_type,
                            const DType& B_type) {
    Getter<itype, comp_type> getterA(A_type), getterB(B_type);
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            std::vector<comp_type> res(8, comp_type(0));
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < 8; i++) {
                    comp_type av, bv;
                    for (size_t j = 0; j < 8; j++) {
                        av = transA ? getterA(A[k * LDA + m * 64 + 8 * j + i])
                                    : getterA(A[m * LDA + k * 64 + 8 * j + i]),
                        bv = transB ? getterB(B[n * LDB + k * 8 + j])
                                    : getterB(B[k * LDB + n * 8 + j]);
                        res[i] += av * bv;
                    }
                }
            }
            for (size_t i = 0; i < 8; i++) {
                C[m * LDC + n * 8 + i] = res[i];
            }
        }
    }
}

template <bool transA, bool transB>
void exec_matrix_mul_quint4x4x32_helper(_megdnn_tensor_in A,
                                        _megdnn_tensor_in B,
                                        _megdnn_tensor_out C,
                                        _megdnn_workspace workspace,
                                        const param::MatrixMul& param) {
    auto convert_layout = [](const TensorLayout& layout) {
        auto ret = layout;
        auto param = layout.dtype.param<dtype::Quantized4Asymm>();
        ret.dtype = dtype::Quantized8Asymm(param.scale, param.zero_point);
        return ret;
    };
    TensorND nA = {workspace.raw_ptr, convert_layout(A.layout)};
    TensorND nB = {workspace.raw_ptr + nA.layout.span().dist_byte(),
                   convert_layout(B.layout)};
    auto convert_4to8 = [](const TensorND& in, const TensorND& out) {
        auto ptr =
                static_cast<uint8_t*>(in.raw_ptr) + in.layout.span().low_byte;
        auto out_ptr =
                out.compatible_ptr<uint8_t>() + out.layout.span().low_byte;
        for (size_t i = 0; i < in.layout.span().dist_elem(); i += 2) {
            uint8_t val = ptr[i / 2];
            uint8_t val0 = val & 0xF;
            uint8_t val1 = (val >> 4) & 0xF;
            out_ptr[i] = val0;
            out_ptr[i + 1] = val1;
        }
    };
    convert_4to8(A, nA);
    convert_4to8(B, nB);
    auto M = C.layout.shape[0], N = C.layout.shape[1];
    auto K = A.layout.shape[param.transposeA ? 0 : 1];
    auto LDA = A.layout.stride[0], LDB = B.layout.stride[0],
         LDC = C.layout.stride[0];
    run_matrix_mul_tpl<uint8_t, dt_int32, transA, transB, dt_int32>(
            nA.compatible_ptr<uint8_t>(), nB.compatible_ptr<uint8_t>(),
            C.compatible_ptr<dt_int32>(), M, N, K, LDA, LDB, LDC,
            nA.layout.dtype, nB.layout.dtype);
}
template <bool transA, bool transB>
void exec_matrix_mul_qint4x4x16_helper(_megdnn_tensor_in A, _megdnn_tensor_in B,
                                       _megdnn_tensor_out C,
                                       _megdnn_workspace workspace,
                                       const param::MatrixMul& param) {
    auto convert_layout = [](const TensorLayout& layout) {
        auto ret = layout;
        auto param = layout.dtype.param<dtype::QuantizedS4>();
        ret.dtype = dtype::QuantizedS8(param.scale);
        return ret;
    };
    TensorND nA = {workspace.raw_ptr, convert_layout(A.layout)};
    TensorND nB = {workspace.raw_ptr + nA.layout.span().dist_byte(),
                   convert_layout(B.layout)};
    auto convert_4to8 = [](const TensorND& in, const TensorND& out) {
        auto ptr = static_cast<int8_t*>(in.raw_ptr) + in.layout.span().low_byte;
        auto out_ptr =
                out.compatible_ptr<int8_t>() + out.layout.span().low_byte;
        for (size_t i = 0; i < in.layout.span().dist_elem(); i += 2) {
            int8_t cur = ptr[i / 2];
            out_ptr[i] = cur << 4;
            out_ptr[i] = out_ptr[i] >> 4;
            out_ptr[i + 1] = cur >> 4;
        }
    };
    convert_4to8(A, nA);
    convert_4to8(B, nB);
    auto M = C.layout.shape[0], N = C.layout.shape[1];
    auto K = A.layout.shape[param.transposeA ? 0 : 1];
    auto LDA = A.layout.stride[0], LDB = B.layout.stride[0],
         LDC = C.layout.stride[0];
    run_matrix_mul_tpl<int8_t, dt_int16, transA, transB, dt_int16>(
            nA.compatible_ptr<int8_t>(), nB.compatible_ptr<int8_t>(),
            C.compatible_ptr<dt_int16>(), M, N, K, LDA, LDB, LDC,
            nA.layout.dtype, nB.layout.dtype);
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
