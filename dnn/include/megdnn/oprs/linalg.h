/**
 * \file dnn/include/megdnn/oprs/linalg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

class BatchedMatrixMulForward
        : public OperatorBase,
          public detail::MultiAlgoOpr<BatchedMatrixMulForward, 3> {
    DEF_OPR_PARAM(MatrixMul);
    DEF_OPR_IMPL(BatchedMatrixMulForward, OperatorBase, 2, 1);

public:
    /**
     * \brief C = op(A) * op(B)
     * \param A (B, m, k) if transposeA is false, (B, k, m) otherwise
     * \param B (B, k, n) if transposeB is false, (B, n, k) otherwise
     * \param C (B, m, n)
     *
     * A, B, C must be 3-dimensional and C must be contiguous. A and B must
     * have stride[2] == 1, and stride[1] >= shape[2],
     * and stride[0] >= shape[1] * stride[1]
     *
     * op(A) = A if transposeA is false, otherwise op(A) = A^t.
     * op(B) = B if transposeB is false, otherwise op(B) = B^t.
     */
    virtual void exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                      _megdnn_tensor_out C, _megdnn_workspace workspace) = 0;
    void deduce_dtype(DType A, DType B, DType &C);
    void deduce_layout(const TensorLayout& A, const TensorLayout& B,
                       TensorLayout& C);
    virtual size_t get_workspace_in_bytes(const TensorLayout& A,
                                          const TensorLayout& B,
                                          const TensorLayout& C) = 0;

protected:
    void check_exec(const TensorLayout& A, const TensorLayout& B,
                    const TensorLayout& C, size_t workspace_in_bytes);
};
using BatchedMatrixMul = BatchedMatrixMulForward;

class MatrixMulForward : public OperatorBase,
                         public detail::MultiAlgoOpr<MatrixMulForward, 3> {
    DEF_OPR_PARAM(MatrixMul);
    DEF_OPR_IMPL(MatrixMulForward, OperatorBase, 2, 1);

public:
    /**
     * \brief C = op(A) * op(B)
     * \param A (m, k) if transposeA is false, (k, m) otherwise
     * \param B (k, n) if transposeB is false, (n, k) otherwise
     * \param C (m, n)
     *
     * A, B, C must be 2-dimensional and C must be contiguous. A and B must
     * have stride[1] == 1, and stride[0] >= shape[1]
     *
     * op(A) = A if transposeA is false, otherwise op(A) = A^t.
     * op(B) = B if transposeB is false, otherwise op(B) = B^t.
     */
    virtual void exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                      _megdnn_tensor_out C, _megdnn_workspace workspace) = 0;
    void deduce_dtype(DType A, DType B, DType& C);
    void deduce_layout(const TensorLayout& A, const TensorLayout& B,
                       TensorLayout& C);
    virtual size_t get_workspace_in_bytes(const TensorLayout& A,
                                          const TensorLayout& B,
                                          const TensorLayout& C) = 0;

    static size_t pack_size (const Param::Format format);
protected:
    void check_exec(const TensorLayout& A, const TensorLayout& B,
                    const TensorLayout& C, size_t workspace_in_bytes);
};
using MatrixMul = MatrixMulForward;

/*!
 * \brief compute the inverse of a batch of matrices
 *
 * Input and output tensors have the same shape [..., n, n] where the last two
 * dimensions represent the matrices.
 *
 * Currently only float32 is supported.
 */
class MatrixInverse : public OperatorBase {
    DEF_OPR_IMPL(MatrixInverse, OperatorBase, 1, 1);
    DEF_OPR_PARAM(Empty);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& dst);

protected:
    /*!
     * \brief get canonized params; throw exception on error.
     *
     * Note that \p batch and \p n can be null
     */
    static void canonize_params(const TensorLayout& layout, size_t* batch,
                                size_t* n);

    /*!
     * \brief canonize and validate input params for exec() impls
     *
     * Since get_workspace_in_bytes() would be called, \p batch and \p n can not
     * be null
     */
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    _megdnn_workspace workspace, size_t* batch, size_t* n);

    virtual size_t get_workspace_in_bytes(size_t batch, size_t n,
                                          size_t dtype_size) = 0;
};

//! inter-product of two vectors
class DotForward : public OperatorBase {
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(DotForward, OperatorBase, 2, 1);

public:
    /**
     * \param[in] A
     * \param[in] B
     * \param[out] C
     *
     * Calculating the dot product of A and B and store it in C.
     * A, B, C must be contiguous. A and B must have the same 1-dimensional
     * shape and non-negative strides. C must be scalar.
     */
    virtual void exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                      _megdnn_tensor_out C, _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& A, const TensorLayout& B,
                       TensorLayout& C);
    virtual size_t get_workspace_in_bytes(const TensorLayout& A,
                                          const TensorLayout& B,
                                          const TensorLayout& C) = 0;

protected:
    void check_exec(const TensorLayout& A, const TensorLayout& B,
                    const TensorLayout& C, size_t workspace_in_bytes);
};
using Dot = DotForward;

/*!
 * \brief Compute the singular value decomposition of a batch of matrices
 *
 * Input tensors have the shape [..., m, n], where the last two
 * dimensions represent the matrices. For the output tensor u, s, vt,
 * the following equation holds: u * diag(s) * vt == src.
 *
 * Currently only float32 is supported.
 */
class SVDForward : public OperatorBase {
    DEF_OPR_IMPL(SVDForward, OperatorBase, 1, 3);
    DEF_OPR_PARAM(SVD);

public:
    /**
     * \brief u, s, vt = SVD(src) and u * diag(s) * vt == src
     * \param src (..., m, n) The input tensor, let p = min(m, n)
     * \param u (..., m, p) if full_matrices is false,
                (..., m, m) if full_matrices is true,
                empty tensor if compute_uv is false.
                The left singular vector.

     * \param s (..., p) The singular values.
     * \param vt (..., p, n) if full_matrices is false,
                 (..., n, n) if full_matrices is true,
                 empty tensor if compute_uv is false.
                 The right singular vector.
     *
     * src must be contiguous. The computation might be significantly faster
     * if compute_uv is false (default to true).
     *
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out u,
                      _megdnn_tensor_out s, _megdnn_tensor_out vt,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& u,
                       TensorLayout& s, TensorLayout& vt);
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& u, const TensorLayout& s,
                                  const TensorLayout& vt);

protected:
    static void canonize_params(const TensorLayout& layout, size_t* batch,
                                size_t* m, size_t* n);
    virtual size_t get_workspace_in_bytes(size_t block_cnt, size_t m, size_t n,
                                          size_t dtype_size) = 0;
    void check_exec(const TensorLayout& src, const TensorLayout& u,
                    const TensorLayout& s, const TensorLayout& vt,
                    size_t workspace_in_bytes);
};

using SVD = SVDForward;

}  // namespace megdnn

#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen
