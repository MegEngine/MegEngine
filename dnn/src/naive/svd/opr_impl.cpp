/**
 * \file dnn/src/naive/svd/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"

#include "src/naive/handle.h"

namespace {

/*
 * This is a modified version of a (not so) simple standalone Singular Value
 * Decomposition implementation by Dhairya Malhotra. It was released into Public
 * Domain by author at
 * https://stackoverflow.com/questions/3856072/single-value-decomposition-implementation-c
 */

#define ROW_MAJOR_MAT(mat, col_dim, x, y) ((mat)[(x) * (col_dim) + (y)])
#define U(i, j) ROW_MAJOR_MAT(U_, dim[0], i, j)
#define S(i, j) ROW_MAJOR_MAT(S_, dim[1], i, j)
#define V(i, j) ROW_MAJOR_MAT(V_, dim[1], i, j)

template <class T>
void GivensL(T* S_, const size_t dim[2], size_t m, T a, T b) {
    T r = sqrt(a * a + b * b);
    if (fabs(r) < 1e-7)
        return;
    T c = a / r;
    T s = -b / r;

    for (size_t i = 0; i < dim[1]; i++) {
        T S0 = S(m + 0, i);
        T S1 = S(m + 1, i);
        S(m, i) += S0 * (c - 1);
        S(m, i) += S1 * (-s);

        S(m + 1, i) += S0 * (s);
        S(m + 1, i) += S1 * (c - 1);
    }
}

template <class T>
void GivensR(T* S_, const size_t dim[2], size_t m, T a, T b) {
    T r = sqrt(a * a + b * b);
    if (fabs(r) < 1e-7)
        return;
    T c = a / r;
    T s = -b / r;

    for (size_t i = 0; i < dim[0]; i++) {
        T S0 = S(i, m + 0);
        T S1 = S(i, m + 1);
        S(i, m) += S0 * (c - 1);
        S(i, m) += S1 * (-s);

        S(i, m + 1) += S0 * (s);
        S(i, m + 1) += S1 * (c - 1);
    }
}

template <class T>
void SVD(const size_t dim[2], T* U_, T* S_, T* V_, T eps = -1) {
    megdnn_assert(dim[0] >= dim[1]);

    {  // Bi-diagonalization
        size_t n = std::min(dim[0], dim[1]);
        std::vector<T> house_vec(std::max(dim[0], dim[1]));
        for (size_t i = 0; i < n; i++) {
            // Column Householder
            {
                T x1 = S(i, i);
                if (x1 < 0)
                    x1 = -x1;

                T x_inv_norm = 0;
                for (size_t j = i; j < dim[0]; j++) {
                    x_inv_norm += S(j, i) * S(j, i);
                }
                if (x_inv_norm > 1e-7)
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                T alpha = sqrt(1 + x1 * x_inv_norm);
                T beta = x_inv_norm / alpha;
                if (fabs(x_inv_norm) < 1e-7)
                    alpha = 0;  // nothing to do

                house_vec[i] = -alpha;
                for (size_t j = i + 1; j < dim[0]; j++) {
                    house_vec[j] = -beta * S(j, i);
                }
                if (S(i, i) < 0)
                    for (size_t j = i + 1; j < dim[0]; j++) {
                        house_vec[j] = -house_vec[j];
                    }
            }
            for (size_t k = i; k < dim[1]; k++) {
                T dot_prod = 0;
                for (size_t j = i; j < dim[0]; j++) {
                    dot_prod += S(j, k) * house_vec[j];
                }
                for (size_t j = i; j < dim[0]; j++) {
                    S(j, k) -= dot_prod * house_vec[j];
                }
            }
            for (size_t k = 0; k < dim[0]; k++) {
                T dot_prod = 0;
                for (size_t j = i; j < dim[0]; j++) {
                    dot_prod += U(k, j) * house_vec[j];
                }
                for (size_t j = i; j < dim[0]; j++) {
                    U(k, j) -= dot_prod * house_vec[j];
                }
            }

            // Row Householder
            if (i >= n - 1)
                continue;
            {
                T x1 = S(i, i + 1);
                if (x1 < -0)
                    x1 = -x1;

                T x_inv_norm = 0;
                for (size_t j = i + 1; j < dim[1]; j++) {
                    x_inv_norm += S(i, j) * S(i, j);
                }
                if (x_inv_norm > 1e-7)
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                T alpha = sqrt(1 + x1 * x_inv_norm);
                T beta = x_inv_norm / alpha;
                if (fabs(x_inv_norm) < 1e-7)
                    alpha = 0;  // nothing to do

                house_vec[i + 1] = -alpha;
                for (size_t j = i + 2; j < dim[1]; j++) {
                    house_vec[j] = -beta * S(i, j);
                }
                if (S(i, i + 1) < 0)
                    for (size_t j = i + 2; j < dim[1]; j++) {
                        house_vec[j] = -house_vec[j];
                    }
            }
            for (size_t k = i; k < dim[0]; k++) {
                T dot_prod = 0;
                for (size_t j = i + 1; j < dim[1]; j++) {
                    dot_prod += S(k, j) * house_vec[j];
                }
                for (size_t j = i + 1; j < dim[1]; j++) {
                    S(k, j) -= dot_prod * house_vec[j];
                }
            }
            for (size_t k = 0; k < dim[1]; k++) {
                T dot_prod = 0;
                for (size_t j = i + 1; j < dim[1]; j++) {
                    dot_prod += V(j, k) * house_vec[j];
                }
                for (size_t j = i + 1; j < dim[1]; j++) {
                    V(j, k) -= dot_prod * house_vec[j];
                }
            }
        }
    }

    size_t k0 = 0;
    if (eps < 0) {
        eps = 1.0;
        while (eps + (T)1.0 > 1.0)
            eps *= 0.5;
        eps *= 64.0;
    }
    while (k0 < dim[1] - 1) {  // Diagonalization
        T S_max = 0.0;
        for (size_t i = 0; i < dim[1]; i++)
            S_max = (S_max > S(i, i) ? S_max : S(i, i));

        while (k0 < dim[1] - 1 && fabs(S(k0, k0 + 1)) <= eps * S_max)
            k0++;
        if (k0 == dim[1] - 1)
            continue;

        size_t n = k0 + 2;
        while (n < dim[1] && fabs(S(n - 1, n)) > eps * S_max)
            n++;

        T alpha = 0;
        T beta = 0;
        // Compute mu
        if (n - k0 == 2 && fabs(S(k0, k0)) < 1e-7 &&
            fabs(S(k0 + 1, k0 + 1)) < 1e-7) {
            alpha = 0;
            beta = 1;
        } else {
            T C[2][2];
            C[0][0] = S(n - 2, n - 2) * S(n - 2, n - 2);
            if (n - k0 > 2)
                C[0][0] += S(n - 3, n - 2) * S(n - 3, n - 2);
            C[0][1] = S(n - 2, n - 2) * S(n - 2, n - 1);
            C[1][0] = S(n - 2, n - 2) * S(n - 2, n - 1);
            C[1][1] = S(n - 1, n - 1) * S(n - 1, n - 1) +
                      S(n - 2, n - 1) * S(n - 2, n - 1);

            T b = -(C[0][0] + C[1][1]) / 2;
            T c = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            T d = 0;
            if (b * b - c > 0)
                d = sqrt(b * b - c);
            else {
                T b = (C[0][0] - C[1][1]) / 2;
                T c = -C[0][1] * C[1][0];
                if (b * b - c > 0)
                    d = sqrt(b * b - c);
            }

            T lambda1 = -b + d;
            T lambda2 = -b - d;

            T d1 = lambda1 - C[1][1];
            d1 = (d1 < 0 ? -d1 : d1);
            T d2 = lambda2 - C[1][1];
            d2 = (d2 < 0 ? -d2 : d2);
            T mu = (d1 < d2 ? lambda1 : lambda2);

            alpha = S(k0, k0) * S(k0, k0) - mu;
            beta = S(k0, k0) * S(k0, k0 + 1);
        }

        for (size_t k = k0; k < n - 1; k++) {
            size_t dimU[2] = {dim[0], dim[0]};
            size_t dimV[2] = {dim[1], dim[1]};
            GivensR(S_, dim, k, alpha, beta);
            GivensL(V_, dimV, k, alpha, beta);

            alpha = S(k, k);
            beta = S(k + 1, k);
            GivensL(S_, dim, k, alpha, beta);
            GivensR(U_, dimU, k, alpha, beta);

            alpha = S(k, k + 1);
            beta = S(k, k + 2);
        }

        {  // Make S bi-diagonal again
            for (size_t i0 = k0; i0 < n - 1; i0++) {
                for (size_t i1 = 0; i1 < dim[1]; i1++) {
                    if (i0 > i1 || i0 + 1 < i1)
                        S(i0, i1) = 0;
                }
            }
            for (size_t i0 = 0; i0 < dim[0]; i0++) {
                for (size_t i1 = k0; i1 < n - 1; i1++) {
                    if (i0 > i1 || i0 + 1 < i1)
                        S(i0, i1) = 0;
                }
            }
            for (size_t i = 0; i < dim[1] - 1; i++) {
                if (fabs(S(i, i + 1)) <= eps * S_max) {
                    S(i, i + 1) = 0;
                }
            }
        }
    }
}

#undef U
#undef S
#undef V
}  // namespace

namespace megdnn {
namespace naive {

size_t SVDForwardImpl::get_workspace_in_bytes(size_t block_cnt, size_t m,
                                              size_t n, size_t dtype_size) {
    MEGDNN_MARK_USED_VAR(block_cnt);
    return get_workspace_bundle(m, n, dtype_size).total_size_in_bytes();
}

WorkspaceBundle SVDForwardImpl::get_workspace_bundle(size_t m, size_t n,
                                                     size_t dtype_size,
                                                     void* raw_ptr) {
    // Scratchpads for u and v.
    size_t dim0 = std::max(m, n);
    size_t dim1 = std::min(m, n);
    return {raw_ptr,
            {m * n * dtype_size, dim0 * dim0 * dtype_size,
             dim1 * dim1 * dtype_size},
            handle()->alignment_requirement()};
}

template <typename T>
void SVDForwardImpl::exec_internal(_megdnn_tensor_in src, _megdnn_tensor_out u,
                                   _megdnn_tensor_out s, _megdnn_tensor_out vt,
                                   _megdnn_workspace workspace, Param p) {
    size_t block_cnt, m, n;
    canonize_params(src.layout, &block_cnt, &m, &n);

    auto wbundle = get_workspace_bundle(m, n, sizeof(T), workspace.raw_ptr);
    const size_t max_mn = std::max(m, n);
    const size_t min_mn = std::min(m, n);
    const size_t src_block_size = src.layout.dtype.size(m * n);
    size_t src_off = m * n;
    size_t s_off = min_mn;
    size_t u_off = m * min_mn;
    size_t v_off = min_mn * n;

    for (size_t blk = 0; blk < block_cnt; blk++) {
        T* inp = src.ptr<T>() + blk * src_off;
        T* tmp_s = wbundle.get_workspace(0).ptr<T>();
        T* tmp_u = wbundle.get_workspace(1).ptr<T>();
        T* tmp_v = wbundle.get_workspace(2).ptr<T>();
#define TS(x, y) ROW_MAJOR_MAT(tmp_s, min_mn, x, y)  // m x n
#define TU(x, y) ROW_MAJOR_MAT(tmp_u, max_mn, x, y)  // m x m
#define TV(x, y) ROW_MAJOR_MAT(tmp_v, min_mn, x, y)  // n x n
#define INP(x, y) ROW_MAJOR_MAT(inp, n, x, y)
        bool transposed = false;
        if (m < n) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    TS(j, i) = INP(i, j);
                }
            }
            transposed = true;
        } else {
            memcpy(tmp_s, inp, src_block_size);
        }
        memset(tmp_u, 0, wbundle.get_workspace(1).size);
        memset(tmp_v, 0, wbundle.get_workspace(2).size);
        for (size_t i = 0; i < max_mn; i++) {
            TU(i, i) = 1;
        }
        for (size_t i = 0; i < min_mn; i++) {
            TV(i, i) = 1;
        }

        const size_t dim[2] = {max_mn, min_mn};
        ::SVD<T>(dim, tmp_u, tmp_s, tmp_v, (T)-1);

        T* out_s = s.ptr<T>() + blk * s_off;
        std::vector<std::pair<T, int>> sv_idx;
        for (size_t i = 0; i < min_mn; i++) {
            sv_idx.emplace_back(std::abs(TS(i, i)), i);
        }
        std::sort(sv_idx.begin(), sv_idx.end());
        std::reverse(sv_idx.begin(), sv_idx.end());
        for (size_t i = 0; i < min_mn; i++) {
            out_s[i] = sv_idx[i].first;
        }
        if (p.compute_uv) {
            T* out_u = u.ptr<T>() + blk * u_off;
            T* out_v = vt.ptr<T>() + blk * v_off;
#define OU(x, y) ROW_MAJOR_MAT(out_u, min_mn, x, y)
#define OV(x, y) ROW_MAJOR_MAT(out_v, n, x, y)
            for (size_t j = 0; j < min_mn; j++) {
                int tj = sv_idx[j].second;
                const T scale = (TS(tj, tj) < 0 ? -1 : 1);
                for (size_t i = 0; i < m; i++) {
                    OU(i, j) = (transposed ? TV(tj, i) : TU(i, tj)) * scale;
                }
            }
            for (size_t i = 0; i < min_mn; i++) {
                int ti = sv_idx[i].second;
                for (size_t j = 0; j < n; j++) {
                    OV(i, j) = (transposed ? TU(j, ti) : TV(ti, j));
                }
            }
#undef OV
#undef OU
        }
#undef TV
#undef TU
#undef TS
#undef ROW_MAJOR_MAT
    }
}

void SVDForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out u,
                          _megdnn_tensor_out s, _megdnn_tensor_out vt,
                          _megdnn_workspace workspace) {
    check_exec(src.layout, u.layout, s.layout, vt.layout, workspace.size);

    Param p = param();
    megdnn_assert(!p.compute_uv || !p.full_matrices,
                  "Computing full singular vectors is not supported in naive "
                  "implementation.");
    if (src.layout.dtype == dtype::Float32()) {
        using ctype = typename DTypeTrait<dtype::Float32>::ctype;
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                exec_internal<ctype>(src, u, s, vt, workspace, p));
        return;
    }
    megdnn_assert_internal(0);
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
