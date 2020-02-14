/**
 * \file dnn/src/common/cv/linalg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace megdnn {
namespace linalg {
/*!
 * solve linear system Ax=b. note that @A and @b will be modified. result x is
 * store in @b
 */
template <class value_type>
void solve(value_type* A, uint32_t n, value_type* b) {
#define AT(i, j) A[(i)*n + (j)]

    auto swap_row = [&](uint32_t i, uint32_t j, uint32_t start) {
        if (i == j)
            return;
        for (size_t k = start; k < n; k++)
            std::swap(AT(i, k), AT(j, k));
        std::swap(b[i], b[j]);
    };

    auto mult_row_scalar = [&](uint32_t row, value_type f, uint32_t start) {
        for (size_t j = start; j < n; j++)
            AT(row, j) *= f;
        b[row] *= f;
    };

    for (uint32_t i = 0; i < n; i++) {
        // swap the row which has the max absolute value to row i
        uint32_t idx = i;
        value_type max_abs_val = std::abs(AT(i, i));
        for (uint32_t j = i + 1; j < n; j++) {
            value_type abs_val = std::abs(AT(j, i));
            if (abs_val > max_abs_val) {
                max_abs_val = abs_val;
                idx = j;
            }
        }
        swap_row(i, idx, i);

        mult_row_scalar(i, value_type(1) / AT(i, i), i);
        auto row_i = A + i * n;
        for (uint32_t j = i + 1; j < n; j++) {
            value_type factor = AT(j, i);
            auto row_j = A + j * n;

            uint32_t k = i;
            uint32_t repeat = (n - i) / 8;
            uint32_t left = n - i - repeat * 8;
            while (repeat--) {
                row_j[k] -= row_i[k] * factor;
                row_j[k + 1] -= row_i[k + 1] * factor;
                row_j[k + 2] -= row_i[k + 2] * factor;
                row_j[k + 3] -= row_i[k + 3] * factor;
                row_j[k + 4] -= row_i[k + 4] * factor;
                row_j[k + 5] -= row_i[k + 5] * factor;
                row_j[k + 6] -= row_i[k + 6] * factor;
                row_j[k + 7] -= row_i[k + 7] * factor;
                k += 8;
            }

            switch (left) {
                case 7:
                    row_j[k + 6] -= row_i[k + 6] * factor;
                case 6:
                    row_j[k + 5] -= row_i[k + 5] * factor;
                case 5:
                    row_j[k + 4] -= row_i[k + 4] * factor;
                case 4:
                    row_j[k + 3] -= row_i[k + 3] * factor;
                case 3:
                    row_j[k + 2] -= row_i[k + 2] * factor;
                case 2:
                    row_j[k + 1] -= row_i[k + 1] * factor;
                case 1:
                    row_j[k] -= row_i[k] * factor;
                case 0:;
            }

            b[j] -= b[i] * factor;
        }
    }

    for (int i = int(n) - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            b[j] -= b[i] * AT(j, i);
        }
    }
#undef AT
}

template <class value_type>
void fill_eye(value_type* A, uint32_t n) {
    memset(A, 0, n * n * sizeof(value_type));
    for (uint32_t i = 0; i < n; i++)
        A[i * n + i] = 1;
}

/*!
 * compute the inverse of a matrix A and store it in B. A will be altered.
 */
template <class value_type>
void inverse_mat(value_type* A, value_type* B, uint32_t n) {
#define AT(A, i, j) A[(i)*n + (j)]

    auto swap_row = [&](value_type* A, uint32_t i, uint32_t j, uint32_t start) {
        if (i == j)
            return;
        for (size_t k = start; k < n; k++)
            std::swap(AT(A, i, k), AT(A, j, k));
    };

    auto mult_row_scalar = [&](value_type* A, uint32_t row, value_type f,
                               uint32_t start) {
        for (size_t j = start; j < n; j++)
            AT(A, row, j) *= f;
    };

    auto vec_axpy = [](value_type a, value_type* x, value_type* y, uint32_t m) {
        for (uint32_t i = 0; i < m; i++)
            *(y++) += a * *(x++);
    };

    fill_eye(B, n);

    for (uint32_t i = 0; i < n; i++) {
        // swap the row which has the max absolute value to row i
        uint32_t idx = i;
        value_type max_abs_val = std::abs(AT(A, i, i));
        for (uint32_t j = i + 1; j < n; j++) {
            value_type abs_val = std::abs(AT(A, j, i));
            if (abs_val > max_abs_val) {
                max_abs_val = abs_val;
                idx = j;
            }
        }
        swap_row(A, i, idx, 0);
        swap_row(B, i, idx, 0);

        value_type scale = value_type(1) / AT(A, i, i);

        mult_row_scalar(A, i, scale, i);
        mult_row_scalar(B, i, scale, 0);

        auto A_row_i = A + i * n, B_row_i = B + i * n;
        for (uint32_t j = i + 1; j < n; j++) {
            value_type factor = AT(A, j, i);
            auto A_row_j = A + j * n, B_row_j = B + j * n;
            vec_axpy(-factor, A_row_i + i, A_row_j + i, n - i);
            vec_axpy(-factor, B_row_i, B_row_j, n);
        }
    }

    for (int i = int(n) - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            value_type factor = -AT(A, j, i);
            // vec_axpy(factor, A + i * n, A + j * n, n);
            vec_axpy(factor, B + i * n, B + j * n, n);
        }
    }
#undef AT
}

/// C = A * B
/// A, B must point to memory space different from C
template <class value_type>
void mat_mult(const value_type* A, const value_type* B, value_type* C,
              uint32_t n) {
#define AT(A, i, j) A[(i)*n + (j)]
    memset(C, 0, n * n * sizeof(value_type));
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < n; j++)
                AT(C, i, j) += AT(A, i, k) * AT(B, k, j);
    }
#undef AT
}

template <class value_type>
void transpose_mat(const value_type* A, value_type* B, uint32_t rows,
                   uint32_t cols) {
    for (uint32_t i = 0; i < rows; i++)
        for (uint32_t j = 0; j < cols; j++)
            B[j * rows + i] = A[i * cols + j];
}

/*!
 * C_{dim0xdim2} = A_{dim0xdim1} * B_{dim1xdim2}
 */
template <class value_type>
void mat_mult_non_square(const value_type* A, const value_type* B,
                         value_type* C, uint8_t dim0, uint32_t dim1,
                         uint32_t dim2) {
    memset(C, 0, dim0 * dim2 * sizeof(value_type));
    for (uint32_t k = 0; k < dim1; k++)
        for (uint32_t i = 0; i < dim0; i++)
            for (uint32_t j = 0; j < dim2; j++)
                C[i * dim2 + j] += A[i * dim1 + k] * B[k * dim2 + j];
}

/*!
 * A^{+}_{nxm} = (A^TA)^{-1}A^T
 * where n = rows, m = cols.
 *
 * result will be stored back to A
 *
 * @param A sizeof rows*cols
 * @param buf sizeof (rows + cols + cols) * cols
 */
template <class value_type>
void pseudo_inverse_mat(value_type* A, uint32_t rows, uint32_t cols,
                        value_type* buf) {
    uint32_t &n = rows, &m = cols;

    value_type *B = buf,                       // m x n, A^T
            *C = buf + n * m,                  // m x m, (A^TA)
                    *D = buf + n * m + m * m;  // m x m, (A^TA)^{-1}

    transpose_mat(A, B, n, m);
    mat_mult_non_square(B, A, C, m, n, m);
    inverse_mat(C, D, m);
    mat_mult_non_square(D, B, A, m, m, n);
}

/*!
 * solve linear system Ax=b with squre-loss using pseudo inverse matrix.
 *
 * @param A  rows x cols, will be altered
 * @param b  rows x 1
 * @param x  cols x 1
 * @param buf buffer used by pseudo_inverse_mat. see doc for pseudo_inverse_mat
 * for detail.
 */
template <class value_type>
void solve_pseudo(value_type* A, uint32_t rows, uint32_t cols,
                  const value_type* b, value_type* x, value_type* buf) {
    pseudo_inverse_mat(A, rows, cols, buf);
    // A is actual A^{+} now
    mat_mult_non_square(A, b, x, cols, rows, 1);
}

}  // namespace linalg
}  // namespace megdnn

// vim: syntax=cpp.doxygen
