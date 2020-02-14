/**
 * Copyright (c) 2018, Alibaba Group Holding Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *----------------------------------------------------------------------------
 *
 * \file dnn/src/common/winograd/winograd_generator.cpp
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 *
 * ----------------------------------------------------------------
 */

#include "src/common/winograd/winograd_generator.h"
#include "src/common/utils.h"
#include <stdio.h>
#include <cmath>
#include <cstring>

using namespace megdnn;
using namespace winograd;

namespace {

WinogradGenerator::Matrix computeA(const std::vector<float>& a, int m, int n) {
    WinogradGenerator::Matrix res(n, m);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < m - 1; ++x) {
            if (x == 0 && y == 0) {
                res.at(y, x) = 1.0f;
            } else {
                res.at(y, x) = ::powf(a[x], (float)y);
            }
        }
        if (y == n - 1) {
            res.at(y, m - 1) = 1.0f;
        } else {
            res.at(y, m - 1) = 0.0f;
        }
    }
    return res;
}

WinogradGenerator::Matrix computeF(const std::vector<float>& a, int alpha) {
    WinogradGenerator::Matrix res(1, alpha);
    for (int x = 0; x < alpha; ++x) {
        float product = 1.0f;
        for (int i = 0; i < alpha; ++i) {
            if (x == i) {
                continue;
            }
            product *= (a[x] - a[i]);
        }
        res.at(0, x) = product;
    }
    return res;
}

WinogradGenerator::Matrix computeT(const std::vector<float>& a, int n) {
    WinogradGenerator::Matrix res(n, n + 1);
    for (int y = 0; y < n; ++y) {
        auto line = res.data() + res.cols() * y;
        std::memset(line, 0, res.cols() * sizeof(float));
        line[y] = 1.0f;
        line[n] = -::powf(a[y], (float)n);
    }
    return res;
}

WinogradGenerator::Matrix computeL(const std::vector<float>& a, int n) {
    megdnn_assert(n >= 1);
    WinogradGenerator::Matrix res(n, n);
    for (int k = 0; k < n; ++k) {
        WinogradGenerator::Matrix p(1, 1);
        p.at(0, 0) = 1.0f;
        WinogradGenerator::Matrix p2(1, 2);
        for (int i = 0; i < n; ++i) {
            if (i == k) {
                continue;
            }
            p2.at(0, 0) = -a[i];
            p2.at(0, 1) = 1.0f;
            p = p.poly_multi(p2);
        }
        std::memcpy(res.data() + res.cols() * k, p.data(), n * sizeof(float));
    }
    return res;
}

WinogradGenerator::Matrix computeB(const std::vector<float>& a, int alpha) {
    WinogradGenerator::Matrix res;
    auto L = computeL(a, alpha - 1);
    auto fdiag = computeF(a, alpha - 1);
    L.div_per_line(fdiag);

    L.transpose();

    auto T = computeT(a, alpha - 1);
    WinogradGenerator::Matrix BT = L.mul(T);

    WinogradGenerator::Matrix B(alpha, alpha);
    for (int y = 0; y < alpha - 1; ++y) {
        std::memcpy(B.data() + B.cols() * y, BT.data() + BT.cols() * y,
                    alpha * sizeof(float));
    }
    for (int x = 0; x < alpha - 1; ++x) {
        B.at(alpha - 1, x) = 0;
    }
    B.at(alpha - 1, alpha - 1) = 1.0f;

    return B;
}

WinogradGenerator::Matrix computeFPlusOne(const std::vector<float>& a,
                                          int alpha) {
    auto fdiag = computeF(a, alpha - 1);
    WinogradGenerator::Matrix res(1, alpha);
    for (int i = 0; i < alpha - 1; i++) {
        res.at(0, i) = fdiag.at(0, i);
    }
    res.at(0, alpha - 1) = 1;
    //! change sign if res[0, 0] < 0
    res.at(0, 0) = std::abs(res.at(0, 0));

    return res;
}

}  // namespace

float& WinogradGenerator::Matrix::at(size_t row, size_t col) {
    return m_data[row * m_cols + col];
}

const float& WinogradGenerator::Matrix::at(size_t row, size_t col) const {
    return m_data[row * m_cols + col];
}

void WinogradGenerator::Matrix::transpose() {
    WinogradGenerator::Matrix res(m_cols, m_rows);
    for (size_t r = 0; r < m_rows; r++) {
        for (size_t c = 0; c < m_cols; c++) {
            res.at(c, r) = m_data[r * m_cols + c];
        }
    }
    *this = std::move(res);
}

void WinogradGenerator::Matrix::print(const char* msg) const {
    printf("%s\n", msg);

    for (size_t y = 0; y < m_rows; ++y) {
        for (size_t x = 0; x < m_cols; ++x) {
            printf("%.7f\t", at(y, x));
        }
        printf("\n");
    }
}

WinogradGenerator::Matrix WinogradGenerator::Matrix::mul(const Matrix& rhs) {
    WinogradGenerator::Matrix res(rows(), rhs.cols());
    for (size_t r = 0; r < res.rows(); r++) {
        for (size_t c = 0; c < res.cols(); c++) {
            res.at(r, c) = 0.f;
            for (size_t k = 0; k < cols(); k++) {
                res.at(r, c) += at(r, k) * rhs.at(k, c);
            }
        }
    }
    std::swap(m_rows, m_cols);
    return res;
}

WinogradGenerator::Matrix WinogradGenerator::Matrix::poly_multi(
        const Matrix& B) {
    megdnn_assert(rows() == 1 && B.rows() == 1);
    auto aw = cols();
    auto bw = B.cols();

    WinogradGenerator::Matrix res(1, aw + bw - 1);

    for (size_t i = 0; i < aw + bw - 1; ++i) {
        res.at(0, i) = 0.0f;
    }
    for (size_t y = 0; y < bw; ++y) {
        auto bValue = B.at(0, y);
        for (size_t x = 0; x < aw; ++x) {
            auto aValue = this->at(0, x);
            res.at(0, x + y) += bValue * aValue;
        }
    }
    return res;
}

void WinogradGenerator::Matrix::div_per_line(
        const WinogradGenerator::Matrix& line) {
    megdnn_assert(line.rows() == 1 && line.cols() >= m_rows);

    for (size_t y = 0; y < m_rows; ++y) {
        for (size_t x = 0; x < m_cols; ++x) {
            at(y, x) /= line.at(0, y);
        }
    }
}

void WinogradGenerator::Matrix::mul_per_row(
        const WinogradGenerator::Matrix& line) {
    megdnn_assert(line.rows() == 1 && line.cols() >= m_cols);
    for (size_t y = 0; y < m_rows; ++y) {
        for (size_t x = 0; x < m_cols; ++x) {
            at(y, x) *= line.at(0, x);
        }
    }
}



WinogradGenerator::WinogradGenerator(size_t m, size_t r, float interp) {
    size_t alpha = m + r - 1;

    std::vector<float> a(alpha);
    a[0] = 0.0f;
    int sign = 1;
    for (size_t i = 0; i < alpha - 1; ++i) {
        int value = 1 + i / 2;
        a[i + 1] = sign * value * interp;
        sign *= -1;
    }

    generate(m, r, a);
}

WinogradGenerator::WinogradGenerator(size_t m, size_t r,
                                     const std::vector<float>& interp_points) {
    megdnn_assert(interp_points.size() == m + r - 2,
                  "interp_points should be %zu, but got: %zu", m + r - 2,
                  interp_points.size());

    generate(m, r, interp_points);
}

void WinogradGenerator::generate(size_t m, size_t r,
                                 const std::vector<float>& interp_points) {
    size_t alpha = m + r - 1;
    m_A = computeA(interp_points, alpha, m);
    m_A.transpose();

    auto fdiag = computeFPlusOne(interp_points, alpha);

    m_G = computeA(interp_points, alpha, r);
    m_G.transpose();
    m_G.div_per_line(fdiag);

    m_B = computeB(interp_points, alpha);
    m_B.mul_per_row(fdiag);
}

// vim: syntax=cpp.doxygen
