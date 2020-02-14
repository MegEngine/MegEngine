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
 * ----------------------------------------------------------------
 *
 * \file dnn/src/common/winograd/winograd_generator.h
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

#pragma once
#include <vector>
#include <cstddef>
#include <memory>
#include "src/common/utils.h"

namespace megdnn {
namespace winograd {

/**
 * \brief generator winograd matrix, A/B/G
 */
class WinogradGenerator {
public:
    WinogradGenerator(size_t m, size_t r, float interp = 0.5f);
    WinogradGenerator(size_t m, size_t r,
                      const std::vector<float>& interp_points);
    ~WinogradGenerator() = default;

    class Matrix {
    public:
        Matrix(size_t rows, size_t cols) : m_rows{rows}, m_cols{cols} {
            m_data.resize(rows * cols);
        }
        Matrix() = default;
        Matrix(Matrix&& rhs) {
            m_data = std::move(rhs.m_data);
            m_rows = rhs.m_rows;
            m_cols = rhs.m_cols;
        }
        Matrix& operator=(Matrix&& rhs) {
            m_data = std::move(rhs.m_data);
            m_rows = rhs.m_rows;
            m_cols = rhs.m_cols;
            return *this;
        }

        Matrix(const Matrix& rhs) {
            m_data = rhs.m_data;
            m_rows = rhs.m_rows;
            m_cols = rhs.m_cols;
        }
        Matrix& operator=(const Matrix& rhs) {
            m_data = rhs.m_data;
            m_rows = rhs.m_rows;
            m_cols = rhs.m_cols;
            return *this;
        }

        size_t rows() const { return m_rows; }
        size_t cols() const { return m_cols; }

        float& at(size_t row, size_t col);
        const float& at(size_t row, size_t col) const;
        float* data() { return m_data.data(); }
        const float* data() const { return m_data.data(); }

        void transpose();
        void div_per_line(const Matrix& line);
        Matrix mul(const Matrix& rhs);
        void mul_per_row(const Matrix& line);
        Matrix poly_multi(const Matrix& rhs);
        void print(const char* msg) const;

    private:
        std::vector<float> m_data;
        size_t m_rows;
        size_t m_cols;
    };

    const Matrix& A() const { return m_A; }
    const Matrix& B() const { return m_B; }
    const Matrix& G() const { return m_G; }

private:
    void generate(size_t m, size_t r, const std::vector<float>& interp_points);
    Matrix m_A;
    Matrix m_G;
    Matrix m_B;
};

/////////////////////// WinogradCoeff ////////////////////////////
/**
 * \brief Contains the winograd coeff
 */
template <typename ctype>
class WinogradCoeff {
    std::unique_ptr<WinogradGenerator> m_generator;

    std::vector<ctype> generate(float rescale,
                                const WinogradGenerator::Matrix& m) {
        std::vector<ctype> ret;
        for (size_t r = 0; r < m.rows(); r++) {
            for (size_t c = 0; c < m.cols(); c++) {
                float val = m.at(r, c) * rescale;
                if (std::is_integral<ctype>::value) {
                    megdnn_assert(
                            std::abs(val - std::round(val)) < 1e-4,
                            "invalid rescale args, %f(item) * %f(rescale) is "
                            "not near %f\n",
                            m.at(r, c), rescale, std::round(val));
                    ret.push_back(static_cast<ctype>(std::round(val)));
                } else {
                    ret.push_back(static_cast<ctype>(val));
                }
            }
        }
        return ret;
    }

public:
    WinogradCoeff(size_t m, size_t r, const std::vector<float>& interp_points) {
        m_generator = std::make_unique<WinogradGenerator>(m, r, interp_points);
    }

    std::vector<ctype> A(float rescale) {
        return generate(rescale, m_generator->A());
    }

    std::vector<ctype> B(float rescale) {
        return generate(rescale, m_generator->B());
    }

    std::vector<ctype> G(float rescale) {
        return generate(rescale, m_generator->G());
    }
};

}  // namespace winograd
}  // namespace megdnn

// vim: syntax=cpp.doxygen
