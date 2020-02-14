/**
 * \file dnn/test/naive/rng.h
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
#include <utility>
#include "megdnn/dtype/half.hpp"

namespace half_float {
static inline std::ostream& operator<<(std::ostream& stream, half v) {
    return stream << static_cast<float>(v);
}
}  // namespace half_float

namespace megdnn {
namespace test {

//! get mean and variance
template <typename ctype>
std::pair<double, double> get_mean_var(const ctype* src, size_t size,
                                       ctype expected_mean) {
    double sum = 0, sum2 = 0;
    for (size_t i = 0; i < size; ++i) {
        auto cur = src[i] - expected_mean;
        sum += cur;
        sum2 += cur * cur;
    }
    double mean = sum / size;
    return {mean + expected_mean, sum2 / size - mean * mean};
}

template <typename ctype>
void assert_uniform_correct(const ctype* src, size_t size);

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
