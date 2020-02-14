/**
 * \file dnn/test/common/comparator.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "test/common/comparator.h"
#include "test/common/utils.h"

#include "megdnn/dtype.h"

namespace megdnn {
namespace test {

template <typename T>
bool DefaultComparator<T>::is_same(T expected, T actual) const
{
    return expected == actual;
}

template <>
class DefaultComparator<dt_float32> {
    public:
        bool is_same(dt_float32 expected, dt_float32 actual) const
        {
            return std::abs(diff(expected, actual)) < 1e-3;
        }
};

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen

