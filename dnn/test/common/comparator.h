/**
 * \file dnn/test/common/comparator.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

namespace megdnn {
namespace test {

template <typename T>
class DefaultComparator {
public:
    bool is_same(T expected, T actual) const;
};

}  // namespace test
}  // namespace megdnn

#include "test/common/comparator.inl"

// vim: syntax=cpp.doxygen
