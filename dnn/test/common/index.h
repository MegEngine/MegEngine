/**
 * \file dnn/test/common/index.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {
/**
 * array: index in the array form
 * linear: a single index number by assuming contiguous layout
 * offset: the memory offset in nr elements (can be negative)
 *
 * dtype is ignored.
 */
class Index {
public:
    Index(TensorLayout layout, size_t linear);
    Index(TensorLayout layout, TensorShape array);

    std::string to_string() const;

    TensorShape array() const { return m_array; }
    TensorLayout layout() const { return m_layout; }
    size_t linear_index() const { return m_linear; }
    ptrdiff_t offset() const { return m_offset; }
    /**
     * Add a universal offset to all return values to make the minimal
     * offset zero.
     */
    size_t positive_offset() const {
        return m_offset - m_layout.span().low_elem;
    }

private:
    TensorLayout m_layout;
    size_t m_linear;
    TensorShape m_array;
    ptrdiff_t m_offset;

    void linear_to_array();
    void array_to_linear();
    void array_to_offset();
};

class IndexRNG final : public RNG {
    size_t& m_size;
    std::mt19937_64 m_rng;

public:
    IndexRNG(size_t& sz, size_t seed) : m_size{sz}, m_rng(seed) {}

    void gen(const TensorND& tensor) override {
        std::uniform_int_distribution<int> dist(-static_cast<int>(m_size),
                                                m_size - 1);
        auto ptr = tensor.ptr<int>() + tensor.layout.span().low_elem;
        for (size_t i = 0; i < tensor.layout.span().dist_elem(); ++i)
            ptr[i] = dist(m_rng);
    }
};
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
