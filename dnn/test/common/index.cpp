/**
 * \file dnn/test/common/index.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/index.h"

#include "test/common/utils.h"

namespace megdnn {
namespace test {

Index::Index(TensorLayout layout, size_t linear):
    m_layout(layout),
    m_linear(linear)
{
    linear_to_array();
    array_to_offset();
}

Index::Index(TensorLayout layout, TensorShape array):
    m_layout(layout),
    m_array(array)
{
    array_to_linear();
    array_to_offset();
}

void Index::linear_to_array()
{
    auto linear = m_linear;
    auto &array = m_array;
    array.ndim = m_layout.ndim;
    for (size_t j = m_layout.ndim; j > 0; --j) {
        size_t i = j-1;
        array[i] = linear % m_layout[i];
        linear /= m_layout[i];
    }
    megdnn_assert(linear == 0);
}

void Index::array_to_linear()
{
    auto &linear = m_linear;
    megdnn_assert(m_array.ndim == m_layout.ndim);
    linear = 0;
    for (size_t i = 0; i < m_array.ndim; ++i) {
        megdnn_assert(m_array[i] < m_layout[i]);
        linear = linear * m_layout[i] + m_array[i];
    }
}

void Index::array_to_offset()
{
    auto &offset = m_offset;
    megdnn_assert(m_array.ndim == m_layout.ndim);
    offset = 0;
    for (size_t i = 0; i < m_array.ndim; ++i) {
        megdnn_assert(m_array[i] < m_layout[i]);
        offset += m_array[i] * m_layout.stride[i];
    }
}

std::string Index::to_string() const
{
    std::string res = "";
    res.append("{");
    res.append("array=");
    res.append(m_array.to_string());
    res.append(",linear=");
    res.append(std::to_string(m_linear));
    res.append(",offset=");
    res.append(std::to_string(m_offset));
    res.append("}");
    return res;
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
