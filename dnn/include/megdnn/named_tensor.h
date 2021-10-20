/**
 * \file dnn/include/megdnn/named_tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megdnn/internal/defs.h"
#include "megdnn/opr_param_defs.h"

#include <array>
#include <string>
#include "megdnn/thin/small_vector.h"

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {

class Dimension {
public:
    enum class Name : char {
        N = 'N',  // Batch size
        C = 'C',  // input channel
        H = 'H',  // input height
        W = 'W',  // input width
        G = 'G',  // group
        K = 'K',  // output channel
        R = 'R',  // filter height
        S = 'S',  // filter width
        P = 'P',  // output height
        Q = 'Q',  // output width
    };
    static constexpr uint32_t UNDETERMINED_EXTENT =
            std::numeric_limits<uint32_t>::max();
    static const Name NAME_ALL[];
    static const int NR_NAMES;
    Dimension() = default;
    Dimension(const std::string& expr);
    Dimension(Name name, uint32_t stride, uint32_t extent = UNDETERMINED_EXTENT)
            : m_name{name}, m_stride{stride}, m_extent{extent} {}
    Dimension(const Dimension& rhs) { operator=(rhs); }
    Dimension& operator=(const Dimension& rhs);
    bool operator==(const Dimension& rhs) const;
    bool operator<(const Dimension& rhs) const;
    Dimension operator*(const Dimension& rhs) const;
    Dimension operator/(const Dimension& rhs) const;
    std::string to_string() const;
    Name name() const { return m_name; }
    uint32_t extent() const { return m_extent; }
    uint32_t stride() const { return m_stride; }

private:
    Name m_name;
    uint32_t m_stride;
    uint32_t m_extent;
};

struct NamedTensorShape {
    using Format = param::ConvBias::Format;
    static constexpr size_t MAX_NDIM = MEGDNN_MAX_NDIM;

    std::array<Dimension, MAX_NDIM> dims;
    size_t ndim = 0;

    NamedTensorShape() = default;
    NamedTensorShape(const NamedTensorShape& rhs) = default;
    NamedTensorShape(const SmallVector<Dimension>& init_shape);
    NamedTensorShape(std::initializer_list<Dimension> init_shape);
    std::string to_string() const;

    bool eq_shape(const NamedTensorShape& rhs) const;
    Dimension& operator[](size_t i) { return dims[i]; }
    Dimension operator[](size_t i) const { return dims[i]; }
    NamedTensorShape static make_named_tensor_shape(Format format);
};
}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
