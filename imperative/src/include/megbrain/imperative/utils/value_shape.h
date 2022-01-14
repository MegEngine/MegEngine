/**
 * \file imperative/src/include/megbrain/imperative/utils/visit.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <vector>

#include "megbrain/imperative/utils/span.h"
#include "megbrain/tensor.h"

namespace mgb::imperative {

/**
 * \brief like TensorShape, but allow real scalar shape.
 *
 */
struct ValueShape {
    size_t shape[TensorShape::MAX_NDIM];
    int ndim = 0;

    ValueShape() = default;
    ValueShape(std::initializer_list<size_t> dims) {
        for (auto&& dim : dims) {
            shape[ndim++] = dim;
        }
    }
    ValueShape(Span<size_t> dims) {
        for (auto&& dim : dims) {
            shape[ndim++] = dim;
        }
    }

    size_t& operator[](int axis) { return shape[axis]; }

    size_t operator[](int axis) const { return shape[axis]; }

    size_t at(int axis) const {
        mgb_assert(axis < ndim);
        return shape[axis];
    }

    size_t total_nr_elems() const {
        size_t prod = 1;
        for (int i = 0; i < ndim; ++i) {
            prod *= shape[i];
        }
        return prod;
    }

    bool is_scalar() const { return ndim == 0; }

    std::string to_string() const {
        std::string buffer = "{";
        for (size_t i = 0; i < ndim; ++i) {
            if (i) {
                buffer.append(",");
            }
            buffer.append(std::to_string(shape[i]));
        }
        buffer.append("}");
        return buffer;
    }

    static ValueShape from(TensorShape tensor_shape) {
        mgb_assert(tensor_shape.ndim);
        return Span<size_t>{tensor_shape.shape, tensor_shape.ndim};
    }

    TensorShape as_tensor_shape() const {
        mgb_assert(ndim != 0);
        TensorShape ret;
        for (size_t i = 0; i < ndim; ++i) {
            ret.shape[i] = shape[i];
        }
        ret.ndim = ndim;
        return ret;
    }

    bool operator==(const ValueShape& rhs) const {
        if (ndim != rhs.ndim) {
            return false;
        }
        for (size_t i = 0; i < ndim; ++i) {
            if (shape[i] != rhs.shape[i]) {
                return false;
            }
        }
        return true;
    }
};

static_assert(sizeof(size_t) >= sizeof(int));
static_assert(TensorShape::MAX_NDIM == 7);
static_assert(sizeof(ValueShape) <= sizeof(size_t) * 8);

}  // namespace mgb::imperative