/**
 * \file dnn/src/common/param_pack.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"
#include "src/common/utils.h"

using namespace megdnn;

void ParamPackConcatSplitBase::check_exec(const TensorLayout& concated,
                                          const TensorLayout& table,
                                          const TensorLayout& parts) {
    megdnn_assert(table.dtype == dtype::Int32{}, "bad dtype: %s",
                  table.dtype.name());
    megdnn_assert(concated.ndim == 1 && table.ndim == 1 && parts.ndim == 1 &&
                          concated.stride[0] == 1 && table.stride[0] == 1 &&
                          parts.stride[0] == 1,
                  "bad layout: concated=%s table=%s parts=%s",
                  concated.to_string().c_str(), table.to_string().c_str(),
                  parts.to_string().c_str());
    megdnn_assert(table.shape[0] == concated.shape[0] * 2,
                  "concated=%zu table=%zu", concated.shape[0], table.shape[0]);
}

std::vector<dt_int32> ParamPackConcatSplitBase::gen_table(
        const TensorShapeArray& shapes, size_t alignment, size_t dtype_size) {
    megdnn_assert(alignment && (alignment & (alignment - 1)) == 0,
                  "alignment must be power of 2: %zu", alignment);
    if (alignment < dtype_size)
        alignment = dtype_size;

    megdnn_assert(alignment % dtype_size == 0,
                  "alignment must be multiple of dtype size: %zu vs %zu",
                  alignment, dtype_size);
    alignment /= dtype_size;

    auto get_aligned = [alignment](size_t v) {
        auto mod = v & (alignment - 1);
        return v + ((alignment - mod) & (alignment - 1));
    };

    size_t offset = 0;
    for (auto&& i : shapes) {
        offset = get_aligned(offset) + i.total_nr_elems();
    }

    std::vector<dt_int32> table(offset * 2);
    auto outer_table = table.data(), inner_table = outer_table + offset;

    offset = 0;
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto aligned = get_aligned(offset);
        for (size_t j = offset; j < aligned; ++j) {
            inner_table[j] = outer_table[j] = -1;
        }
        offset = aligned;
        auto cur_size = shapes[i].total_nr_elems();
        for (size_t j = 0; j < cur_size; ++j) {
            outer_table[offset + j] = i;
            inner_table[offset + j] = j;
        }
        offset += cur_size;
    }
    megdnn_assert(offset * 2 == table.size());
    return table;
}

// vim: syntax=cpp.doxygen
