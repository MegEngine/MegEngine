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
                                          const TensorLayout& offsets,
                                          const TensorLayout& parts) {
    megdnn_assert(offsets.dtype == dtype::Int32{}, "bad dtype: %s",
                  offsets.dtype.name());
    megdnn_assert(concated.ndim == 1 && offsets.ndim == 1 && parts.ndim == 1 &&
                          concated.stride[0] == 1 && offsets.stride[0] == 1 &&
                          parts.stride[0] == 1,
                  "bad layout: concated=%s offsets=%s parts=%s",
                  concated.to_string().c_str(), offsets.to_string().c_str(),
                  parts.to_string().c_str());
}

std::vector<dt_int32> ParamPackConcatSplitBase::gen_offsets(
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

    std::vector<dt_int32> offsets(shapes.size() << 1);
    size_t offset = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        offset = get_aligned(offset);
        offsets[i << 1] = offset;
        offset += shapes[i].total_nr_elems();
        offsets[(i << 1) + 1] = offset;
    }
    return offsets;
}

// vim: syntax=cpp.doxygen
