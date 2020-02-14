/**
 * \file dnn/src/naive/topk/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace naive {

class TopKImpl : public TopK {
protected:
    template <typename ctype>
    void dispatch_with_ctype(int k, size_t m, size_t n, ptrdiff_t lda,
                             const ctype* data, ctype* values, int* indices,
                             void* workspace);

    void do_exec(int k, _megdnn_tensor_in data, _megdnn_tensor_out values,
                 int32_t* indices, _megdnn_workspace workspace) override;

public:
    using TopK::TopK;

    size_t get_workspace_in_bytes(int k, const TensorLayout& data,
                                  const TensorLayout& values,
                                  const TensorLayout& indices) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
