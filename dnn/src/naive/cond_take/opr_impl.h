/**
 * \file dnn/src/naive/cond_take/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class CondTakeImpl: public CondTake {
    template<typename ctype>
    void dispatch_genidx(size_t size, dt_int32 *dest, const ctype *inp);

    public:
        using CondTake::CondTake;

        size_t get_workspace_in_bytes(const TensorLayout& data) override;

        Output exec(
                _megdnn_tensor_in data, _megdnn_tensor_in mask,
                _megdnn_workspace workspace,
                DynOutMallocPolicyCall malloc_policy) override;
};

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen
