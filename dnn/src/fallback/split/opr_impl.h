/**
 * \file dnn/src/fallback/split/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/split/opr_impl.h"

namespace megdnn {
namespace fallback {

class SplitImpl: public naive::SplitForwardImpl {
    public:
        using SplitForwardImpl::SplitForwardImpl;
        void exec(_megdnn_tensor_in src,
                _megdnn_out const TensorNDArray &dsts,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayoutArray &dsts) override
        {
            return sizeof(size_t) * dsts.size();
        }
};

} // namespace fallback
} // namespace megdnn
// vim: syntax=cpp.doxygen



