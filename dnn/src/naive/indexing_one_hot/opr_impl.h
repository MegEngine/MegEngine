/**
 * \file dnn/src/naive/indexing_one_hot/opr_impl.h
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

    class IndexingOneHotForwardImpl final: public IndexingOneHotForward {
        public:
            using IndexingOneHotForward::IndexingOneHotForward;
            void exec(_megdnn_tensor_in src, _megdnn_tensor_in index,
                    _megdnn_tensor_out dst,
                    _megdnn_workspace workspace) override;
            size_t get_workspace_in_bytes(const TensorLayout &,
                    const TensorLayout &,
                    const TensorLayout &) override {
                return 0;
            }
    };

    class IndexingSetOneHotForwardImpl final: public IndexingSetOneHotForward {
        public:
            using IndexingSetOneHotForward::IndexingSetOneHotForward;
            void exec(_megdnn_tensor_inout data, _megdnn_tensor_in index,
                    _megdnn_tensor_in sub,
                    _megdnn_workspace workspace) override;
            size_t get_workspace_in_bytes(const TensorLayout &,
                    const TensorLayout &,
                    const TensorLayout &) override {
                return 0;
            }
    };
}
}

// vim: syntax=cpp.doxygen

