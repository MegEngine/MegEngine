/**
 * \file dnn/src/naive/indexing_multi_axis_vec/opr_impl.h
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

    class IndexingMultiAxisVecImpl final: public IndexingMultiAxisVec {
        public:
            using IndexingMultiAxisVec::IndexingMultiAxisVec;

            size_t get_workspace_in_bytes(size_t) override {
                return 0;
            }

            void exec(_megdnn_tensor_in src, const IndexDesc &index,
                    _megdnn_tensor_out dst,
                    _megdnn_workspace workspace) override;
    };

    class IndexingSetMultiAxisVecImpl final: public IndexingSetMultiAxisVec {
        public:
            using IndexingSetMultiAxisVec::IndexingSetMultiAxisVec;

            size_t get_workspace_in_bytes(size_t) override {
                return 0;
            }

            void exec(_megdnn_tensor_in data, _megdnn_tensor_out value,
                    const IndexDesc &index,
                    _megdnn_workspace workspace) override;
    };

    class IndexingIncrMultiAxisVecImpl final: public IndexingIncrMultiAxisVec {
        public:
            using IndexingIncrMultiAxisVec::IndexingIncrMultiAxisVec;

            size_t get_workspace_in_bytes(size_t) override {
                return 0;
            }

            void exec(_megdnn_tensor_in data, _megdnn_tensor_out value,
                    const IndexDesc &index,
                    _megdnn_workspace workspace) override;
    };
}
}

// vim: syntax=cpp.doxygen

