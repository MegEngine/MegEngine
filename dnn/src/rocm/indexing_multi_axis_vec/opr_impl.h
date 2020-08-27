/**
 * \file dnn/src/rocm/indexing_multi_axis_vec/opr_impl.h
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
namespace rocm {

    class IndexingMultiAxisVecImpl final: public IndexingMultiAxisVec {
        void* m_error_tracker = nullptr;

        public:
            using IndexingMultiAxisVec::IndexingMultiAxisVec;

            size_t get_workspace_in_bytes(size_t dst_idx_size) override;

            void exec(_megdnn_tensor_in src, const IndexDesc &index,
                    _megdnn_tensor_out dst,
                    _megdnn_workspace workspace) override;

            void set_error_tracker(void* tracker) override {
                m_error_tracker = tracker;
            }
    };

    class IndexingSetMultiAxisVecImpl final: public IndexingSetMultiAxisVec {
        void* m_error_tracker = nullptr;

        public:
            using IndexingSetMultiAxisVec::IndexingSetMultiAxisVec;

            size_t get_workspace_in_bytes(size_t dst_idx_size) override;

            void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
                    const IndexDesc &index,
                    _megdnn_workspace workspace) override;

            void set_error_tracker(void* tracker) override {
                m_error_tracker = tracker;
            }
    };

    class IndexingIncrMultiAxisVecImpl final: public IndexingIncrMultiAxisVec {
        void* m_error_tracker = nullptr;

        public:
            using IndexingIncrMultiAxisVec::IndexingIncrMultiAxisVec;

            size_t get_workspace_in_bytes(size_t dst_idx_size) override;

            void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
                    const IndexDesc &index,
                    _megdnn_workspace workspace) override;

            void set_error_tracker(void* tracker) override {
                m_error_tracker = tracker;
            }
    };
}
}

// vim: syntax=cpp.doxygen

