/**
 * \file dnn/src/cuda/lrn/opr_impl.h
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

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class LRNForwardImpl final: public LRNForward {
    public:
        using LRNForward::LRNForward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &) override {
            return 0;
        }
    private:
        TensorDesc src_desc, dst_desc;
        LRNDesc lrn_desc;
        void setup_descs(const TensorLayout &src, const TensorLayout &dst);
};

class LRNBackwardImpl final: public LRNBackward {
    public:
        using LRNBackward::LRNBackward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in dst,
                _megdnn_tensor_in diff,
                _megdnn_tensor_out grad,
                _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override {
            return 0;
        }
    private:
        TensorDesc src_desc, dst_desc, diff_desc, grad_desc;
        LRNDesc lrn_desc;
        void setup_descs(const TensorLayout &src,
                const TensorLayout &dst,
                const TensorLayout &diff,
                const TensorLayout &grad);
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
