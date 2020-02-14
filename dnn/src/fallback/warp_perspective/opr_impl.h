/**
 * \file dnn/src/fallback/warp_perspective/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/warp_perspective/opr_impl.h"

namespace megdnn {
namespace fallback {

class WarpPerspectiveImpl: public naive::WarpPerspectiveForwardImpl {
    protected:
        template <typename ctype, typename mtype>
        void kern_fallback(const KernParam<ctype, mtype> &kern_param);
    public:
        using naive::WarpPerspectiveForwardImpl::WarpPerspectiveForwardImpl;
        size_t get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &mat,
                const TensorLayout &mat_idx,
                const TensorLayout &dst) override;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in mat,
                _megdnn_tensor_in mat_idx,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
    private:
        template <typename ctype>
        bool is_resize_optimizable(ctype *mat);
        template <bool is_border_constant, typename ctype, typename mtype>
        void kern_resize(const KernParam<ctype, mtype>& kern_param);
};

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
