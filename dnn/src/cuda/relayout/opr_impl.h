/**
 * \file dnn/src/cuda/relayout/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class RelayoutForwardImpl final : public RelayoutForward {
    class Param {
        TensorND m_src, m_dst;
        RelayoutForwardImpl* const m_opr;

    public:
        Param(const TensorND& src, const TensorND& dst, RelayoutForwardImpl* opr);

        size_t dtype_size() const { return m_src.layout.dtype.size(); }

        //! try to transpose
        bool try_transpose();

        //! try to copy by cudaMemcpy
        bool try_copy_contig();

        //! try to copy by cudaMemcpy2DAsync
        bool try_copy_2d(bool cross_dev);

        void copy_general();

        //! try to copy if last contiguous
        bool try_copy_last_contig();
    };

    //! expand *dst* to 2 dims to match *src*
    static bool expand_dim2(TensorLayout& dst, const TensorLayout& src);

    cudaStream_t stream() const { return cuda_stream(handle()); }

public:
    using RelayoutForward::RelayoutForward;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, Handle* src_handle) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
