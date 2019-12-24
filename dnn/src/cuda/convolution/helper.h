/**
 * \file dnn/src/cuda/convolution/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "./opr_impl.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/handle.h"
#include "src/common/utils.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace cuda {
namespace convolution {
    using CanonizedFilterMeta = ConvolutionForward::CanonizedFilterMeta;

    //! conv size descriptor in the forward view
    struct ForwardSizeArgs {
        HandleImpl *handle;
        const TensorLayout *src_layout;
        const TensorLayout *filter_layout;
        CanonizedFilterMeta filter_meta;
        const TensorLayout *dst_layout;
    };

    //! whether cudnn is supported for a filter meta
    bool is_cudnn_supported(const ForwardSizeArgs &args);

    //! get workspace bundle for matmul algo
    WorkspaceBundle matmul_get_workspace_bundle(const ForwardSizeArgs &args);

    struct CUDNNForwardDescs {
        TensorDesc src_desc, dst_desc;
        FilterDesc<param::Convolution> filter_desc;
        ConvDesc conv_desc;
        void set(const TensorLayout &src,
                const CanonizedFilterMeta &filter,
                const TensorLayout &dst,
                const param::Convolution &param)
        {
            src_desc.set(src, param.format);
            filter_desc.set(filter);
            dst_desc.set(dst, param.format);
            conv_desc.set(src.dtype, param, filter.group);
        }
    };

    struct CUDNNBwdDataDescs {
        TensorDesc diff_desc, grad_desc;
        FilterDesc<param::Convolution> filter_desc;
        ConvDesc conv_desc;
        void set(const CanonizedFilterMeta &filter,
                const TensorLayout &diff,
                const TensorLayout &grad,
                const param::Convolution &param)
        {
            filter_desc.set(filter);
            diff_desc.set(diff, param.format);
            grad_desc.set(grad, param.format);
            conv_desc.set(filter.dtype, param, filter.group);
        }
    };

    struct CUDNNBwdFilterDescs {
        TensorDesc diff_desc, src_desc;
        FilterDesc<param::Convolution> grad_desc;
        ConvDesc conv_desc;
        void set(const TensorLayout &src,
                const TensorLayout &diff,
                const CanonizedFilterMeta &grad,
                const param::Convolution &param)
        {
            src_desc.set(src, param.format);
            diff_desc.set(diff, param.format);
            grad_desc.set(grad);
            conv_desc.set(src.dtype, param, grad.group);
        }
    };

    /*!
     * \brief flip conv filter
     *
     * Flip conv filter pointed by \p raw_ptr, store result in workspace, and
     * change \p raw_ptr to workspace.
     */
    void flip_filter(const ForwardSizeArgs &args,
            const Workspace &workspace, void *&raw_ptr);

} // namespace convolution
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
