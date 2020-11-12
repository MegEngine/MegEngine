/**
 * \file dnn/src/cuda/conv_bias/helper.h
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
#include "src/cuda/handle.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/common/utils.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace cuda {

class ConvBiasDesc {
public:
    ConvBiasDesc();
    void set_conv_bias(DType data_type, const param::ConvBias& param,
                       const size_t nr_group);
    void set_conv(DType data_type, const param::ConvBias& param,
                  const size_t nr_group);
    ~ConvBiasDesc();
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t act_desc;
};

namespace conv_bias {
    using CanonizedFilterMeta = ConvBiasForward::CanonizedFilterMeta;

    //! conv size descriptor in the forward view
    struct BiasForwardSizeArgs {
        HandleImpl *handle;
        const TensorLayout *src_layout;
        const TensorLayout *filter_layout;
        const TensorLayout *bias_layout;
        const TensorLayout *z_layout;
        CanonizedFilterMeta filter_meta;
        const TensorLayout *dst_layout;
        param::ConvBias::NonlineMode nonlinear_mode;
    };

    //! whether cudnn is supported for a filter meta
    bool is_cudnn_supported(const BiasForwardSizeArgs& args);

    //! get workspace bundle for matmul algo
    WorkspaceBundle matmul_get_workspace_bundle(
            const BiasForwardSizeArgs& args);

    /*!
     * \brief flip conv filter
     *
     * Flip conv filter pointed by \p raw_ptr, store result in workspace, and
     * change \p raw_ptr to workspace.
     */
    void flip_filter(const BiasForwardSizeArgs& args,
                     const Workspace& workspace, void*& raw_ptr);

    struct CUDNNForwardDescs {
        TensorDesc src_desc, dst_desc, bias_desc, z_desc;
        FilterDesc<param::ConvBias> filter_desc;
        ConvBiasDesc conv_desc;

        void set_conv_bias(const TensorLayout& src,
                           const CanonizedFilterMeta& filter,
                           const TensorLayout& dst, const TensorLayout& bias,
                           const TensorLayout& z,
                           const param::ConvBias& param) {
            using Format = param::ConvBias::Format;
            Format src_format, dst_format;
            src_format = dst_format = param.format;
            if (param.format == Format::NCHW4_NCHW) {
                src_format = Format::NCHW4;
                dst_format = Format::NCHW;
            }
            src_desc.set(src, src_format);
            filter_desc.set(filter);
            if (z.ndim > 0) {
                z_desc.set(z, dst_format);
            }
            dst_desc.set(dst, dst_format);
            conv_desc.set_conv_bias(src.dtype, param, filter.group);

            // cudnn requires the bias to be float tensor.
            auto float_bias_layout = bias;
            float_bias_layout.dtype = dtype::Float32();
            if (param.format == param::ConvBias::Format::NCHW4 ||
                param.format == param::ConvBias::Format::NCHW32) {
                // cudnn require bias to be NCHW, not NCHW4.
                float_bias_layout = float_bias_layout.reshape(
                        {float_bias_layout[0],
                         float_bias_layout[1] * float_bias_layout[4],
                         float_bias_layout[2], float_bias_layout[3]});
                bias_desc.set(float_bias_layout);
            } else if (param.format == param::ConvBias::Format::NCHW4_NCHW) {
                megdnn_assert(float_bias_layout.ndim == 4,
                              "NCHW4_NCHW format assumes bias tensor is stored "
                              "in NCHW layout, ndim(expected:4,got:%zu)",
                              float_bias_layout.ndim);
                bias_desc.set(float_bias_layout);
            } else {
                bias_desc.set(float_bias_layout, param.format);
            }
        }

        void set_conv(const TensorLayout& src,
                      const CanonizedFilterMeta& filter,
                      const TensorLayout& dst, const param::ConvBias& param) {
            using Format = param::ConvBias::Format;
            Format src_format, dst_format;
            src_format = dst_format = param.format;
            if (param.format == Format::NCHW4_NCHW) {
                src_format = Format::NCHW4;
                dst_format = Format::NCHW;
            }
            src_desc.set(src, src_format);
            filter_desc.set(filter);
            dst_desc.set(dst, dst_format);
            conv_desc.set_conv(src.dtype, param, filter.group);
        }
    };

    bool check_bias_share_in_channel(const TensorLayout& bias,
                                     const param::ConvBias::Format format);

}  // namespace conv_bias
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
