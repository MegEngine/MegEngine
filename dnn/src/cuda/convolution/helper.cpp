/**
 * \file dnn/src/cuda/convolution/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

bool convolution::is_cudnn_supported(const ForwardSizeArgs &args) {
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }

    // CUDNN_STATUS_EXECUTION_FAILED on Tegra K1, so disable CUDNN
    // on Tegra K1.
    if (args.handle->is_tegra_k1())
        return false;

    // TODO: We only support NCHW format now. It seems cuDNN provides support
    // for NHWC as well.
    if (args.filter_meta.format == param::Convolution::Format::NCHW4) {
        if (args.dst_layout->dtype.enumv() != DTypeEnum::Int8 &&
                args.dst_layout->dtype.enumv() != DTypeEnum::QuantizedS8) {
            return false;
        }
    } else if (args.filter_meta.format != param::Convolution::Format::NCHW) {
        return false;
    }
    auto& fm = args.filter_meta;
    bool supported = true;
    supported &= (fm.spatial_ndim == 2);
#if CUDNN_VERSION < 7000
    supported &= (fm.group == 1);
#endif
#if CUDNN_VERSION < 7500
    supported &= (fm.dilation[0] == 1 && fm.dilation[1] == 1);
#endif
    return supported;
}

WorkspaceBundle convolution::matmul_get_workspace_bundle(
        const ForwardSizeArgs &args) {
    auto dtype = args.src_layout->dtype;
    auto &&fm = args.filter_meta;
    megdnn_assert(fm.group == 1);
    auto N = args.src_layout->shape[0];
    auto OC = fm.ocpg,
         IC = fm.icpg,
         FH = fm.spatial[0],
         FW = fm.spatial[1];
    auto OH = args.dst_layout->shape[2],
         OW = args.dst_layout->shape[3];
    SmallVector<size_t> sizes{
            dtype.size() * args.dst_layout->total_nr_elems(),
            dtype.size() * IC*FH*FW*OH*OW*N
    };
    if (args.filter_meta.should_flip) {
        sizes.push_back(dtype.size() * OC * IC * FH * FW);
    }
    return {nullptr, std::move(sizes)};
}

void convolution::flip_filter(const ForwardSizeArgs &args,
        const Workspace &workspace, void *&raw_ptr) {
    auto &&fm = args.filter_meta;
    megdnn_assert(fm.group == 1 && fm.spatial_ndim == 2);
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto dtype = fm.dtype;
    megdnn_assert(workspace.size >= dtype.size() * OC * IC * FH * FW);

    TensorND src{raw_ptr, {{OC, IC, FH, FW}, dtype}},
             dst{workspace.raw_ptr + (FH * FW - 1) * dtype.size(), src.layout};
    dst.layout.stride[2] = -dst.layout.stride[2];
    dst.layout.stride[3] = -dst.layout.stride[3];
    args.handle->relayout_opr()->exec(src, dst);
    raw_ptr = workspace.raw_ptr;
}

// vim: syntax=cpp.doxygen
