/**
 * \file dnn/src/aarch64/conv_bias/fp32/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/conv_bias/fp32/algos.h"
#include "src/aarch64/conv_bias/fp32/stride2_kern.h"
#include "src/arm_common/conv_bias/direct/multi_thread_common.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/fallback/conv_bias/common.h"

#include "midout.h"

using namespace megdnn;
using namespace aarch64;

MIDOUT_DECL(megdnn_aarch64_conv_bias_stride2_conv2357_fp32)
bool ConvBiasImpl::AlgoF32DirectStride2::usable(const NCBKernSizeParam& param,
                                                AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_aarch64_conv_bias_stride2_conv2357_fp32, 0, 0) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        return param.filter_meta.format == param::ConvBias::Format::NCHW &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_type.enumv() == DTypeEnum::Float32 &&
               param.dst_type.enumv() == DTypeEnum::Float32 &&
               !fm.should_flip && fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
               fm.dilation[1] == 1 && fm.stride[0] == 2 && fm.stride[1] == 2 &&
               FH == fm.spatial[1] &&
               (FH == 2 || FH == 3 || FH == 5 || FH == 7);
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoF32DirectStride2::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_aarch64_conv_bias_stride2_conv2357_fp32, 0, 1) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto wbundle = arm_common::MultithreadDirectConvCommon<
                float, float>::get_bundle_stride(param, large_group);
        return wbundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}
SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectStride2::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_aarch64_conv_bias_stride2_conv2357_fp32, 0, 2) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectStride2::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    auto FH = fm.spatial[0];
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    bool large_group = group >= param.nr_threads;
    using Func = std::function<void(const float*, const float*, float*, size_t,
                                    size_t, size_t, size_t, size_t)>;
    Func conv = nullptr;
    if (FH == 2) {
        conv = fp32::conv_stride2::do_conv_2x2_stride2;
    } else if (FH == 3) {
        conv = fp32::conv_stride2::do_conv_3x3_stride2;
    } else if (FH == 5) {
        conv = fp32::conv_stride2::do_conv_5x5_stride2;
    } else if (FH == 7) {
        conv = fp32::conv_stride2::do_conv_7x7_stride2;
    }

    WorkspaceBundle bundle = arm_common::MultithreadDirectConvCommon<
            float, float>::get_bundle_stride(param, large_group);
    SmallVector<NCBKern> ret_kerns;

    //! Dense conv and small group
    if (large_group) {
        //! Channel wise conv and big groups
        auto exec_one_group = [bundle, conv](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            bundle.set(kern_param.workspace_ptr);
            for (size_t ic = 0; ic < IC; ic++) {
                arm_common::MultithreadDirectConvCommon<float, float>::
                        copy_padding_kern_stride(bundle, kern_param, ncb_index,
                                                 {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                arm_common::MultithreadDirectConvCommon<
                        float, float>::do_conv_kern_stride(bundle, kern_param,
                                                           ncb_index, conv,
                                                           {ncb_index.thread_id,
                                                            0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            arm_common::MultithreadDirectConvCommon<float, float>::
                    copy_padding_kern_stride(bundle, kern_param, ncb_index,
                                             ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle, conv](const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            arm_common::MultithreadDirectConvCommon<
                    float, float>::do_conv_kern_stride(bundle, kern_param,
                                                       ncb_index, conv,
                                                       ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}
