/**
 * \file dnn/src/arm_common/conv_bias/direct/multi_thread_common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"

namespace megdnn {
namespace arm_common {

template <class io_ctype, class compute_ctype>
class MultithreadDirectConvCommon {
public:
    using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
    using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
    using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

    using kern_direct_conv_f32 =
            std::function<void(const compute_ctype* src,
                               const compute_ctype* filter, compute_ctype* dst,
                               size_t, size_t, size_t, size_t, size_t, size_t)>;
    using kern_direct_conv_f32_stride = std::function<void(
            const compute_ctype* src, const compute_ctype* filter,
            compute_ctype* dst, size_t, size_t, size_t, size_t, size_t)>;

    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param,
                                      bool m_large_group);
    static WorkspaceBundle get_bundle_stride(const NCBKernSizeParam& param,
                                             bool m_large_group);
    static void weight_flip_kern(const WorkspaceBundle& bundle,
                                 const NCBKernParam& kern_param,
                                 const NCBKernIndex& ncb_index,
                                 const CpuNDRange& workspace_ids);
    static void copy_padding_kern(const WorkspaceBundle& bundle,
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index,
                                  const CpuNDRange& workspace_ids);
    static void copy_padding_kern_stride(const WorkspaceBundle& bundle,
                                         const NCBKernParam& kern_param,
                                         const NCBKernIndex& ncb_index,
                                         const CpuNDRange& workspace_ids);
    static void do_conv_kern(const WorkspaceBundle& bundle,
                             const NCBKernParam& kern_param,
                             const NCBKernIndex& ncb_index,
                             const kern_direct_conv_f32& fun,
                             const CpuNDRange& workspace_ids);
    static void do_conv_kern_stride(const WorkspaceBundle& bundle,
                                    const NCBKernParam& kern_param,
                                    const NCBKernIndex& ncb_index,
                                    const kern_direct_conv_f32_stride& fun,
                                    const CpuNDRange& workspace_ids);
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
