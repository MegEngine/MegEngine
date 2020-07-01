/**
 * \file dnn/src/arm_common/conv_bias/int8/stride1.h
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

namespace megdnn {
namespace arm_common {
namespace direct_int8_stride1 {
using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

using conv_fun = std::function<void(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids)>;

bool can_conv_direct_stride1_int8(const NCBKernSizeParam& param);

WorkspaceBundle get_bundle(const NCBKernSizeParam& param, bool m_large_group);

void copy_padding_kern(const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
                       const NCBKernIndex& ncb_index,
                       const CpuNDRange& workspace_ids);

template <size_t filter, BiasMode bias_mode, typename Op>
void do_conv_kern(const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
                  const NCBKernIndex& ncb_index,
                  const CpuNDRange& workspace_ids);

SmallVector<ConvBiasImpl::NCBKern> get_kimpls(const NCBKernSizeParam& param,
                                              bool);
}  // namespace direct_int8_stride1
}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
