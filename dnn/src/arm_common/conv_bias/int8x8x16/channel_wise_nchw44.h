/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/channel_wise_nchw44.h
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
namespace channel_wise_nchw44 {

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

using conv_fun = std::function<void(const WorkspaceBundle& bundle,
                                    const NCBKernParam& kern_param,
                                    const NCBKernIndex& ncb_index)>;

namespace stride1 {

bool is_available(const NCBKernSizeParam& param);

WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

template <bool quantized, size_t filter, BiasMode bias_mode, typename Op>
void do_conv_kern(const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
                  const NCBKernIndex& ncb_index);

SmallVector<ConvBiasImpl::NCBKern> get_kimpls(const NCBKernSizeParam& param);
}  // namespace stride1

namespace stride2 {
bool is_available(const NCBKernSizeParam& param);

WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

template <bool quantized, size_t filter, BiasMode bias_mode, typename Op>
void do_conv_kern(const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
                  const NCBKernIndex& ncb_index);

SmallVector<ConvBiasImpl::NCBKern> get_kimpls(const NCBKernSizeParam& param);

}  // namespace stride2
}  // namespace direct_int8_stride1
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
