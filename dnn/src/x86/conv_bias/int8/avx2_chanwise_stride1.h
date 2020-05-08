/**
 * \file src/x86/conv_bias/int8/avx2_chanwsie_stride1.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
namespace avx2_chanwise_stride1 {
using NCBKern = fallback::ConvBiasImpl::NCBKern;
using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

using conv_fun = std::function<void(WorkspaceBundle bundle,
                                    const NCBKernParam& kern_param,
                                    const NCBKernIndex& ncb_index)>;

bool need_dst_copy(const NCBKernSizeParam& param);

bool need_src_copy(const NCBKernSizeParam& param);

void get_rectified_size(const NCBKernSizeParam& param, size_t& IH2, size_t& IW2,
                        size_t& OH2, size_t& OW2);

SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param,
                                WorkspaceBundle bundle);

}  // namespace avx2_chanwise_stride1
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
