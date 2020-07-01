/**
 * \file dnn/src/x86/conv_bias/int8/avx2_direct_conv_stride1.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
namespace direct_conv_avx2_stride1 {

using NCBKern = fallback::ConvBiasImpl::NCBKern;
using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;

SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param,
                                const WorkspaceBundle& bundle);

}  // namespace direct_conv_avx2_stride1
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
