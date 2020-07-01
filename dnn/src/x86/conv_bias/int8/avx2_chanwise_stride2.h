/**
 * \file src/x86/conv_bias/int8/avx2_chanwsie_stride2.h
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

#include "src/x86/conv_bias/int8/common_helper.h"
#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
namespace avx2_chanwise_stride2 {
using conv_fun = std::function<void(const WorkspaceBundle& bundle,
                                    const NCBKernParam& kern_param,
                                    const NCBKernIndex& ncb_index)>;
SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param,
                                const WorkspaceBundle& bundle);

}  // namespace avx2_chanwise_stride2
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
