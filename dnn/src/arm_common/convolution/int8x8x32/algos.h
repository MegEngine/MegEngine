/**
 * \file dnn/src/arm_common/convolution/int8x8x32/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/arm_common/convolution/opr_impl.h"

namespace megdnn {
namespace arm_common {

#if __ARM_FEATURE_DOTPROD
/* ===================== ConvolutionBackwardData ===================== */

class ConvolutionBackwardDataImpl::AlgoSdot8DirectStride1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH32_I8x8x32_DECONV_STRIDE1"; }

    bool usable(ConvolutionBackwardDataImpl*,
                const NCBKernSizeParam& param) const override;

    size_t get_workspace(ConvolutionBackwardDataImpl*,
                         const NCBKernSizeParam& param) const override;

    ncb_kern_t dispatch_kern(ConvolutionBackwardDataImpl*,
                             const NCBKernSizeParam&) const override;

    void* type() const override { return sm_arm_common_algo_type; }
};

class ConvolutionBackwardDataImpl::AlgoSdot8DirectStride2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH32_I8x8x32_DECONV_STRIDE2"; }

    bool usable(ConvolutionBackwardDataImpl*,
                const NCBKernSizeParam& param) const override;

    size_t get_workspace(ConvolutionBackwardDataImpl*,
                         const NCBKernSizeParam& param) const override;

    ncb_kern_t dispatch_kern(ConvolutionBackwardDataImpl*,
                             const NCBKernSizeParam&) const override;

    void* type() const override { return sm_arm_common_algo_type; }
};

#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
