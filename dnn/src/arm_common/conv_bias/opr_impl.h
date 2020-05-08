/**
 * \file dnn/src/arm_common/conv_bias/opr_impl.h
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
#include "src/common/utils.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ConvBiasImpl : public fallback::ConvBiasImpl {
public:
    using fallback::ConvBiasImpl::ConvBiasImpl;
    using FallbackConvBiasImpl = fallback::ConvBiasImpl;
    using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

    bool is_thread_safe() const override { return true; }

    SmallVector<AlgoBase*> algo_pack() override;

    bool is_matmul_quantized_prefer(
            const ConvBiasImpl::NCBKernSizeParam& ncb_param) override;
    class AlgoPack;

protected:
    static void* const sm_arm_common_algo_type;

    const char* get_algorithm_set_name() const override;

private:
    class AlgoS8DirectStride1;
    class AlgoS8DirectStride1NCHW44;
    class AlgoS8DirectStride2;
    class AlgoS8DirectStride2NCHW44;
    class AlgoS8DirectStride2NCHWNCHW44;
    class AlgoQU8DirectStride1;
    class AlgoQU8DirectStride2;
    class AlgoFP32WinogradF23_4x4;
    class AlgoFP32WinogradF63;
    class AlgoFP32WinogradF63_4x4;
    class AlgoFP32WinogradF54;
    class AlgoFP32WinogradF45;

    class AlgoS8ChanWiseStride1NCHW44;
    class AlgoS8ChanWiseStride2NCHW44;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoFP16WinogradF23;
    class AlgoFP16WinogradF45;
    class AlgoFP16WinogradF63;
    class AlgoFP16WinogradF23_8x8;
#endif
#if __ARM_FEATURE_DOTPROD
    class AlgoDotS8DirectStride1;
    class AlgoDotS8DirectStride2;
    class AlgoDotU8DirectStride1;
    class AlgoDotU8DirectStride2;
#endif
    class AlgoF32Direct;
    class AlgoF32DirectStride1;
    class AlgoF32DirectStride2;
    class AlgoI8x8x16Direct;
    class AlgoI8x8x16Stride2;
    class AlgoI8x8x16Stride2Filter2;
    class AlgoS8WinogradF23_8x8;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16Direct;
    class AlgoF16DirectStride1;
#endif
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
