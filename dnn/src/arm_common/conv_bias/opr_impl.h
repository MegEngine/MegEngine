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
#include "src/common/algo_base.h"

namespace megdnn {
namespace arm_common {

class ConvBiasImpl : public fallback::ConvBiasImpl {
public:
    using fallback::ConvBiasImpl::ConvBiasImpl;
    bool is_thread_safe() const override { return true; }
    class AlgoBase : public fallback::ConvBiasImpl::AlgoBase {
    public:
        AlgoBase() : fallback::ConvBiasImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::ARM_COMMON;
        }
    };

    SmallVector<fallback::ConvBiasImpl::AlgoBase*> get_all_packed_algo() override;

    bool is_matmul_quantized_prefer(
            const fallback::ConvBiasImpl::NCBKernSizeParam& ncb_param)
            const override;

    SmallVector<AlgoCategory> suggest_algo_category_order(
            const NCBKernSizeParam& param) const override;

    MEGDNN_FB_DECL_GET_ALGO_FROM_DESC(ConvBiasImpl);

protected:
    const char* get_algorithm_set_name() const override;

private:
    class AlgoS8DirectStride1;
    class AlgoS8DirectStride2;
    class AlgoS8DirectNCHW44;
    class AlgoS8x8x16DirectNCHW44;
    class AlgoS8DirectNCHWNCHW44;
    class AlgoQU8DirectStride1;
    class AlgoQU8DirectStride2;
    class AlgoFP32WinogradF23_4x4;
    class AlgoFP32WinogradF63;
    class AlgoFP32WinogradF63_4x4;
    class AlgoFP32WinogradF54;
    class AlgoFP32WinogradF45;

    class AlgoFP32WinogradF23_4x4_NCHW44;
    class AlgoFP32WinogradF63_4x4_NCHW44;
    class AlgoFP32WinogradF73_4x4_NCHW44;

    class AlgoS8ChanWiseStride1NCHW44;
    class AlgoS8ChanWiseStride2NCHW44;
    class AlgoS8x8x16ChanWiseStride1Stride2NCHW44;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoFP16WinogradF23;
    class AlgoFP16WinogradF45;
    class AlgoFP16WinogradF63;
    class AlgoFP16WinogradF23_8x8;
#endif
#if __ARM_FEATURE_DOTPROD
    class AlgoDotS8DirectNCHWNCHW44;
    class AlgoDotS8DirectStride1;
    class AlgoDotS8DirectStride2;
    class AlgoDotU8DirectStride1;
    class AlgoDotU8DirectStride2;

    class AlgoDotS8Direct_NCHW44;
#endif
    class AlgoF32Direct;
    class AlgoF32DirectStride1;
    class AlgoF32DirectStride2;
    class AlgoF32DirectNCHWNCHW44;
    class AlgoF32ChannelWiseNCHW44;
    class AlgoF32DirectNCHW44;

    class AlgoI8x8x16Direct;
    class AlgoI8x8x16Stride2;
    class AlgoI8x8x16Stride2Filter2;
    class AlgoI8x8x16DirectNCHWNCHW44;
    class AlgoS8WinogradF23_8x8;
    class AlgoS8CF32WinogradF23_4x4_NCHW44;
    class AlgoS8WinogradF23_8x8_NCHW44;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16Direct;
    class AlgoF16DirectStride1;
#endif

    class AlgoPack;
    static const AlgoPack& algo_pack();
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
