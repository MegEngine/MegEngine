/**
 * \file dnn/src/arm_common/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8x8x16/algos.h"
#include "src/arm_common/conv_bias/quint8/algos.h"

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "src/arm_common/convolution/opr_impl.h"
#include "src/arm_common/matrix_mul/opr_impl.h"
#include "src/common/opr_delegate.h"

#include "include/megdnn/oprs/nn.h"
#include "src/arm_common/conv_bias/f16/algos.h"
#include "src/arm_common/conv_bias/fp32/algos.h"
#include "src/arm_common/conv_bias/int8/stride1.h"
#include "src/arm_common/conv_bias/int8/stride2.h"
#include "src/arm_common/conv_bias/quint8/stride1.h"
#include "src/arm_common/conv_bias/quint8/stride2.h"
#include "src/arm_common/convolution/opr_impl.h"

using namespace megdnn;
using namespace arm_common;

namespace {
uint8_t arm_common_algo_type_storage;
}  // anonymous namespace

class ConvBiasImpl::AlgoPack : NonCopyableObj {
    AlgoQU8DirectStride2 qu8_direct_stride2_large_group{true};
    AlgoQU8DirectStride2 qu8_direct_stride2_small_group{false};
    AlgoQU8DirectStride1 qu8_direct_stride1_large_group{true};
    AlgoQU8DirectStride1 qu8_direct_stride1_small_group{false};
    AlgoS8DirectStride2 s8_direct_stride2_large_group{true};
    AlgoS8DirectStride2 s8_direct_stride2_small_group{false};
    AlgoS8DirectNCHW44 s8_direct_nchw44;
    AlgoS8DirectNCHWNCHW44 s8_direct_nchw_nchw44;
    AlgoS8DirectStride1 s8_direct_stride1_large_group{true};
    AlgoS8DirectStride1 s8_direct_stride1_small_group{false};
    AlgoS8ChanWiseStride1NCHW44 s8_channel_wise_stride1_nchw44;
    AlgoS8ChanWiseStride2NCHW44 s8_channel_wise_stride2_nchw44;

#if __ARM_FEATURE_DOTPROD
    AlgoDotS8DirectStride1 ds8_direct_stride1_large_group{true};
    AlgoDotS8DirectStride1 ds8_direct_stride1_small_group{false};
    AlgoDotS8DirectStride2 ds8_direct_stride2_large_group{true};
    AlgoDotS8DirectStride2 ds8_direct_stride2_small_group{false};
    AlgoDotU8DirectStride1 du8_direct_stride1_large_group{true};
    AlgoDotU8DirectStride1 du8_direct_stride1_small_group{false};
    AlgoDotU8DirectStride2 du8_direct_stride2_large_group{true};
    AlgoDotU8DirectStride2 du8_direct_stride2_small_group{false};

    AlgoDotS8Direct_NCHW44 ds8_direct_nchw44;
    AlgoDotS8DirectNCHWNCHW44 ds8_direct_nchw_nchw44;
#endif

    AlgoF32DirectNCHWNCHW44 f32_direct_stride2_nchw_nchw44;
    AlgoF32ChannelWiseNCHW44 f32_chanel_wise_nchw44;
    AlgoF32DirectNCHW44 f32_direct_nchw44;

    AlgoF32Direct f32_direct_large_group{true};
    AlgoF32Direct f32_direct_small_group{false};
    AlgoF32DirectStride2 f32_direct_stride2_large_group{true};
    AlgoF32DirectStride2 f32_direct_stride2_small_group{false};
    AlgoF32DirectStride1 f32_direct_stride1_large_group{true};
    AlgoF32DirectStride1 f32_direct_stride1_small_group{false};

    AlgoI8x8x16Direct i8x8x16_direct_large_group{true};
    AlgoI8x8x16Direct i8x8x16_direct_small_group{false};
    AlgoI8x8x16Stride2 i8x8x16_stride2_large_group{true};
    AlgoI8x8x16Stride2 i8x8x16_stride2_small_group{false};
    AlgoI8x8x16Stride2Filter2 i8x8x16_stride2_filter2;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16Direct f16_direct_large_group{true};
    AlgoF16Direct f16_direct_small_group{false};
    AlgoF16DirectStride1 f16_direct_stride1_large_group{true};
    AlgoF16DirectStride1 f16_direct_stride1_small_group{false};
#endif

    SmallVector<std::unique_ptr<AlgoBase>> refhold;

public:
    AlgoPack() {
#if __ARM_FEATURE_DOTPROD
        direct_algos.emplace_back(&ds8_direct_stride1_large_group);
        direct_algos.emplace_back(&ds8_direct_stride1_small_group);
        direct_algos.emplace_back(&ds8_direct_stride2_large_group);
        direct_algos.emplace_back(&ds8_direct_stride2_small_group);
        direct_algos.emplace_back(&du8_direct_stride1_large_group);
        direct_algos.emplace_back(&du8_direct_stride1_small_group);
        direct_algos.emplace_back(&du8_direct_stride2_large_group);
        direct_algos.emplace_back(&du8_direct_stride2_small_group);

        direct_algos.emplace_back(&ds8_direct_nchw44);
        direct_algos.emplace_back(&ds8_direct_nchw_nchw44);
#endif
        direct_algos.emplace_back(&qu8_direct_stride2_large_group);
        direct_algos.emplace_back(&qu8_direct_stride2_small_group);
        direct_algos.emplace_back(&qu8_direct_stride1_large_group);
        direct_algos.emplace_back(&qu8_direct_stride1_small_group);
        direct_algos.emplace_back(&s8_direct_stride2_large_group);
        direct_algos.emplace_back(&s8_direct_stride2_small_group);
        direct_algos.emplace_back(&s8_direct_nchw44);
        direct_algos.emplace_back(&s8_direct_nchw_nchw44);
        direct_algos.emplace_back(&s8_direct_stride1_large_group);
        direct_algos.emplace_back(&s8_direct_stride1_small_group);

        direct_algos.emplace_back(&s8_channel_wise_stride1_nchw44);
        direct_algos.emplace_back(&s8_channel_wise_stride2_nchw44);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        direct_algos.emplace_back(&f16_direct_stride1_large_group);
        direct_algos.emplace_back(&f16_direct_stride1_small_group);
        direct_algos.emplace_back(&f16_direct_large_group);
        direct_algos.emplace_back(&f16_direct_small_group);
#endif
        direct_algos.emplace_back(&i8x8x16_direct_large_group);
        direct_algos.emplace_back(&i8x8x16_direct_small_group);
        direct_algos.emplace_back(&i8x8x16_stride2_filter2);
        direct_algos.emplace_back(&i8x8x16_stride2_large_group);
        direct_algos.emplace_back(&i8x8x16_stride2_small_group);

        direct_algos.emplace_back(&f32_direct_stride2_nchw_nchw44);
        direct_algos.emplace_back(&f32_chanel_wise_nchw44);
        direct_algos.emplace_back(&f32_direct_nchw44);

        direct_algos.emplace_back(&f32_direct_stride1_large_group);
        direct_algos.emplace_back(&f32_direct_stride1_small_group);
        direct_algos.emplace_back(&f32_direct_stride2_large_group);
        direct_algos.emplace_back(&f32_direct_stride2_small_group);
        direct_algos.emplace_back(&f32_direct_large_group);
        direct_algos.emplace_back(&f32_direct_small_group);

        static CpuOprDelegationStorage<2> storage;
        auto matmul_opr = storage.get<MatrixMul, 0>();
        auto&& matmul_algos =
                static_cast<arm_common::MatrixMulImpl*>(matmul_opr)
                        ->algo_pack();
        for (auto&& algo : matmul_algos) {
            if (algo->type() == nullptr)
                continue;
            for (uint32_t tile_size : {16, 8, 24, 32}) {
                refhold.emplace_back(new AlgoFP32WinogradF23_4x4(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF63(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF63_4x4(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF54(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF45(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF23_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF63_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                refhold.emplace_back(new AlgoFP16WinogradF23(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP16WinogradF45(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP16WinogradF63(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP16WinogradF23_8x8(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
#endif
                refhold.emplace_back(new AlgoS8WinogradF23_8x8(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoS8CF32WinogradF23_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoS8WinogradF23_8x8_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
            }
        }
    }
    SmallVector<AlgoBase*> direct_algos;
    SmallVector<AlgoBase*> winograd_algos;
};

SmallVector<ConvBiasImpl::AlgoBase*> ConvBiasImpl::algo_pack() {
    static AlgoPack sl_algo_pack;
    auto&& algos = fallback::ConvBiasImpl::algo_pack();
    algos.insert(algos.begin(), sl_algo_pack.direct_algos.begin(),
                 sl_algo_pack.direct_algos.end());
    algos.insert(algos.end(), sl_algo_pack.winograd_algos.begin(),
                 sl_algo_pack.winograd_algos.end());
    return std::move(algos);
}

void* const ConvBiasImpl::sm_arm_common_algo_type =
        &arm_common_algo_type_storage;

bool ConvBiasImpl::is_matmul_quantized_prefer(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    // fallback::ConvBiasImpl::NCBKernParam conv_ncb_param;
    fallback::ConvBiasImpl::NCBKernSizeParam conv_ncb_param(
            param, 0, param::MatrixMul::Format::DEFAULT, {}, 0,
            BiasMode::NO_BIAS, param::ConvBias::NonlineMode::IDENTITY);
    conv_ncb_param.dst_type = param.bias_type;
    conv_ncb_param.filter_meta.group = 1;

    bool conv_direct_unusable = false;
    if (param.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
        param.dst_type.enumv() == DTypeEnum::QuantizedS32) {
        conv_direct_unusable =
                !arm_common::direct_int8_stride1::can_conv_direct_stride1_int8(
                        conv_ncb_param) &&
                !arm_common::direct_int8_stride2::can_conv_direct_stride2_int8(
                        conv_ncb_param);
    } else if (param.dst_type.enumv() == DTypeEnum::Quantized8Asymm) {
        conv_direct_unusable =
                !arm_common::direct_quint8_stride1::
                        can_conv_direct_stride1_quint8(conv_ncb_param) &&
                !arm_common::direct_quint8_stride2::
                        can_conv_direct_stride2_quint8(conv_ncb_param);
    }
    return conv_direct_unusable;
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    // arm common version 0
    return "AC0";
}

// vim: syntax=cpp.doxygen
