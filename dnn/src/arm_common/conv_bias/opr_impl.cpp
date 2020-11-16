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

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/base.h"
#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8x8x16/algos.h"
#include "src/arm_common/conv_bias/quint8/algos.h"

#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/opr_impl.h"
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

bool is_fallback_or_naive(const detail::Algorithm* algo) {
    return algo->handle_type() == Handle::HandleType::NAIVE ||
           algo->handle_type() == Handle::HandleType::FALLBACK;
}

}  // anonymous namespace

class ConvBiasImpl::AlgoPack : NonCopyableObj {
    AlgoQU8DirectStride2 qu8_direct_stride2;
    AlgoQU8DirectStride1 qu8_direct_stride1;
    AlgoS8DirectStride2 s8_direct_stride2;
    AlgoS8DirectNCHW44 s8_direct_nchw44;
    AlgoS8x8x16DirectNCHW44 s8x8x16_direct_nchw44;
    AlgoS8DirectNCHWNCHW44 s8_direct_nchw_nchw44;
    AlgoS8DirectStride1 s8_direct_stride1;
    AlgoS8ChanWiseStride1NCHW44 s8_channel_wise_stride1_nchw44;
    AlgoS8ChanWiseStride2NCHW44 s8_channel_wise_stride2_nchw44;
    AlgoS8x8x16ChanWiseStride1Stride2NCHW44
            s8x8x16_channel_wise_stride1_stride2_nchw44;

#if __ARM_FEATURE_DOTPROD
    AlgoDotS8DirectStride1 ds8_direct_stride1;
    AlgoDotS8DirectStride2 ds8_direct_stride2;
    AlgoDotU8DirectStride1 du8_direct_stride1;
    AlgoDotU8DirectStride2 du8_direct_stride2;

    AlgoDotS8Direct_NCHW44 ds8_direct_nchw44;
    AlgoDotS8DirectNCHWNCHW44 ds8_direct_nchw_nchw44;
#endif

    AlgoF32DirectNCHWNCHW44 f32_direct_stride2_nchw_nchw44;
    AlgoF32ChannelWiseNCHW44 f32_chanel_wise_nchw44;
    AlgoF32DirectNCHW44 f32_direct_nchw44;

    AlgoF32Direct f32_direct;
    AlgoF32DirectStride2 f32_direct_stride2;
    AlgoF32DirectStride1 f32_direct_stride1;

    AlgoI8x8x16Direct i8x8x16_direct;
    AlgoI8x8x16Stride2 i8x8x16_stride2;
    AlgoI8x8x16Stride2Filter2 i8x8x16_stride2_filter2;
    AlgoI8x8x16DirectNCHWNCHW44 i8x8x16_nchw_nchw44;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16Direct f16_direct;
    AlgoF16DirectStride1 f16_direct_stride1;
#endif

    SmallVector<std::unique_ptr<AlgoBase>> refhold;
    fallback::ConvBiasImpl::AlgoBase::Mapper m_all_algos_map;
    SmallVector<fallback::ConvBiasImpl::AlgoBase*> m_direct_algos;
    SmallVector<fallback::ConvBiasImpl::AlgoBase*> m_winograd_algos;

public:
    AlgoPack() {
#if __ARM_FEATURE_DOTPROD
        m_direct_algos.emplace_back(&ds8_direct_stride1);
        m_direct_algos.emplace_back(&ds8_direct_stride2);
        m_direct_algos.emplace_back(&du8_direct_stride1);
        m_direct_algos.emplace_back(&du8_direct_stride2);

        m_direct_algos.emplace_back(&ds8_direct_nchw44);
        m_direct_algos.emplace_back(&ds8_direct_nchw_nchw44);
#endif
        m_direct_algos.emplace_back(&qu8_direct_stride2);
        m_direct_algos.emplace_back(&qu8_direct_stride1);
        m_direct_algos.emplace_back(&s8_direct_stride2);
        m_direct_algos.emplace_back(&s8_direct_nchw44);
        m_direct_algos.emplace_back(&s8x8x16_direct_nchw44);
        m_direct_algos.emplace_back(&s8_direct_nchw_nchw44);
        m_direct_algos.emplace_back(&s8_direct_stride1);

        m_direct_algos.emplace_back(
                &s8x8x16_channel_wise_stride1_stride2_nchw44);
        m_direct_algos.emplace_back(&s8_channel_wise_stride1_nchw44);
        m_direct_algos.emplace_back(&s8_channel_wise_stride2_nchw44);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        m_direct_algos.emplace_back(&f16_direct_stride1);
        m_direct_algos.emplace_back(&f16_direct);
#endif
        m_direct_algos.emplace_back(&i8x8x16_direct);
        m_direct_algos.emplace_back(&i8x8x16_stride2_filter2);
        m_direct_algos.emplace_back(&i8x8x16_stride2);
        m_direct_algos.emplace_back(&i8x8x16_nchw_nchw44);

        m_direct_algos.emplace_back(&f32_direct_stride2_nchw_nchw44);
        m_direct_algos.emplace_back(&f32_chanel_wise_nchw44);
        m_direct_algos.emplace_back(&f32_direct_nchw44);

        m_direct_algos.emplace_back(&f32_direct_stride1);
        m_direct_algos.emplace_back(&f32_direct_stride2);
        m_direct_algos.emplace_back(&f32_direct);

        static CpuOprDelegationStorage<2> storage;
        auto matmul_opr = storage.get<MatrixMul, 0>();
        using MatmulFormat = param::MatrixMul::Format;
        auto&& matmul_algos =
                static_cast<arm_common::MatrixMulImpl*>(matmul_opr)
                        ->select_algo_type(
                                {AlgoDataType::FLOAT32, MatmulFormat::MK4});
        for (auto&& algo : matmul_algos) {
            if (is_fallback_or_naive(algo))
                continue;
            for (uint32_t tile_size : {16, 8, 24, 32}) {
                refhold.emplace_back(new AlgoFP32WinogradF23_4x4(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF63_4x4(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF63_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF23_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
//! uncomment this when low precision mode is done
#if 0
                refhold.emplace_back(new AlgoFP32WinogradF73_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
#endif
                //! Qint8x8x32 winograd compute with fp32
                refhold.emplace_back(new AlgoS8CF32WinogradF23_4x4_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
            }
        }
        matmul_algos = static_cast<arm_common::MatrixMulImpl*>(matmul_opr)
                               ->select_algo_type({AlgoDataType::FLOAT32,
                                                   MatmulFormat::DEFAULT});
        for (auto&& algo : matmul_algos) {
            if (is_fallback_or_naive(algo))
                continue;
            for (uint32_t tile_size : {16, 8, 24, 32}) {
                refhold.emplace_back(new AlgoFP32WinogradF63(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF54(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF45(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
            }
        }

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        matmul_algos = static_cast<arm_common::MatrixMulImpl*>(matmul_opr)
                               ->select_algo_type({AlgoDataType::FLOAT16,
                                                   MatmulFormat::DEFAULT});
        for (auto&& algo : matmul_algos) {
            if (is_fallback_or_naive(algo))
                continue;
            for (uint32_t tile_size : {16, 8, 24, 32}) {
                refhold.emplace_back(new AlgoFP16WinogradF23(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP16WinogradF45(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP16WinogradF63(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
            }
        }
        matmul_algos = static_cast<arm_common::MatrixMulImpl*>(matmul_opr)
                               ->select_algo_type({AlgoDataType::FLOAT16,
                                                   MatmulFormat::MK8});
        for (auto&& algo : matmul_algos) {
            if (is_fallback_or_naive(algo))
                continue;
            for (uint32_t tile_size : {16, 8, 24, 32}) {
                refhold.emplace_back(new AlgoFP16WinogradF23_8x8(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
            }
        }
#endif
        matmul_algos = static_cast<arm_common::MatrixMulImpl*>(matmul_opr)
                               ->select_algo_type({AlgoDataType::INT16X16X32,
                                                   MatmulFormat::MK8});
        for (auto&& algo : matmul_algos) {
            if (is_fallback_or_naive(algo))
                continue;
            for (uint32_t tile_size : {16, 8, 24, 32}) {
                refhold.emplace_back(new AlgoS8WinogradF23_8x8(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoS8WinogradF23_8x8_NCHW44(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                m_winograd_algos.emplace_back(refhold.back().get());
            }
        }


        for (auto&& algo : m_direct_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
        for (auto&& algo : m_winograd_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<fallback::ConvBiasImpl::AlgoBase*>& direct_algos()
            const {
        return m_direct_algos;
    }
    const SmallVector<fallback::ConvBiasImpl::AlgoBase*>& winograd_algos()
            const {
        return m_winograd_algos;
    }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const ConvBiasImpl::AlgoPack& ConvBiasImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(ConvBiasImpl)

SmallVector<fallback::ConvBiasImpl::AlgoBase*>
ConvBiasImpl::get_all_packed_algo() {
    auto&& algos = fallback::ConvBiasImpl::get_all_packed_algo();
    algos.insert(algos.begin(), algo_pack().direct_algos().begin(),
                 algo_pack().direct_algos().end());
    algos.insert(algos.end(), algo_pack().winograd_algos().begin(),
                 algo_pack().winograd_algos().end());
    return std::move(algos);
}

bool ConvBiasImpl::is_matmul_quantized_prefer(
        const ConvBiasImpl::NCBKernSizeParam& param) const {
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

SmallVector<AlgoCategory> ConvBiasImpl::suggest_algo_category_order(
        const NCBKernSizeParam& param) const {
    auto IC = param.filter_meta.icpg;
    auto OC = param.filter_meta.ocpg;
    auto FH = param.filter_meta.spatial[0];
    auto FW = param.filter_meta.spatial[1];
    //! TODO: now winograd only support fast-run
    if (param.filter_meta.format == param::ConvBias::Format::NCHW_WINOGRAD ||
        param.filter_meta.format == param::ConvBias::Format::NCHW44_WINOGRAD ||
        param.filter_meta.format == param::ConvBias::Format::NCHW88_WINOGRAD) {
        return {AlgoCategory::WINOGRAD};
    }
    //! im2col
    bool im2col_prefer = (IC >= 32 || OC >= 32);
    //! quantized algo use matmul when direct algo is unusable
    if (param.src_type.category() == DTypeCategory::QUANTIZED) {
        im2col_prefer = is_matmul_quantized_prefer(param);
    }
    //! conv1x1
    im2col_prefer |= (FH == 1 && FW == 1);
    //! nchw44 and nchw44-dot hybird mode is direct
    if (param.filter_meta.format == param::ConvBias::Format::NCHW44 ||
        param.filter_meta.format == param::ConvBias::Format::NCHW44_DOT) {
        if (IC < 4) {
            im2col_prefer = false;
        }
    }
    if (im2col_prefer) {
        return {AlgoCategory::IM2COL, AlgoCategory::DIRECT,
                AlgoCategory::NAIVE};
    } else {
        return {AlgoCategory::DIRECT, AlgoCategory::IM2COL,
                AlgoCategory::NAIVE};
    }
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    // arm common version 0
    return "AC0";
}

// vim: syntax=cpp.doxygen
