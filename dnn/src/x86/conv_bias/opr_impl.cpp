/**
 * \file dnn/src/x86/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/x86/conv_bias/opr_impl.h"
#include <algorithm>
#include <memory>
#include "src/common/metahelper.h"
#include "src/common/opr_delegate.h"
#include "src/x86/conv_bias/f32/algos.h"
#include "src/x86/conv_bias/int8/algo_usable_preferred.h"
#include "src/x86/conv_bias/int8/algos.h"
#include "src/x86/matrix_mul/opr_impl.h"

using namespace megdnn;
using namespace x86;

namespace {
uint8_t x86_algo_type_storage;
void* x86_algo_type = &x86_algo_type_storage;
}  // anonymous namespace
#if MEGDNN_X86_WITH_MKL_DNN
void* ConvBiasImpl::AlgoMkldnnQint8::type() const {
    return x86_algo_type;
}
void* ConvBiasImpl::AlgoMkldnnMatmulQint8::type() const {
    return x86_algo_type;
}
void* ConvBiasImpl::AlgoMkldnnConv::type() const {
    return x86_algo_type;
}
#endif

void* ConvBiasImpl::AlgoDirect::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoDirectStride2::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoMatrixMul::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoDirectAvx2Stride1Int8::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoFP32WinogradF63_8x8::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoFP32WinogradF23_8x8::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoAVX2DirectConvStride2::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoChanWiseAvx2Stride1Qint8::type() const {
    return x86_algo_type;
}

void* ConvBiasImpl::AlgoChanWiseAvx2Stride2Qint8::type() const {
    return x86_algo_type;
}

class ConvBiasImpl::AlgoPack : NonCopyableObj {
    AlgoDirect stride1_direct;
    AlgoDirectStride2 stride2_direct;
    AlgoDirectAvx2Stride1Int8 avx2_stride1_direct_int8;
    AlgoAVX2DirectConvStride2 avx2_stride2_direct;
    AlgoChanWiseAvx2Stride1Qint8 avx2_stride1_chanwsie_qint8;
    AlgoChanWiseAvx2Stride2Qint8 avx2_stride2_chanwsie_qint8;
    AlgoMatrixMul matmul;
#if MEGDNN_X86_WITH_MKL_DNN
    AlgoMkldnnMatmulQint8 mkldnn_matmul_qint8;
    //! Because the mkldnnconv need handle
    AlgoMkldnnQint8 mkldnn_qint8;
    AlgoMkldnnConv mkldnn_conv_fp32;
#endif
    SmallVector<std::unique_ptr<AlgoBase>> refhold;

public:
    AlgoPack() {
        //! FIXME: preference to use mkldnn algo on VNNI devices
        //! But now mkldnn algo preference issue with NCHW->NHWC->NCHW
#if MEGDNN_X86_WITH_MKL_DNN
        //! Create the mkldnn algo
        all_algos.emplace_back(&mkldnn_conv_fp32);
        all_algos.emplace_back(&mkldnn_matmul_qint8);
        all_algos.emplace_back(&mkldnn_qint8);
#endif
        all_algos.emplace_back(&stride1_direct);
        all_algos.emplace_back(&stride2_direct);
        all_algos.emplace_back(&avx2_stride1_chanwsie_qint8);
        all_algos.emplace_back(&avx2_stride2_chanwsie_qint8);
        all_algos.emplace_back(&avx2_stride1_direct_int8);
        all_algos.emplace_back(&avx2_stride2_direct);
        all_algos.emplace_back(&matmul);

        static CpuOprDelegationStorage<> storage;
        auto matmul_opr = storage.get<MatrixMul>();
        auto&& matmul_algos =
                static_cast<MatrixMulImpl*>(matmul_opr)->algo_pack();
        for (auto&& algo : matmul_algos) {
            if (algo->type() == nullptr)
                continue;
            for (uint32_t tile_size : {8, 16, 24}) {
                refhold.emplace_back(new AlgoFP32WinogradF63_8x8(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
                refhold.emplace_back(new AlgoFP32WinogradF23_8x8(
                        static_cast<fallback::MatrixMulImpl::AlgoBase*>(algo),
                        tile_size));
                winograd_algos.emplace_back(refhold.back().get());
            }
        }
    }
    SmallVector<AlgoBase*> all_algos;
    SmallVector<AlgoBase*> winograd_algos;
};

SmallVector<ConvBiasImpl::AlgoBase*> ConvBiasImpl::algo_pack() {
    static AlgoPack sl_algo_pack;
    auto&& algos = fallback::ConvBiasImpl::algo_pack();
    algos.insert(algos.begin(), sl_algo_pack.all_algos.begin(),
                 sl_algo_pack.all_algos.end());
    algos.insert(algos.end(), sl_algo_pack.winograd_algos.begin(),
                 sl_algo_pack.winograd_algos.end());
    return std::move(algos);
}

void ConvBiasImpl::get_rectified_img_size(size_t IH, size_t IW, size_t FH,
                                          size_t FW, size_t OH, size_t OW,
                                          size_t PH, size_t PW, size_t& IH2,
                                          size_t& IW2, size_t& OH2,
                                          size_t& OW2) {
    OW2 = (OW + 7) >> 3 << 3;
    OH2 = OH;
    IH2 = std::max(IH, OH2 + FH - 1 + 2 * PH);
    IW2 = std::max(IW, OW2 + FW - 1 + 2 * PW);
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    // x86 version 0
    return "X0";
}

bool ConvBiasImpl::is_matmul_quantized_prefer(
        const ConvBiasImpl::NCBKernSizeParam& param) const {
    bool conv_direct_chanwise_mkldnn_usable = true;
    if (param.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
        param.dst_type.enumv() == DTypeEnum::QuantizedS32) {
        conv_direct_chanwise_mkldnn_usable =
                chanwise_avx2_stride1_qint8_usable_preferred(param) ||
                chanwise_avx2_stride2_qint8_usable_preferred(param) ||
                direct_avx2_stride1_int8_usable_preferred(param) ||
                direct_avx2_stride2_int8_usable_preferred(param);
#if MEGDNN_X86_WITH_MKL_DNN
        conv_direct_chanwise_mkldnn_usable =
                conv_direct_chanwise_mkldnn_usable ||
                mkldnn_qint8_usable_preferred(param) ||
                mkldnn_matmul_qint8_usable_preferred(param);
#endif
    }

    return !conv_direct_chanwise_mkldnn_usable ||
           (is_supported(SIMDType::VNNI) &&
            !chanwise_avx2_stride1_qint8_usable_preferred(param) &&
            !chanwise_avx2_stride2_qint8_usable_preferred(param));
}

SmallVector<AlgoCategory>
ConvBiasImpl::suggest_algo_category_order(const NCBKernSizeParam& param) const {
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
    //! nchw88 use mkl-dnn which algo is direct
    if (param.filter_meta.format == param::ConvBias::Format::NCHW88) {
        return {AlgoCategory::DIRECT, AlgoCategory::IM2COL};
    }
    //! im2col + matmul
    bool im2col_prefer = (IC >= 32 || OC >= 32);
    //! quantized algo use matmul when direct algo is unusable
    if (param.src_type.category() == DTypeCategory::QUANTIZED) {
        im2col_prefer = is_matmul_quantized_prefer(param);
    }
    //! conv1x1
    im2col_prefer |= (FH == 1 && FW == 1);
    //! x86 8x8x16 not optmized, so it will use fallback im2col+matmul
    if (param.deduce_algo_data_type() == AlgoDataType::INT8X8X16) {
        im2col_prefer = true;
    }
    if (im2col_prefer) {
        return {AlgoCategory::IM2COL, AlgoCategory::DIRECT,
                AlgoCategory::NAIVE};
    } else {
        return {AlgoCategory::DIRECT, AlgoCategory::IM2COL,
                AlgoCategory::NAIVE};
    }
}

// vim: syntax=cpp.doxygen
