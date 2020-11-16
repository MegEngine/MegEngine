/**
 * \file dnn/src/x86/conv_bias/opr_impl.h
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
namespace x86 {

class ConvBiasImpl : public fallback::ConvBiasImpl {
public:
    using fallback::ConvBiasImpl::ConvBiasImpl;
    class AlgoBase : public fallback::ConvBiasImpl::AlgoBase {
    public:
        AlgoBase() : fallback::ConvBiasImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::X86;
        }
    };

    bool is_thread_safe() const override { return true; }
    SmallVector<fallback::ConvBiasImpl::AlgoBase*> get_all_packed_algo() override;
    SmallVector<AlgoCategory> suggest_algo_category_order(
            const NCBKernSizeParam& param) const override;

    /**
     * \brief Adjust tensor layouts to fulfill alignment requirements.
     * OW2 would be 8-byte aligned.
     * IH2/IW2 would be adjusted to fit OH2/OW2.
     * The influence of padding would be incorporated in IH2/IW2.
     */
    static void get_rectified_img_size(size_t IH, size_t IW, size_t FH,
                                       size_t FW, size_t OH, size_t OW,
                                       size_t PH, size_t PW, size_t& IH2,
                                       size_t& IW2, size_t& OH2, size_t& OW2);

    const char* get_algorithm_set_name() const override;

    bool is_matmul_quantized_prefer(
            const ConvBiasImpl::NCBKernSizeParam& ncb_param) const override;
    static fallback::ConvBiasImpl::AlgoBase* get_algo_from_desc(
            const AlgorithmDesc& desc);

private:
    class AlgoDirect;
    class AlgoDirectStride2;
    class AlgoFP32WinogradF63_8x8;
    class AlgoFP32WinogradF23_8x8;
    class AlgoDirectAvx2Stride1Int8;
    class AlgoAVX2DirectConvStride2;
    class AlgoChanWiseAvx2Stride1Qint8;
    class AlgoChanWiseAvx2Stride2Qint8;
#if MEGDNN_X86_WITH_MKL_DNN
    class AlgoMkldnnConv;
    class AlgoMkldnnQint8;
    class AlgoMkldnnMatmulQint8;
#endif
    class AlgoPack;

    static const AlgoPack& algo_pack();
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
