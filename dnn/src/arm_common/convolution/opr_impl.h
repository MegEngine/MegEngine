/**
 * \file dnn/src/arm_common/convolution/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/common/utils.h"
#include "src/fallback/convolution/opr_impl.h"
#include "src/arm_common/conv_bias/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ConvBiasImpl;

class ConvolutionBackwardDataImpl
        : public fallback::ConvolutionBackwardDataImpl {
public:
    using fallback::ConvolutionBackwardDataImpl::ConvolutionBackwardDataImpl;

protected:
    class AlgoBase : public fallback::ConvolutionBackwardDataImpl::AlgoBase {
    protected:
        ~AlgoBase() = default;

    public:
        AlgoBase() : fallback::ConvolutionBackwardDataImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::ARM_COMMON;
        }
        virtual bool usable(fallback::ConvolutionBackwardDataImpl* opr,
                            const NCBKernSizeParam& param) const = 0;
        virtual size_t get_workspace(fallback::ConvolutionBackwardDataImpl* opr,
                                     const NCBKernSizeParam& param) const = 0;
        virtual ncb_kern_t dispatch_kern(
                fallback::ConvolutionBackwardDataImpl* opr,
                const NCBKernSizeParam& param) const = 0;
    };

    ncb_kern_t ncb_1g_dispatch_kern(Algorithm* algo,
                                    const NCBKernSizeParam& param) override;

    size_t ncb_1g_get_workspace(Algorithm* algo,
                                const NCBKernSizeParam& param) override;

    const char* get_algorithm_set_name() const override;

    SmallVector<fallback::ConvolutionBackwardDataImpl::AlgoBase*>
    get_all_packed_algo() override;

public:
    MEGDNN_FB_DECL_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl);

private:
#if __ARM_FEATURE_DOTPROD
    class AlgoSdot8DirectStride1;
    class AlgoSdot8DirectStride2;
    class AlgoUdot8DirectStride1;
    class AlgoUdot8DirectStride2;
#endif
    class AlgoPack;
    static const AlgoPack& algo_pack();
};

}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
