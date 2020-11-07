/**
 * \file dnn/src/aarch64/conv_bias/opr_impl.h
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
#include "src/arm_common/conv_bias/opr_impl.h"

namespace megdnn {
namespace aarch64 {

class ConvBiasImpl : public arm_common::ConvBiasImpl {
public:
    using arm_common::ConvBiasImpl::ConvBiasImpl;
    class AlgoBase : public arm_common::ConvBiasImpl::AlgoBase {
    public:
        AlgoBase() : arm_common::ConvBiasImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::AARCH64;
        }
    };

    SmallVector<fallback::ConvBiasImpl::AlgoBase*> algo_pack() override;

protected:
    const char* get_algorithm_set_name() const override;

private:
    class AlgoF32DirectStride2;
    class AlgoS8MatrixMul;
    class AlgoQU8MatrixMul;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16DirectStride2;
#endif
    class AlgoPack;
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
