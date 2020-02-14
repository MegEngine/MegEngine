/**
 * \file dnn/src/x86/elemwise_multi_type/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/elemwise_multi_type/opr_impl.h"

namespace megdnn {
namespace x86 {

class ElemwiseMultiTypeImpl : public fallback::ElemwiseMultiTypeImpl {

protected:
    void on_quantized_mode(const ElemwiseOpParamN<1>& param,
                           const TensorND& dst, Elemwise::Mode mode) override;

    void on_quantized_mode(const ElemwiseOpParamN<2>& param,
                           const TensorND& dst, Elemwise::Mode mode) override;

    void on_quantized_mode(const ElemwiseOpParamN<3>& param,
                           const TensorND& dst, Elemwise::Mode mode) override;

public:
    using fallback::ElemwiseMultiTypeImpl::ElemwiseMultiTypeImpl;
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
