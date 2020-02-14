/**
 * \file dnn/src/x86/lrn/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/lrn/opr_impl.h"

namespace megdnn {
namespace x86 {

class LRNImpl: public naive::LRNForwardImpl {
    public:
        using naive::LRNForwardImpl::LRNForwardImpl;
        void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
};

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen

