/**
 * \file dnn/src/x86/type_cvt/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/type_cvt/opr_impl.h"
#include "src/x86/handle.h"

namespace megdnn {
namespace x86 {

class TypeCvtImpl: public fallback::TypeCvtImpl {
    public:
        using fallback::TypeCvtImpl::TypeCvtImpl;
        void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) override;
        bool is_thread_safe() const override { return true; }
};

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen

