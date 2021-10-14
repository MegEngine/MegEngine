/**
 * \file dnn/src/fallback/powc/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/naive/powc/opr_impl.h"

namespace megdnn {
namespace fallback {

class PowCImpl final : public naive::PowCImpl {
    template <typename T>
    void do_exec_ct(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i);

public:
    using naive::PowCImpl::PowCImpl;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
