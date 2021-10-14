/**
 * \file dnn/src/naive/powc/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace naive {

class PowCImpl : public PowC {
    template <typename T>
    void do_exec_ct(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i);

public:
    using PowC::PowC;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
