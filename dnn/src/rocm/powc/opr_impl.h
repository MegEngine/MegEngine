/**
 * \file dnn/src/rocm/powc/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs/general.h"

namespace megdnn {
namespace rocm {

class PowCImpl final : public PowC {
public:
    using PowC::PowC;
    void do_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                 const float* exp_f, const int* exp_i) override;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen

