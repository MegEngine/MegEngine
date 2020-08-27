/**
 * \file dnn/src/rocm/linspace/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace rocm {

class LinspaceImpl final: public Linspace {
    public:
        using Linspace::Linspace;
        void exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) override;
        size_t get_workspace_in_bytes(const TensorLayout &) override {
            return 0;
        }
};

} // namespace rocm
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
