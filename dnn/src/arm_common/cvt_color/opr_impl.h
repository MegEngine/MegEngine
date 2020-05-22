/**
 * \file dnn/src/arm_common/cvt_color/opr_impl.h
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
namespace arm_common {

class CvtColorImpl: public CvtColor {
    private:
        template <typename T>
        void cvt_color_exec(_megdnn_tensor_in src,
            _megdnn_tensor_out dst);

    public:
        using CvtColor::CvtColor;

        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;

        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &) override
        {
            return 0;
        }
};

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
