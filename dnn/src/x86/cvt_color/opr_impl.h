/**
 * \file dnn/src/x86/cvt_color/opr_impl.h
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
#include "src/naive/cvt_color/opr_impl.h"

namespace megdnn {
namespace x86 {

class CvtColorImpl: public naive::CvtColorImpl {
    private:
        template <typename T>
        void cvt_color_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst);

    public:
        using naive::CvtColorImpl::CvtColorImpl;

        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;
};

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
