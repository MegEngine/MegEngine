/**
 * \file dnn/src/fallback/resize/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/naive/resize/opr_impl.h"

namespace megdnn {
namespace fallback {

class ResizeImpl : public naive::ResizeImpl {
public:
    using naive::ResizeImpl::ResizeImpl;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }

private:
    // ctype: C type of input data type.
    template <typename ctype>
    void kern_fallback(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_fallback_nhwc(const KernParam<ctype>& kern_param);

};  // class ResizeImpl

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
