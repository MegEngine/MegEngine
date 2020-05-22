/**
 * \file dnn/src/arm_common/separable_filter/opr_impl.h
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
class SeparableFilterImpl : public SeparableFilterForward {
public:
    using SeparableFilterForward::SeparableFilterForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter_x,
              _megdnn_tensor_in filter_y, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }

private:
    template <typename T>
    void separable_filter_exec(_megdnn_tensor_in src,
                               _megdnn_tensor_in filter_x,
                               _megdnn_tensor_in filter_y,
                               _megdnn_tensor_out dst);
    void separable_filter_exec_8u(_megdnn_tensor_in src,
                                  _megdnn_tensor_in filter_x,
                                  _megdnn_tensor_in filter_y,
                                  _megdnn_tensor_out dst);
};

} // namespace arm_common
} // namespace megdnn
// vim: syntax=cpp.doxygen
