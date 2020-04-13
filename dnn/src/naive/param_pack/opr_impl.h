/**
 * \file dnn/src/naive/param_pack/opr_impl.h
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
namespace naive {

class ParamPackConcatImpl final : public ParamPackConcat {
public:
    using ParamPackConcat::ParamPackConcat;
    void exec(_megdnn_tensor_in srcs, _megdnn_tensor_in offsets,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorShapeArray&, const TensorShape&,
                                  const TensorShape&) override {
        return 0;
    }

private:
    template <typename T>
    void exec_internal(_megdnn_tensor_in srcs, int32_t* offsets,
                       _megdnn_tensor_out dst, _megdnn_workspace workspace);
};

}  // namespace naive
}  // namespace megdnn
