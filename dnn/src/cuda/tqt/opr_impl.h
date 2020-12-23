/**
 * \file dnn/src/cuda/tqt/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"
namespace megdnn {
namespace cuda {

class TQTForwardImpl final : public TQTForward {
public:
    using TQTForward::TQTForward;
    void exec(_megdnn_tensor_in input, _megdnn_tensor_in scale,
              _megdnn_tensor_out output, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, /* input */
                                  const TensorLayout&, /* scale */
                                  const TensorLayout& /* output */) override {
        return 0;
    }

private:
    void exec_noncontig(_megdnn_tensor_in input, _megdnn_tensor_in scale,
                        _megdnn_tensor_out output);
};

class TQTBackwardImpl final : public TQTBackward {
public:
    using TQTBackward::TQTBackward;
    void exec(_megdnn_tensor_in diff, _megdnn_tensor_in input,
              _megdnn_tensor_in scale, _megdnn_tensor_out grad_x,
              _megdnn_tensor_out grad_s, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& /* diff */,
                                  const TensorLayout& /* input */,
                                  const TensorLayout& /* scale */,
                                  const TensorLayout& /* grad_x */,
                                  const TensorLayout& /* grad_s */) override {
        return 0;
    }

private:
    void exec_noncontig(_megdnn_tensor_in diff, _megdnn_tensor_in input,
                        _megdnn_tensor_in scale, _megdnn_tensor_out grad_x,
                        _megdnn_tensor_out grad_s);
};

}  // namespace cuda
}  // namespace megdnn
