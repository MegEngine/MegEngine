/**
 * \file dnn/src/cuda/warp_affine/opr_impl.h
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
namespace cuda {

class WarpAffineImpl : public WarpAffine {
public:
    using WarpAffine::WarpAffine;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout& mat,
                                  const TensorLayout&) override {
        //! Use workspace to store the transform matrix if inverse is false
        //! use double for the workspace dtype as float may cause accuracy error
        return mat.total_nr_elems() * sizeof(double);
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
