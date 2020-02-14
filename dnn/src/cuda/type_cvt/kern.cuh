/**
 * \file dnn/src/cuda/type_cvt/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2n(const TensorND& dest, const TensorND& src,
                      cudaStream_t stream);

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2q(
        const TensorND& dest, const TensorND& src,
        const CudaDTypeParam<dtype_dest>& param,
        cudaStream_t stream);

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2n(
        const TensorND& dest, const TensorND& src,
        const CudaDTypeParam<dtype_src>& param,
        cudaStream_t stream);

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2q(
        const TensorND& dest, const TensorND& src,
        const CudaDTypeParam<dtype_src>& src_param,
        const CudaDTypeParam<dtype_dest>& dst_param,
        cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
