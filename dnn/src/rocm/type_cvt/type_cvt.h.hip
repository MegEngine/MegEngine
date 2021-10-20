/**
 * \file dnn/src/rocm/type_cvt/type_cvt.h.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "hip_header.h"
#include "megdnn/basic_types.h"

namespace megdnn {
namespace rocm {

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2n(const TensorND& dest, const TensorND& src,
                      hipStream_t stream);

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2q(
        const TensorND& dest, const TensorND& src,
        const DTypeParam<dtype_dest>& param,
        hipStream_t stream);

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2n(
        const TensorND& dest, const TensorND& src,
        const DTypeParam<dtype_src>& param,
        hipStream_t stream);

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2q(
        const TensorND& dest, const TensorND& src,
        const DTypeParam<dtype_src>& src_param,
        const DTypeParam<dtype_dest>& dst_param,
        hipStream_t stream);

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
