/**
 * \file dnn/src/cuda/cvt_color/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/cvt_color/opr_impl.h"
#include "src/cuda/cvt_color/cvt_color.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/cv/cvt_color.h"

#include <type_traits>

namespace megdnn {
namespace cuda {

using namespace megcv;
using namespace cvt_color;


void CvtColorImpl::cvt_color_exec_8u(_megdnn_tensor_in src_tensor,
                                     _megdnn_tensor_in dst_tensor) {
    auto stream = cuda_stream(this->handle());
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<uchar> src = TensorND2Mat<uchar>(src_tensor, i);
        Mat<uchar> dst = TensorND2Mat<uchar>(dst_tensor, i);

        cvt_color_8u_proxy(src.ptr(), dst.ptr(), src.rows(), src.cols(),
                           src.step(), dst.rows(), dst.cols(), dst.step(),
                           static_cast<uint32_t>(param().mode), stream);
    }
}

void CvtColorImpl::cvt_color_exec_32f(_megdnn_tensor_in src_tensor,
                                      _megdnn_tensor_in dst_tensor) {
    auto stream = cuda_stream(this->handle());
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<float> src = TensorND2Mat<float>(src_tensor, i);
        Mat<float> dst = TensorND2Mat<float>(dst_tensor, i);

        cvt_color_32f_proxy(src.ptr(), dst.ptr(), src.rows(), src.cols(),
                            src.step(), dst.rows(), dst.cols(), dst.step(),
                            static_cast<uint32_t>(param().mode), stream);
    }
}

void CvtColorImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                        _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);

    if (dst.layout.dtype == dtype::Float32()) {
        cvt_color_exec_32f(src, dst);
    } else if (dst.layout.dtype == dtype::Uint8()) {
        cvt_color_exec_8u(src, dst);
    } else {
        megdnn_throw("Unsupported datatype of Resize optr.");
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen