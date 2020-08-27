/**
 * \file dnn/src/rocm/convolution/forward/matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/rocm/utils.h"
#include "src/rocm/utils.h.hip"
#include "src/rocm/convolution/helper.h"
#include "src/rocm/convolution/im2col.h.hip"

using namespace megdnn;
using namespace rocm;

bool ConvolutionForwardImpl::AlgoMatmul::is_available(
        const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.src_layout->dtype.category() == DTypeCategory::FLOAT &&
           args.opr->param().compute_mode != Param::ComputeMode::FLOAT32 &&
           fm.group == 1 && fm.spatial_ndim == 2;
}

size_t ConvolutionForwardImpl::AlgoMatmul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return matmul_get_workspace_bundle(args).total_size_in_bytes();
}

void ConvolutionForwardImpl::AlgoMatmul::exec(const ExecArgs& args) const {
#define cb(DType)                                        \
    if (args.src_layout->dtype == DType()) {             \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_internal<ctype>(args);                      \
        return;                                          \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb

    megdnn_assert_internal(0);
}

template <typename T>
void ConvolutionForwardImpl::AlgoMatmul::exec_internal(const ExecArgs& args) {
    auto&& fm = args.filter_meta;
    size_t N = args.src_layout->shape[0], IC = fm.icpg,
           IH = args.src_layout->shape[2], IW = args.src_layout->shape[3],
           OC = fm.ocpg, OH = args.dst_layout->shape[2],
           OW = args.dst_layout->shape[3], FH = fm.spatial[0],
           FW = fm.spatial[1], PH = fm.padding[0], PW = fm.padding[1],
           SH = fm.stride[0], SW = fm.stride[1], DH = fm.dilation[0],
           DW = fm.dilation[1];
    auto stream = hip_stream(args.handle);
    auto wbundle = matmul_get_workspace_bundle(args);
    wbundle.set(args.workspace.raw_ptr);
    T* dst_t = static_cast<T*>(wbundle.get(0));
    T* col = static_cast<T*>(wbundle.get(1));
    convolution::im2col<T>(args.src_tensor->ptr<T>(), col, N,
                           args.src_layout->stride[0], IC, IH, IW, FH, FW, OH,
                           OW, PH, PW, SH, SW, DH, DW, stream);
    TensorLayout Al({OC, IC * FH * FW}, typename DTypeTrait<T>::dtype()),
            Bl({IC * FH * FW, OH * OW * N}, typename DTypeTrait<T>::dtype()),
            Cl({OC, OH * OW * N}, typename DTypeTrait<T>::dtype());
    TensorND A(args.filter_tensor->ptr<T>(), Al), B(col, Bl), C(dst_t, Cl);
    if (fm.should_flip) {
        convolution::flip_filter(args, wbundle.get_workspace(2), A.raw_ptr);
    }
    args.handle->matmul_opr()->exec(A, B, C, Workspace());
    TensorLayout C2l({OC * OH * OW, N}, typename DTypeTrait<T>::dtype()),
            C3l = C2l;
    C3l.stride[0] = 1;
    C3l.stride[1] = args.dst_tensor->layout.stride[0];
    TensorND C2(dst_t, C2l);
    TensorND C3(args.dst_tensor->ptr<T>(), C3l);
    args.handle->relayout_opr()->exec(C2, C3);
}

// vim: syntax=cpp.doxygen
