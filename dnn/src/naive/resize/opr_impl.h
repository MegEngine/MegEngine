/**
 * \file dnn/src/naive/resize/opr_impl.h
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
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class ResizeImpl : public Resize {
public:
    using Format = Param::Format;
    template <typename ctype>
    struct KernParam {
        Format format;
        size_t n, c, ih, iw, oh, ow;
        ptrdiff_t s_in, s_ic, s_ih, s_iw;
        ctype *sptr, *dptr;
        Workspace workspace;

        static KernParam from_tensors(Format format, _megdnn_tensor_in src,
                                      _megdnn_tensor_out dst,
                                      _megdnn_workspace workspace);
    };

    using Resize::Resize;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }
private:
    // ctype: C type of input data type.
    template <typename ctype>
    void kern_naive(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_naive_nhwc(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_naive_nhwcd4(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_naive_nchw4(const KernParam<ctype>& kern_param);

};  // class ResizeImpl

#define UNPACK_RESIZE_FWD_KERN_PARAM(p)                                \
    auto N = p.n, C = p.c, IH = p.ih, IW = p.iw, OH = p.oh, OW = p.ow; \
    ctype* __restrict sptr = p.sptr;                                   \
    ctype* __restrict dptr = p.dptr;

#define UNPACK_RESIZE_FWD_KERN_PARAM_WITH_STRIDE(p)                  \
    UNPACK_RESIZE_FWD_KERN_PARAM(p)                                  \
    auto S_IN = p.s_in, S_IC = p.s_ic, S_IH = p.s_ih, S_IW = p.s_iw;

class ResizeBackwardImpl: public ResizeBackward {
public:
    using ResizeBackward::ResizeBackward;
    void exec(_megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
