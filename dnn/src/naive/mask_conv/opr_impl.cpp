/**
 * \file dnn/src/naive/mask_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/dtype.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/mask_conv/opr_impl.h"

namespace {
using namespace megdnn;
template <typename ctype>
void mask_propagate_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                         size_t FH, size_t FW, size_t SH, size_t SW, size_t PH,
                         size_t PW, size_t DH, size_t DW) {
    size_t OH = dst.layout[0];
    size_t OW = dst.layout[1];
    size_t IH = src.layout[0];
    size_t IW = src.layout[1];
    auto src_ptr = src.ptr<ctype>();
    auto dst_ptr = dst.ptr<ctype>();
    memset(dst_ptr, 0, sizeof(ctype) * OH * OW);
    for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow) {
            bool decided = false;
            for (size_t fh = 0; fh < FH && !decided; ++fh) {
                for (size_t fw = 0; fw < FW && !decided; ++fw) {
                    size_t ih = oh * SH + fh * DH;
                    size_t iw = ow * SW + fw * DW;
                    if (ih < PH || ih >= IH + PH || iw < PW || iw >= IW + PW) {
                        continue;
                    }
                    if (src_ptr[(ih - PH) * IW + (iw - PW)] != 0) {
                        dst_ptr[oh * OW + ow] = 1;
                        decided = true;
                    }
                }
            }
        }
}

template <typename ctype>
void set_zero_by_mask(_megdnn_tensor_out dst, _megdnn_tensor_in mask) {
    auto mask_ptr = mask.ptr<ctype>();
    auto dst_ptr = dst.ptr<float>();
    for (size_t n = 0; n < dst.layout[0]; ++n)
        for (size_t oc = 0; oc < dst.layout[1]; ++oc) {
            for (size_t oh = 0; oh < dst.layout[2]; ++oh) {
                for (size_t ow = 0; ow < dst.layout[3]; ++ow) {
                    if (mask_ptr[oh * dst.layout[3] + ow] == 0) {
                        size_t dst_idx = n * dst.layout.stride[0] +
                                         oc * dst.layout.stride[1] +
                                         oh * dst.layout.stride[2] +
                                         ow * dst.layout.stride[3];
                        dst_ptr[dst_idx] = 0;
                    }
                }
            }
        }
}
}  // namespace

namespace megdnn {
namespace naive {

MaskConvForwardImpl::MaskConvForwardImpl(Handle* handle)
        : MaskConvForward(handle) {
    m_conv_opr = this->handle()->create_operator<Convolution>();
}

void MaskConvForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                               _megdnn_tensor_in mask, _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    MEGDNN_MARK_USED_VAR(mask);
    m_conv_opr->param() = this->param();
    m_conv_opr->exec(src, filter, dst, nullptr, workspace);
#define cb(DType)                                                         \
    if (mask.layout.dtype == DType()) {                                   \
        using ctype = typename DTypeTrait<DType>::ctype;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(set_zero_by_mask<ctype>(dst, mask)); \
        return;                                                           \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb);
#undef cb
    megdnn_assert_internal(0);
}

size_t MaskConvForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                   const TensorLayout& filter,
                                                   const TensorLayout& mask,
                                                   const TensorLayout& dst) {
    MEGDNN_MARK_USED_VAR(mask);
    m_conv_opr->param() = this->param();
    return m_conv_opr->get_workspace_in_bytes(src, filter, dst, nullptr);
}

void MaskPropagateImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                             _megdnn_workspace) {
    auto p = param();
#define cb(DType)                                                         \
    if (src.layout.dtype == DType()) {                                    \
        using ctype = typename DTypeTrait<DType>::ctype;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(mask_propagate_exec<ctype>(          \
                src, dst, p.kernel_h, p.kernel_w, p.stride_h, p.stride_w, \
                p.pad_h, p.pad_w, p.dilate_h, p.dilate_w));               \
        return;                                                           \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
    megdnn_assert_internal(0);
}
}  // namespace naive
}  // namespace megdnn
