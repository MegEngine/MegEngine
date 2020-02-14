/**
 * \file dnn/src/naive/local/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/local/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

using namespace megdnn;
using namespace naive;

LocalForwardImpl::float_noncontig_batch_kern
LocalForwardImpl::dispatch_float_noncontig_batch(
        const TensorLayout &src,
        const TensorLayout &/*filter*/,
        const TensorLayout &/*dst*/) {
    if (src.dtype == dtype::Float32()) {
        if (param().mode == Mode::CROSS_CORRELATION) {
            return &naive_kern<true, float>;
        } else {
            return &naive_kern<false, float>;
        }
    } else if (MEGDNN_FLOAT16_SELECT(src.dtype == dtype::Float16(), false)) {
        MEGDNN_INC_FLOAT16(
        megdnn_assert(src.dtype == dtype::Float16());
        if (param().mode == Mode::CROSS_CORRELATION) {
            return &naive_kern<true MEGDNN_COMMA dt_float16>;
        } else {
            return &naive_kern<false MEGDNN_COMMA dt_float16>;
        });
    } else {
        megdnn_assert_internal(false);
        return nullptr;
    }
}

template<bool is_xcorr, typename dtype>
void LocalForwardImpl::naive_kern(const FloatNoncontigBatchKernParam &param) {
    UNPACK_LOCAL_FLOAT_NONCONTIG_BATCH_KERN_PARAM(param, dtype);

    static_cast<void>(workspace);
    rep(n, N) rep(oc, OC) rep(oh, OH) rep(ow, OW) {
        auto &dval = dst[n*OUT_BS + oc*OH*OW + oh*OW + ow];
        dval = 0.0f;
        rep(fh, FH) rep(fw, FW) {
            size_t ih = SH*oh;
            size_t iw = SW*ow;
            if (is_xcorr) {
                ih += fh;
                iw += fw;
            } else {
                ih += FH-fh-1;
                iw += FW-fw-1;
            }
            ih -= PH;
            iw -= PW;
            if (ih < static_cast<size_t>(IH) && iw < static_cast<size_t>(IW)) {
                rep(ic, IC)  {
                    auto sval = src[n*INP_BS + ic*IH*IW + ih*IW + iw];
                    auto fval = filter[oh*OW*IC*FH*FW*OC + ow*IC*FH*FW*OC +
                        ic*FH*FW*OC + fh*FW*OC + fw*OC + oc];
                    dval += sval * fval;
                }
            }
        }
    }
}

void LocalForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    exec_use_float_noncontig_batch(src, filter, dst, workspace);
}

LocalForwardImpl::FloatNoncontigBatchKernParam
LocalForwardImpl::make_float_kern_param(
        _megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace) const {
    return {
        src.raw_ptr, filter.raw_ptr, dst.raw_ptr,
        // n
        src.layout.shape[0],
        // ic, ih, iw, oc, oh, ow, fh, fw
        src.layout.shape[1], src.layout.shape[2], src.layout.shape[3],
        dst.layout.shape[1], dst.layout.shape[2], dst.layout.shape[3],
        filter.layout.shape[3], filter.layout.shape[4],
        // ph, pw, sh, sw
        param().pad_h, param().pad_w, param().stride_h, param().stride_w,
        // inp_bs, out_bs
        src.layout.stride[0], dst.layout.stride[0],
        workspace.raw_ptr
    };
}

void LocalForwardImpl::exec_use_float_noncontig_batch(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {

    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    auto fp = make_float_kern_param(src, filter, dst, workspace);
    auto kptr = dispatch_float_noncontig_batch(
            src.layout, filter.layout, dst.layout);
    auto kern = [fp, kptr]() {
        kptr(fp);
    };
    static_cast<naive::HandleImpl*>(handle())->dispatch_kern(kern);
}

void LocalBackwardDataImpl::exec(_megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
    size_t N = grad.layout.shape[0], IC = grad.layout.shape[1],
         IH = grad.layout.shape[2], IW = grad.layout.shape[3];
    size_t FH = filter.layout.shape[3], FW = filter.layout.shape[4];
    size_t OC = diff.layout.shape[1],
         OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    size_t ph = param().pad_h, pw = param().pad_w;
    size_t sh = param().stride_h, sw = param().stride_w;
    auto gptr = grad.ptr<dt_float32>(),
         fptr = filter.ptr<dt_float32>(),
         hptr = diff.ptr<dt_float32>();
    auto mode = param().mode;
    auto kern = [=]() {
        memset(gptr, 0, sizeof(float_t) * N*IC*IH*IW);
        rep(n, N) rep(oc, OC) rep(oh, OH) rep(ow, OW) {
            //auto &hval = hptr[n*OC*OH*OW + oc*OH*OW + oh*OW + ow];
            auto &hval = hptr[((n * OC + oc) * OH + oh) * OW + ow];
            rep(ic, IC) rep(fh, FH) rep(fw, FW) {
                size_t ih = -ph + sh*oh;
                size_t iw = -pw + sw*ow;
                if (mode == Mode::CROSS_CORRELATION) {
                    ih += fh;
                    iw += fw;
                } else {
                    ih += FH-fh-1;
                    iw += FW-fw-1;
                }

                if (ih < IH && iw < IW) {
                    //auto &gval = gptr[n*IC*IH*IW + ic*IH*IW + ih*IW + iw];
                    //auto fval = fptr[oh*OW*IC*FH*FW*OC + ow*IC*FH*FW*OC +
                    //    ic*FH*FW*OC + fh*FW*OC + fw*OC + oc];

                    auto &gval = gptr[(((n * IC + ic) * IH) + ih) * IW + iw];
                    auto fval = fptr[((((oh * OW + 
                                         ow) * IC + 
                                         ic) * FH + 
                                         fh) * FW +
                                         fw) * OC + oc];
                    gval += fval * hval;
                }
            }
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

void LocalBackwardFilterImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    size_t N = src.layout.shape[0], IC = src.layout.shape[1],
         IH = src.layout.shape[2], IW = src.layout.shape[3];
    size_t FH = grad.layout.shape[3], FW = grad.layout.shape[4];
    size_t OC = diff.layout.shape[1],
         OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    size_t ph = param().pad_h, pw = param().pad_w;
    size_t sh = param().stride_h, sw = param().stride_w;
    auto gptr = grad.ptr<dt_float32>(),
         sptr = src.ptr<dt_float32>(),
         hptr = diff.ptr<dt_float32>();
    auto mode = param().mode;
    auto kern = [=]() {
        memset(gptr, 0, sizeof(float_t) * OH*OW*IC*FH*FW*OC);
        rep(n, N) rep(oc, OC) rep(oh, OH) rep(ow, OW) {
            //auto &hval = hptr[n*OC*OH*OW + oc*OH*OW + oh*OW + ow];
            auto &hval = hptr[((n * OC + oc) * OH + oh) * OW + ow];
            rep(ic, IC) rep(fh, FH) rep(fw, FW) {
                size_t ih = -ph + sh*oh;
                size_t iw = -pw + sw*ow;
                if (mode == Mode::CROSS_CORRELATION) {
                    ih += fh;
                    iw += fw;
                } else {
                    ih += FH-fh-1;
                    iw += FW-fw-1;
                }

                if (ih < IH && iw < IW) {
                    //auto sval = sptr[n*IC*IH*IW + ic*IH*IW + ih*IW + iw];
                    //auto &gval = gptr[oh*OW*IC*FH*FW*OC + ow*IC*FH*FW*OC +
                    //    ic*FH*FW*OC + fh*FW*OC + fw*OC + oc];

                    auto sval = sptr[((n * IC + ic) * IH + ih) * IW + iw];
                    auto &gval = gptr[((((oh * OW + 
                                         ow) * IC + 
                                         ic) * FH + 
                                         fh) * FW +
                                         fw) * OC + oc];
                    gval += sval * hval;
                }
            }
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

// vim: syntax=cpp.doxygen
