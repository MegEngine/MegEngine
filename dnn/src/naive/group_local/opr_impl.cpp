/**
 * \file dnn/src/naive/group_local/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/group_local/opr_impl.h"

#include "src/naive/handle.h"
#include <cstring>

namespace {

template <typename dtype>
void forward(const dtype *src, const dtype *filter, dtype *dst,
        size_t N, size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OC, size_t OH, size_t OW,
        size_t group,
        size_t pad_h, size_t pad_w,
        size_t stride_h, size_t stride_w)
{
    size_t ICg = IC / group;
    size_t OCg = OC / group;
    for (size_t n = 0; n < N; ++n)
    for (size_t gid = 0; gid < group; ++gid)
    for (size_t ocg = 0; ocg < OCg; ++ocg)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    {
        float res = 0;
        size_t oc = gid*OCg + ocg;
        for (size_t fh = 0; fh < FH; ++fh)
        for (size_t fw = 0; fw < FW; ++fw)
        for (size_t icg = 0; icg < ICg; ++icg)
        {
            size_t ih = oh*stride_h - pad_h + fh;
            size_t iw = ow*stride_w - pad_w + fw;
            size_t ic = gid*ICg + icg;
            if (ih < IH && iw < IW) {
                auto fval = filter[((((((gid*OH+oh)*OW+ow)*ICg+
                                    icg)*FH+fh)*FW+fw)*OCg+ocg)];
                auto sval = src[n*IC*IH*IW + ic*IH*IW + ih*IW + iw];
                res += fval*sval;
            }
        }
        dst[n*OC*OH*OW + oc*OH*OW + oh*OW + ow] = res;
    }
}

void backward_data(const float *filter, const float *diff, float *grad,
        size_t N, size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OC, size_t OH, size_t OW,
        size_t group,
        size_t pad_h, size_t pad_w,
        size_t stride_h, size_t stride_w)
{
    auto ICg = IC / group;
    auto OCg = OC / group;
    memset(grad, 0, sizeof(float) * N*IC*IH*IW);
    for (size_t n = 0; n < N; ++n)
    for (size_t gid = 0; gid < group; ++gid)
    for (size_t ocg = 0; ocg < OCg; ++ocg)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    {
        size_t oc = gid*OCg + ocg;
        for (size_t fh = 0; fh < FH; ++fh)
        for (size_t fw = 0; fw < FW; ++fw)
        for (size_t icg = 0; icg < ICg; ++icg)
        {
            size_t ih = oh*stride_h - pad_h + fh;
            size_t iw = ow*stride_w - pad_w + fw;
            size_t ic = gid*ICg + icg;
            if (ih < IH && iw < IW) {
                auto fval = filter[((((((gid*OH+oh)*OW+ow)*ICg+
                                    icg)*FH+fh)*FW+fw)*OCg+ocg)];
                auto dval = diff[n*OC*OH*OW + oc*OH*OW + oh*OW + ow];
                auto &sval = grad[n*IC*IH*IW + ic*IH*IW + ih*IW + iw];
                sval += fval*dval;
            }
        }
    }
}

void backward_filter(const float *src, const float *diff, float *grad,
        size_t N, size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OC, size_t OH, size_t OW,
        size_t group,
        size_t pad_h, size_t pad_w,
        size_t stride_h, size_t stride_w)
{
    auto ICg = IC / group;
    auto OCg = OC / group;
    memset(grad, 0, sizeof(float) * group*OH*OW*ICg*FH*FW*OCg);
    for (size_t n = 0; n < N; ++n)
    for (size_t gid = 0; gid < group; ++gid)
    for (size_t ocg = 0; ocg < OCg; ++ocg)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    {
        size_t oc = gid*OCg + ocg;
        for (size_t fh = 0; fh < FH; ++fh)
        for (size_t fw = 0; fw < FW; ++fw)
        for (size_t icg = 0; icg < ICg; ++icg)
        {
            size_t ih = oh*stride_h - pad_h + fh;
            size_t iw = ow*stride_w - pad_w + fw;
            size_t ic = gid*ICg + icg;
            if (ih < IH && iw < IW) {
                auto sval = src[n*IC*IH*IW + ic*IH*IW + ih*IW + iw];
                auto &fval = grad[((((((gid*OH+oh)*OW+ow)*ICg+
                                    icg)*FH+fh)*FW+fw)*OCg+ocg)];
                auto dval = diff[n*OC*OH*OW + oc*OH*OW + oh*OW + ow];
                fval += sval*dval;
            }
        }
    }
}

} // anonymous namespace

namespace megdnn {
namespace naive {

void GroupLocalForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    auto N = src.layout.shape[0], IC = src.layout.shape[1],
         IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto group = filter.layout.shape[0];
    auto FH = filter.layout.shape[4], FW = filter.layout.shape[5];
    auto OC = dst.layout.shape[1],
         OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    if (src.layout.dtype == dtype::Float32() &&
        filter.layout.dtype == dtype::Float32() &&
        dst.layout.dtype == dtype::Float32()) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                forward(src.ptr<dt_float32>(), filter.ptr<dt_float32>(),
                        dst.ptr<dt_float32>(), N, IC, IH, IW, FH, FW, OC, OH,
                        OW, group, param().pad_h, param().pad_w,
                        param().stride_h, param().stride_w));
    } else if (MEGDNN_FLOAT16_SELECT(
                       src.layout.dtype == dtype::Float16() &&
                               filter.layout.dtype == dtype::Float16() &&
                               dst.layout.dtype == dtype::Float16(),
                       false)) {
        MEGDNN_INC_FLOAT16(MEGDNN_DISPATCH_CPU_KERN_OPR(forward(
                src.ptr<dt_float16>(), filter.ptr<dt_float16>(),
                dst.ptr<dt_float16>(), N, IC, IH, IW, FH, FW, OC, OH, OW, group,
                param().pad_h, param().pad_w, param().stride_h,
                param().stride_w)););

    } else {
        megdnn_assert_internal(false);  
    }
}

void GroupLocalBackwardDataImpl::exec(_megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
    auto N = grad.layout.shape[0], IC = grad.layout.shape[1],
         IH = grad.layout.shape[2], IW = grad.layout.shape[3];
    auto group = filter.layout.shape[0];
    auto FH = filter.layout.shape[4], FW = filter.layout.shape[5];
    auto OC = diff.layout.shape[1],
         OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    MEGDNN_DISPATCH_CPU_KERN_OPR(backward_data(filter.ptr<dt_float32>(),
                diff.ptr<dt_float32>(),
                grad.ptr<dt_float32>(),
                N, IC, IH, IW,
                FH, FW,
                OC, OH, OW,
                group, param().pad_h, param().pad_w,
                param().stride_h, param().stride_w));
}

void GroupLocalBackwardFilterImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    auto N = src.layout.shape[0], IC = src.layout.shape[1],
         IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto group = grad.layout.shape[0];
    auto FH = grad.layout.shape[4], FW = grad.layout.shape[5];
    auto OC = diff.layout.shape[1],
         OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    MEGDNN_DISPATCH_CPU_KERN_OPR(backward_filter(src.ptr<dt_float32>(),
                diff.ptr<dt_float32>(),
                grad.ptr<dt_float32>(),
                N, IC, IH, IW,
                FH, FW,
                OC, OH, OW,
                group, param().pad_h, param().pad_w,
                param().stride_h, param().stride_w));
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
