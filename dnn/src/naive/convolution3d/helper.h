/**
 * \file dnn/src/naive/convolution3d/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace naive {
namespace convolution3d {

struct GroupCounter {
    const size_t grp_size;
    size_t cur_grp = 0, cur_off = 0;

    explicit GroupCounter(size_t grp_size):
        grp_size{grp_size}
    {
    }

    void next() {
        if ((++ cur_off) == grp_size) {
            cur_off = 0;
            ++ cur_grp;
        }
    }
};

struct StrategyFwd {
    template<typename st, typename ft, typename dt>
    static void on(st &s, ft &f, dt &d) {
        d += s * f;
    }

    template<typename dt>
    static void init_dval(dt &d) {
        d = 0;
    }
};

struct StrategyBwdData {
    template<typename st, typename ft, typename dt>
    static void on(st &s, ft &f, dt &d) {
        s += f * d;
    } template<typename dt> static void init_dval(dt &) { }
};

struct StrategyBwdFlt {
    template<typename st, typename ft, typename dt>
    static void on(st &s, ft &f, dt &d) {
        f += s * d;
    }

    template<typename dt>
    static void init_dval(dt &) {
    }
};

template <typename stype, typename ftype, typename dtype, class Strategy>
void compute3d(_megdnn_tensor_in src,
        ftype * __restrict fptr,
        _megdnn_tensor_out dst,
        const Convolution3D::CanonizedFilterMeta &filter_meta) {
    size_t spatial_start, channel_pos;
    using Format = param::Convolution3D::Format;
    if (filter_meta.format == Format::NCDHW) {
        spatial_start = 2;
        channel_pos = 1;
    } else {
        megdnn_assert(filter_meta.format == Format::NDHWC,
                "invalid conv format");
        spatial_start = 1;
        channel_pos = 4;
    }
    auto N = src.layout.shape[0],
         ID = src.layout.shape[spatial_start],
         IH = src.layout.shape[spatial_start + 1],
         IW = src.layout.shape[spatial_start + 2];
    auto FD = filter_meta.spatial[0], 
         FH = filter_meta.spatial[1],
         FW = filter_meta.spatial[2];
    auto OC = dst.layout.shape[channel_pos],
         OD = dst.layout.shape[spatial_start],
         OH = dst.layout.shape[spatial_start + 1],
         OW = dst.layout.shape[spatial_start + 2];

    size_t FS_G, FS_OC, FS_IC, FS_SPATIAL;
    if (filter_meta.format == Format::NCDHW) { 
        // g, oc, ic, fd, fh, fw
        FS_SPATIAL = 1;
        FS_IC = FD*FH*FW;
        FS_OC = FS_IC * filter_meta.icpg;
        FS_G = FS_OC * filter_meta.ocpg;
    } else {
        // g, oc, fd, fh, fw, ic
        megdnn_assert(filter_meta.format == Format::NDHWC,
                "invalid conv format");
        FS_IC = 1;
        FS_SPATIAL = filter_meta.icpg;
        FS_OC = FS_SPATIAL * FD*FH*FW;
        FS_G = FS_OC * filter_meta.ocpg;
    }

    int pd = filter_meta.padding[0], 
        ph = filter_meta.padding[1], 
        pw = filter_meta.padding[2];
    size_t sd = filter_meta.stride[0], 
           sh = filter_meta.stride[1],
           sw = filter_meta.stride[2];
    int dd = filter_meta.dilation[0], 
        dh = filter_meta.dilation[1],
        dw = filter_meta.dilation[2];
    stype * __restrict sptr = src.ptr<stype>();
    dtype * __restrict dptr = dst.ptr<dtype>();

    int d_offset = -pd, 
        h_offset = -ph, 
        w_offset = -pw;
   
   if (filter_meta.should_flip) { 
        d_offset += filter_meta.dilated_spatial[0] - 1;
        h_offset += filter_meta.dilated_spatial[1] - 1;
        w_offset += filter_meta.dilated_spatial[2] - 1;
        dd = -dd;
        dh = -dh;
        dw = -dw;
    }

    auto get_linear_addr = [&filter_meta](size_t n, size_t c, size_t d, size_t h, size_t w,
            const TensorLayout &layout) -> size_t {
        if (filter_meta.format == Format::NCDHW) {

            return n*layout.stride[0] +
                c*layout.stride[1] +
                d*layout.stride[2] +
                h*layout.stride[3] +
                w*layout.stride[4];
        } else {
            megdnn_assert(filter_meta.format == Format::NDHWC,
                    "invalid conv format");
            return n*layout.stride[0] +
                d*layout.stride[1] +
                h*layout.stride[2] +
                w*layout.stride[3] +
                c*layout.stride[4];
        }
    };
    for (size_t n = 0; n < N; ++n) {
        GroupCounter gc_out{filter_meta.ocpg};
        for (size_t oc = 0; oc < OC; ++oc, gc_out.next())
        for (size_t od = 0; od < OD; ++od)
        for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow) {
            dtype &dval = dptr[
                get_linear_addr(n, oc, od, oh, ow, dst.layout)];
            Strategy::init_dval(dval);
            for (size_t fd = 0; fd < FD; ++fd)
            for (size_t fh = 0; fh < FH; ++fh)
            for (size_t fw = 0; fw < FW; ++fw) {
                size_t id = sd*od + fd*dd + d_offset,
                       ih = sh*oh + fh*dh + h_offset,
                       iw = sw*ow + fw*dw + w_offset;
                if (id < ID && ih < IH && iw < IW) {
                    size_t ic0 = gc_out.cur_grp * filter_meta.icpg,
                           ic1 = ic0 + filter_meta.icpg;
                    for (size_t ic = ic0; ic < ic1; ++ ic) {
                        stype &sval = sptr[
                            get_linear_addr(n, ic, id, ih, iw, src.layout)];
                        ftype &fval = fptr[
                            gc_out.cur_grp * FS_G +
                            gc_out.cur_off * FS_OC +
                            (ic - ic0) * FS_IC +
                            (fd * FH * FW + fh * FW + fw) * FS_SPATIAL];
                        Strategy::on(sval, fval, dval);
                    }
                }
            }
        }
    }
}

//! forward with only filter ptr
template <typename stype, typename ftype, typename dtype>
void forward(_megdnn_tensor_in src,
        const ftype *fptr,
        _megdnn_tensor_out dst,
        const Convolution3D::CanonizedFilterMeta &filter_meta) {
    megdnn_assert(filter_meta.spatial_ndim == 3);
    compute3d<stype, ftype, dtype, StrategyFwd>(
            src, const_cast<ftype*>(fptr), dst, filter_meta);
}

//! forward with full filter (for API compatibility)
template <typename stype, typename ftype, typename dtype>
void forward(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        const Convolution3D::CanonizedFilterMeta &filter_meta) {
    return forward<stype, ftype, dtype>(src, filter.ptr<ftype>(), dst,
            filter_meta);
}

template <typename ftype, typename dtype, typename gtype>
void backward_data(_megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        const Convolution3D::CanonizedFilterMeta &filter_meta) {
    megdnn_assert(grad.layout.is_contiguous());
    memset(grad.raw_ptr, 0, grad.layout.span().dist_byte());
    megdnn_assert(filter_meta.spatial_ndim == 3);
    compute3d<gtype, ftype, dtype, StrategyBwdData>(
            grad, filter.ptr<ftype>(), diff, filter_meta);
}

template <typename stype, typename dtype, typename gtype>
void backward_filter(_megdnn_tensor_in src,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        const Convolution3D::CanonizedFilterMeta &filter_meta) {
    megdnn_assert(grad.layout.is_contiguous());
    memset(grad.raw_ptr, 0, grad.layout.span().dist_byte());
    megdnn_assert(filter_meta.spatial_ndim == 3);
    compute3d<stype, gtype, dtype, StrategyBwdFlt>(
            src, grad.ptr<gtype>(), diff, filter_meta);
}

} // namespace convolution
} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen

