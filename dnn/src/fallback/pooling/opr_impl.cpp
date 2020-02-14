/**
 * \file dnn/src/fallback/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/pooling/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_pooling)

namespace megdnn {
namespace fallback {
namespace pooling {

void w3x3_w1x1_1d(const float *src, float *dst,
        size_t I, size_t O, size_t P)
{
    const float * __restrict src_ = src;
    float * __restrict dst_ = dst;
    if (P == 0) {
    } else if (P == 1) {
        dst_[0] = std::max(src_[0], src_[1]);
    } else if (P == 2) {
        dst_[0] = src_[0];
        dst_[1] = std::max(src_[0], src_[1]);
    }
    for (size_t o = P; o+P < O; ++o) {
        size_t i = o-P;
        dst_[o] = std::max(std::max(src_[i], src_[i+1]), src_[i+2]);
    }
    if (P == 0) {
    } else if (P == 1) {
        dst_[O-1] = std::max(src_[I-1], src_[I-2]);
    } else if (P == 2) {
        dst_[O-1] = src_[I-1];
        dst_[O-2] = std::max(src_[I-1], src_[I-2]);
    }
}

void w3x3_s1x1(const float *src, float *dst,
        size_t IH, size_t IW,
        size_t OH, size_t OW,
        size_t PH, size_t PW)
{
    // Let tmp[i][j] = max(src[i][j'], src[i][j'+1], ..., src[i][j'+WW-1]),
    // where (i, j') is the corresponding src pixel coordinate for
    // dst pixel coordinate (i, j).
    // cache[] stores lines of tmp in a sliding-window way.
    // cache[0] denotes the line that is currently being processed.
    // The length of each line is OW.
    std::vector<float *> cache(3, nullptr);
    auto shuffle = [&cache]() {
        auto len = cache.size();
        auto ptr = cache.data();
        auto last = cache.back();
        std::memmove(ptr+1, ptr, sizeof(float *) * (len-1));
        cache[0] = last;
    };
    for (auto &ptr: cache) {
        ptr = new float[OW];
        megdnn_assert(ptr, "new failed (possibly lack of memory?)");
    }
    // initialize all lines with the least optimized val (-infinity)
    for (auto ptr: cache) {
        std::fill(ptr, ptr + OW,
                -std::numeric_limits<float>::max());
    }
    // init situation where oh == -1
    {
        int oh = -1;
        // rb for right bracket
        int ih_rb = oh - PH + 3;
        for (int ih = 0; ih < ih_rb; ++ih) {
            shuffle();
            w3x3_w1x1_1d(src + ih*IW, cache[0],
                    IW, OW, PW);
        }
    }
    for (int oh = 0; oh < static_cast<int>(OH); ++oh) {
        shuffle();
        int ih = oh - PH + 3 - 1;
        if (ih >= static_cast<int>(IH)) {
            std::fill(cache[0], cache[0] + OW,
                    -std::numeric_limits<float>::max());
        } else {
            w3x3_w1x1_1d(src + ih*IW, cache[0],
                    IW, OW, PW);
        }
        float * __restrict dst_ = dst;
        for (size_t ow = 0; ow < OW; ++ow) {
            float res = std::max(cache[0][ow],
                    std::max(cache[1][ow], cache[2][ow]));
            dst_[oh*OW + ow] = res;
        }
    }
    // free
    for (auto ptr: cache) {
        delete[] ptr;
    }
}

void w2x2_s2x2_int8(const int8_t *src, int8_t *dst,
        size_t IH, size_t IW, size_t OH, size_t OW)
{
    megdnn_ignore(IH);
    for (size_t ih = 0; ih < OH*2; ++ih) {
        size_t oh = ih >> 1;
        const int8_t * __restrict sptr = src + ih*IW;
        int8_t * __restrict dptr = dst + oh*OW;
        if (ih & 1) {
            for (size_t ow = 0; ow < OW; ++ow) {
                dptr[ow] = std::max(dptr[ow],
                        std::max(sptr[ow*2], sptr[ow*2+1]));
            }
        } else {
            for (size_t ow = 0; ow < OW; ++ow) {
                dptr[ow] = std::max(sptr[ow*2], sptr[ow*2+1]);
            }
        }
    }
}

void w2x2_s2x2_avg_int8(const int8_t *src, int8_t *dst,
        size_t IH, size_t IW, size_t OH, size_t OW)
{
    megdnn_ignore(IH);
    for (size_t oh = 0; oh < OH; ++oh) {
        size_t ih = oh*2;
        const int8_t * __restrict sptr0 = src + (ih+0)*IW;
        const int8_t * __restrict sptr1 = src + (ih+1)*IW;
        int8_t * __restrict dptr = dst + oh*OW;
        for (size_t ow = 0; ow < OW; ++ow) {
            size_t iw = ow*2;
            int32_t v00 = sptr0[iw+0],
                    v01 = sptr0[iw+1],
                    v10 = sptr1[iw+0],
                    v11 = sptr1[iw+1];
            dptr[ow] = (v00+v01+v10+v11) / 4;
        }
    }
}

} // namespace pooling
} // namespace fallback
} // namespace megdnn

namespace megdnn {
namespace fallback {

void PoolingImpl::exec_w3x3_s1x1(_megdnn_tensor_in src,
        _megdnn_tensor_out dst)
{
    auto N = src.layout.shape[0], C = src.layout.shape[1];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    for (size_t nc = 0; nc < N*C; ++nc) {
        pooling::w3x3_s1x1(src.ptr<dt_float32>() + nc*IH*IW,
                dst.ptr<dt_float32>() + nc*OH*OW,
                IH, IW, OH, OW, param().pad_h, param().pad_w);
    }
}

void PoolingImpl::exec_w2x2_s2x2_int8(_megdnn_tensor_in src,
        _megdnn_tensor_out dst)
{
    auto N = src.layout.shape[0], C = src.layout.shape[1];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    for (size_t nc = 0; nc < N*C; ++nc) {
        pooling::w2x2_s2x2_int8(src.ptr<dt_int8>() + nc*IH*IW,
                dst.ptr<dt_int8>() + nc*OH*OW,
                IH, IW, OH, OW);
    }
}

void PoolingImpl::exec_w2x2_s2x2_avg_int8(_megdnn_tensor_in src,
        _megdnn_tensor_out dst)
{
    auto N = src.layout.shape[0], C = src.layout.shape[1];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    for (size_t nc = 0; nc < N*C; ++nc) {
        pooling::w2x2_s2x2_avg_int8(src.ptr<dt_int8>() + nc*IH*IW,
                dst.ptr<dt_int8>() + nc*OH*OW,
                IH, IW, OH, OW);
    }
}

void PoolingImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    if (src.layout.dtype == dtype::Float32() &&
            param().format == Param::Format::NCHW &&
            param().mode == Mode::MAX &&
            param().window_h == 3 && param().window_w == 3 &&
            param().stride_h == 1 && param().stride_w == 1 &&
            param().pad_h <= 2 && param().pad_w <= 2) {
        MIDOUT_BEGIN(megdnn_fallback_pooling, midout_iv(0)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(exec_w3x3_s1x1(src, dst));
        } MIDOUT_END();
        return;
    }
    // regular int conv case
    if (src.layout.dtype == dtype::Int8() &&
            param().mode == Mode::MAX &&
            param().format == Param::Format::NCHW &&
            param().window_h == 2 && param().window_w == 2 &&
            param().stride_h == 2 && param().stride_w == 2 &&
            param().pad_h == 0 && param().pad_w == 0) {
        MIDOUT_BEGIN(megdnn_fallback_pooling, midout_iv(1)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(exec_w2x2_s2x2_int8(src, dst));
        } MIDOUT_END();
        return;
    }
    // int8 2x2 AVERAGE case
    if (src.layout.dtype == dtype::Int8() &&
            param().mode == Mode::AVERAGE &&
            param().format == Param::Format::NCHW &&
            param().window_h == 2 && param().window_w == 2 &&
            param().stride_h == 2 && param().stride_w == 2 &&
            param().pad_h == 0 && param().pad_w == 0) {
        MIDOUT_BEGIN(megdnn_fallback_pooling, midout_iv(2)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(exec_w2x2_s2x2_avg_int8(src, dst));
        } MIDOUT_END();
        return;
    }
    // fallback to naive
    naive::PoolingForwardImpl::exec(src, dst, workspace);
}

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen

