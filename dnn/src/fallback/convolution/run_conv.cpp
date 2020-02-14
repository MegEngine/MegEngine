/**
 * \file dnn/src/fallback/convolution/run_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/convolution/run_conv.h"

#include "src/common/utils.h"
#include "midout.h"

MIDOUT_DECL(megdnn_fallback_conv)

namespace {

bool can_run_xcorr_single_channel_templated(
        size_t /* IH */, size_t /* IW */,
        size_t FH, size_t FW,
        size_t /* OH */, size_t /* OW */,
        size_t /* PH */, size_t /* PW */,
        size_t /* SH */, size_t /* SW */)
{
    return FH == FW && FH >= 1 && FH <= 7;
}

template <int ker_size>
void run_xcorr_single_channel_templated_impl(const float * __restrict src,
        const float * __restrict filter,
        float * __restrict dst,
        size_t IH, size_t IW,
        size_t OH, size_t OW, size_t PH, size_t PW, size_t SH, size_t SW,
        bool add_mode)
{
#define divup(x, y) (((x)+(y)-1)/(y))
#define clear(oh, ow) if (!add_mode) { dst[(oh)*OW + (ow)] = 0; }
#define update(oh, ow, fh, fw) \
    dst[(oh)*OW + (ow)] += filter[(fh)*ker_size + (fw)] * \
        src[((oh)*SH+(fh)-PH)*IW + ((ow)*SW+(fw)-PW)]
    // OH = (IH-ker_size)/stride+1
    // OW = (IW-ker_size)/stride+1
    // good region:
    //  oh*stride-anchor >= 0
    //  oh*stride-anchor+ker_size <= IH
    //  oh >= anchor/stride
    //  oh <= (IH+anchor-ker_size)/stride
    size_t oh_start = divup(PH, SH);
    size_t oh_end = IH+PH>=ker_size ? (IH+PH-ker_size)/SH+1 : 0;
    size_t ow_start = divup(PW, SW);
    size_t ow_end = IW+PW>=ker_size ? (IW+PW-ker_size)/SW+1 : 0;
    if (oh_start > oh_end) oh_start = oh_end = 0;
    if (ow_start > ow_end) ow_start = ow_end = 0;

    for (size_t oh = 0; oh < oh_start; ++oh)
    for (size_t ow = 0; ow < OW; ++ow) {
        clear(oh, ow);
        int ih = oh*SH - PH;
        int iw = ow*SW - PW;
        for (int fh = 0; fh < ker_size; ++fh) if (ih+fh >= 0 && ih+fh < (int)IH)
        for (int fw = 0; fw < ker_size; ++fw) if (iw+fw >= 0 && iw+fw < (int)IW)
        {
            update(oh, ow, fh, fw);
        }
    }
    for (size_t oh = oh_start; oh < oh_end; ++oh) {
        for (size_t ow = 0; ow < ow_start; ++ow) {
            clear(oh, ow);
            int iw = ow*SW - PW;
            for (int fh = 0; fh < ker_size; ++fh)
            for (int fw = 0; fw < ker_size; ++fw)
            {
                if (iw+fw >= 0 && iw+fw < (int)IW) update(oh, ow, fh, fw);
            }
        }
        for (size_t ow = ow_start; ow < ow_end; ++ow) {
            clear(oh, ow);
            for (int fh = 0; fh < ker_size; ++fh)
            for (int fw = 0; fw < ker_size; ++fw)
            {
                update(oh, ow, fh, fw);
            }
        }
        for (size_t ow = ow_end; ow < OW; ++ow) {
            clear(oh, ow);
            int iw = ow*SW - PW;
            for (int fh = 0; fh < ker_size; ++fh)
            for (int fw = 0; fw < ker_size; ++fw)
            {
                if (iw+fw >= 0 && iw+fw < (int)IW) update(oh, ow, fh, fw);
            }
        }
    }
    for (size_t oh = oh_end; oh < OH; ++oh) {
    for (size_t ow = 0; ow < OW; ++ow) {
        clear(oh, ow);
        int ih = oh*SH - PH;
        int iw = ow*SW - PW;
        for (int fh = 0; fh < ker_size; ++fh) if (ih+fh >= 0 && ih+fh < (int)IH)
        for (int fw = 0; fw < ker_size; ++fw) if (iw+fw >= 0 && iw+fw < (int)IW)
        {
            update(oh, ow, fh, fw);
        }
    }
    }
#undef divup
#undef clear
#undef update
}

void run_xcorr_single_channel_templated(
        const float *src, const float *filter, float *dst,
        size_t IH, size_t IW, size_t FH, size_t FW,
        size_t OH, size_t OW, size_t PH, size_t PW, size_t SH, size_t SW,
        bool add_mode)
{
    (void)FW;
#define DISPATCH(ker_size) \
    if (FH == ker_size) { \
        MIDOUT_BEGIN(megdnn_fallback_conv, ker_size) { \
            run_xcorr_single_channel_templated_impl<ker_size>( \
                    src, filter, dst, \
                    IH, IW, OH, OW, PH, PW, SH, SW, add_mode); \
        } MIDOUT_END(); \
        return; \
    }
    DISPATCH(1)
    DISPATCH(2)
    DISPATCH(3)
    DISPATCH(4)
    DISPATCH(5)
    DISPATCH(6)
    DISPATCH(7)
#undef DISPATCH
    megdnn_throw(megdnn_mangle(
                "internal error in conv template dispatching: impossible"));
}

void run_xcorr_single_channel_nontemplated(
        const float *src, const float *filter, float *dst,
        size_t IH, size_t IW, size_t FH_, size_t FW_,
        size_t OH, size_t OW, size_t PH, size_t PW, size_t SH, size_t SW,
        bool add_mode)
{
#define divup(x, y) (((x)+(y)-1)/(y))
#define clear(oh, ow) if (!add_mode) { dst[(oh)*OW + (ow)] = 0; }
#define update(oh, ow, fh, fw) \
    dst[(oh)*OW + (ow)] += filter[(fh)*FW + (fw)] * \
        src[((oh)*SH+(fh)-PH)*IW + ((ow)*SW+(fw)-PW)]
    // OH = (IH-ker_size)/stride+1
    // OW = (IW-ker_size)/stride+1
    // good region:
    //  oh*stride-anchor >= 0
    //  oh*stride-anchor+ker_size <= IH
    //  oh >= anchor/stride
    //  oh <= (IH+anchor-ker_size)/stride
    int FH = FH_, FW = FW_;
    size_t oh_start = divup(PH, SH);
    size_t oh_end = IH+PH>=FH_ ? (IH+PH-FH)/SH+1 : 0;
    size_t ow_start = divup(PW, SW);
    size_t ow_end = IW+PW>=FW_ ? (IW+PW-FW)/SW+1 : 0;
    if (oh_start > oh_end) oh_start = oh_end = 0;
    if (ow_start > ow_end) ow_start = ow_end = 0;
    for (size_t oh = 0; oh < oh_start; ++oh)
    for (size_t ow = 0; ow < OW; ++ow) {
        clear(oh, ow);
        int ih = oh*SH - PH;
        int iw = ow*SW - PW;
        for (int fh = 0; fh < FH; ++fh) if (ih+fh >= 0 && ih+fh < (int)IH)
        for (int fw = 0; fw < FW; ++fw) if (iw+fw >= 0 && iw+fw < (int)IW)
        {
            update(oh, ow, fh, fw);
        }
    }
    for (size_t oh = oh_start; oh < oh_end; ++oh) {
        for (size_t ow = 0; ow < ow_start; ++ow) {
            clear(oh, ow);
            int iw = ow*SW - PW;
            for (int fh = 0; fh < FH; ++fh)
            for (int fw = 0; fw < FW; ++fw)
            {
                if (iw+fw >= 0 && iw+fw < (int)IW) update(oh, ow, fh, fw);
            }
        }
        for (size_t ow = ow_start; ow < ow_end; ++ow) {
            clear(oh, ow);
            for (int fh = 0; fh < FH; ++fh)
            for (int fw = 0; fw < FW; ++fw)
            {
                update(oh, ow, fh, fw);
            }
        }
        for (size_t ow = ow_end; ow < OW; ++ow) {
            clear(oh, ow);
            int iw = ow*SW - PW;
            for (int fh = 0; fh < FH; ++fh)
            for (int fw = 0; fw < FW; ++fw)
            {
                if (iw+fw >= 0 && iw+fw < (int)IW) update(oh, ow, fh, fw);
            }
        }
    }
    for (size_t oh = oh_end; oh < OH; ++oh) {
    for (size_t ow = 0; ow < OW; ++ow) {
        clear(oh, ow);
        int ih = oh*SH - PH;
        int iw = ow*SW - PW;
        for (int fh = 0; fh < FH; ++fh) if (ih+fh >= 0 && ih+fh < (int)IH)
        for (int fw = 0; fw < FW; ++fw) if (iw+fw >= 0 && iw+fw < (int)IW)
        {
            update(oh, ow, fh, fw);
        }
    }
    }
#undef divup
#undef clear
#undef update
}

void run_xcorr_single_channel(const float *src, const float *filter, float *dst,
        size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OH, size_t OW,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        bool add_mode)
{
    if (can_run_xcorr_single_channel_templated(IH, IW, FH, FW, OH, OW,
                PH, PW, SH, SW)) {
        run_xcorr_single_channel_templated(src, filter, dst,
                IH, IW, FH, FW, OH, OW, PH, PW, SH, SW,
                add_mode);
    } else {
        MIDOUT_BEGIN(megdnn_fallback_conv, void) {
            run_xcorr_single_channel_nontemplated(src, filter, dst,
                    IH, IW, FH, FW, OH, OW, PH, PW, SH, SW,
                    add_mode);
        } MIDOUT_END();
    }
}

/*================ ConvolutionBackwardData =============*/

template <int ker_size>
void conv_backdata_single_channel_templated_impl(const float * __restrict diff,
        const float * __restrict filter,
        float * __restrict grad,
        size_t IH, size_t IW,
        size_t OH, size_t OW, size_t PH, size_t PW, size_t SH, size_t SW){
#define divup(x, y) (((x) + (y)-1) / (y))
#define update(oh, ow, fh, fw, val) \
    grad[(oh+fh)*OW + (ow+fw)] += filter[(fh)*ker_size + (fw)] * val
    size_t ih_start = divup(PH, SH);
    size_t ih_end = OH+PH>=ker_size ? (OH+PH-ker_size)/SH+1 : 0;
    size_t iw_start = divup(PW, SW);
    size_t iw_end = OW+PW>=ker_size ? (OW+PW-ker_size)/SW+1 : 0;
    if (ih_start > ih_end) ih_start = ih_end = 0;
    if (iw_start > iw_end) iw_start = iw_end = 0;
    for (size_t ih = 0; ih < ih_start; ++ih)
    for (size_t iw = 0; iw < IW; ++iw) {
        int oh = ih*SH - PH;
        int ow = iw*SW - PW;
        float val = diff[ih*IW + iw];
        for (int fh = 0; fh < ker_size; ++fh) if (oh+fh >= 0 && oh+fh < (int)OH)
        for (int fw = 0; fw < ker_size; ++fw) if (ow+fw >= 0 && ow+fw < (int)OW)
        {
            update(oh, ow, fh, fw, val);
        }
    }
    for (size_t ih = ih_start; ih < ih_end; ++ih) {
        int oh = ih*SH - PH;
        for (size_t iw = 0; iw < iw_start; ++iw) {
            int ow = iw*SW - PW;
            float val = diff[ih*IW + iw];
            for (int fh = 0; fh < ker_size; ++fh)
            for (int fw = 0; fw < ker_size; ++fw)
            {
                if (ow+fw >= 0 && ow+fw < (int)OW) update(oh, ow, fh, fw, val);
            }
        }
        for (size_t iw = iw_start; iw < iw_end; ++iw) {
            int ow = iw*SW - PW;
            float val = diff[ih*IW + iw];
            for (int fh = 0; fh < ker_size; ++fh)
            for (int fw = 0; fw < ker_size; ++fw)
            {
                update(oh, ow, fh, fw, val);
            }
        }
        for (size_t iw = iw_end; iw < IW; ++iw) {
            int ow = iw*SW - PW;
            float val = diff[ih*IW + iw];
            for (int fh = 0; fh < ker_size; ++fh)
            for (int fw = 0; fw < ker_size; ++fw)
            {
                if (ow+fw >= 0 && ow+fw < (int)OW) update(oh, ow, fh, fw, val);
            }
        }
    }
    for (size_t ih = ih_end; ih < IH; ++ih) {
    for (size_t iw = 0; iw < IW; ++iw) {
        int oh = ih*SH - PH;
        int ow = iw*SW - PW;
        float val = diff[ih*IW + iw];
        for (int fh = 0; fh < ker_size; ++fh) if (oh+fh >= 0 && oh+fh < (int)OH)
        for (int fw = 0; fw < ker_size; ++fw) if (ow+fw >= 0 && ow+fw < (int)OW)
        {
            update(oh, ow, fh, fw, val);
        }
    }
    }
#undef divup
#undef update
}

void conv_backdata_single_channel_templated(
        const float *src, const float *filter, float *dst,
        size_t IH, size_t IW, size_t FH, size_t FW,
        size_t OH, size_t OW, size_t PH, size_t PW, size_t SH, size_t SW)
{
    megdnn_ignore(FW);
#define DISPATCH(ker_size) \
    if (FH == ker_size) { \
        MIDOUT_BEGIN(megdnn_fallback_conv, ker_size) { \
            conv_backdata_single_channel_templated_impl<ker_size>( \
                    src, filter, dst, \
                    IH, IW, OH, OW, PH, PW, SH, SW); \
        } MIDOUT_END(); \
        return; \
    }
    DISPATCH(1)
    DISPATCH(2)
    DISPATCH(3)
    DISPATCH(4)
    DISPATCH(5)
    DISPATCH(6)
    DISPATCH(7)
#undef DISPATCH
    megdnn_throw(
            megdnn_mangle("internal error in conv_backdata template "
                          "dispatching: impossible"));
}

void conv_backdata_single_channel_nontemplated(
        const float *diff, const float *filter, float *grad,
        size_t IH, size_t IW, size_t FH_, size_t FW_,
        size_t OH, size_t OW, size_t PH, size_t PW, size_t SH, size_t SW){
#define divup(x, y) (((x) + (y)-1) / (y))
#define update(oh, ow, fh, fw, val) \
    grad[(oh+fh)*OW + (ow+fw)] += filter[(fh)*FW + (fw)] * val
    int FH = FH_, FW = FW_;
    size_t ih_start = divup(PH, SH);
    size_t ih_end = OH+PH>=FH_ ? (OH+PH-FH)/SH+1 : 0;
    size_t iw_start = divup(PW, SW);
    size_t iw_end = OW+PW>=FW_ ? (OW+PW-FW)/SW+1 : 0;
    if (ih_start > ih_end) ih_start = ih_end = 0;
    if (iw_start > iw_end) iw_start = iw_end = 0;
    for (size_t ih = 0; ih < ih_start; ++ih)
    for (size_t iw = 0; iw < IW; ++iw) {
        int oh = ih*SH - PH;
        int ow = iw*SW - PW;
        float val = diff[ih*IW + iw];
        for (int fh = 0; fh < FH; ++fh) if (oh+fh >= 0 && oh+fh < (int)OH)
        for (int fw = 0; fw < FW; ++fw) if (ow+fw >= 0 && ow+fw < (int)OW)
        {
            update(oh, ow, fh, fw, val);
        }
    }
    for (size_t ih = ih_start; ih < ih_end; ++ih) {
        int oh = ih*SH - PH;
        for (size_t iw = 0; iw < iw_start; ++iw) {
            int ow = iw*SW - PW;
            float val = diff[ih*IW + iw];
            for (int fh = 0; fh < FH; ++fh)
            for (int fw = 0; fw < FW; ++fw)
            {
                if (ow+fw >= 0 && ow+fw < (int)OW) update(oh, ow, fh, fw, val);
            }
        }
        for (size_t iw = iw_start; iw < iw_end; ++iw) {
            int ow = iw*SW - PW;
            float val = diff[ih*IW + iw];
            for (int fh = 0; fh < FH; ++fh)
            for (int fw = 0; fw < FW; ++fw)
            {
                update(oh, ow, fh, fw, val);
            }
        }
        for (size_t iw = iw_end; iw < IW; ++iw) {
            int ow = iw*SW - PW;
            float val = diff[ih*IW + iw];
            for (int fh = 0; fh < FH; ++fh)
            for (int fw = 0; fw < FW; ++fw)
            {
                if (ow+fw >= 0 && ow+fw < (int)OW) update(oh, ow, fh, fw, val);
            }
        }
    }
    for (size_t ih = ih_end; ih < IH; ++ih) {
    for (size_t iw = 0; iw < IW; ++iw) {
        int oh = ih*SH - PH;
        int ow = iw*SW - PW;
        float val = diff[ih*IW + iw];
        for (int fh = 0; fh < FH; ++fh) if (oh+fh >= 0 && oh+fh < (int)OH)
        for (int fw = 0; fw < FW; ++fw) if (ow+fw >= 0 && ow+fw < (int)OW)
        {
            update(oh, ow, fh, fw, val);
        }
    }
    }
#undef divup
#undef update
}

void conv_backdata_single_channel(const float *diff, const float *filter, float *grad,
        size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OH, size_t OW,
        size_t PH, size_t PW,
        size_t SH, size_t SW)
{
    if (can_run_xcorr_single_channel_templated(IH, IW, FH, FW, OH, OW,
                PH, PW, SH, SW)) {
        conv_backdata_single_channel_templated(diff, filter, grad,
                IH, IW, FH, FW, OH, OW, PH, PW, SH, SW);
    } else {
        MIDOUT_BEGIN(megdnn_fallback_conv, void) {
            conv_backdata_single_channel_nontemplated(diff, filter, grad,
                    IH, IW, FH, FW, OH, OW, PH, PW, SH, SW);
        } MIDOUT_END();
    }
}

} // anonymous namespace

namespace megdnn {
namespace fallback {
namespace convolution {

void run_conv(const float *src, const float *filter, float *dst, void *workspace,
        size_t IH, size_t IW, size_t IC,
        size_t FH, size_t FW,
        size_t OH, size_t OW, size_t OC,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        bool xcorr)
{
    for (size_t oc = 0; oc < OC; ++oc)
    for (size_t ic = 0; ic < IC; ++ic)
    {
        // ut for untransposed
        const float *fut = filter + oc*IC*FH*FW + ic*FH*FW;
        const float *f;
        if (!xcorr) {
            // need transpose
            f = (float *)workspace;
            for (size_t fh = 0; fh < FH; ++fh)
            for (size_t fw = 0; fw < FW; ++fw)
            {
                ((float *)f)[fh*FW + fw] = fut[(FH-fh-1)*FW + (FW-fw-1)];
            }
        } else {
            // do not need transpose
            f = fut;
        }
        run_xcorr_single_channel(src + ic*IH*IW, f, dst + oc*OH*OW,
                IH, IW, FH, FW, OH, OW, PH, PW, SH, SW,
                ic > 0);
    }
}

void run_conv_backward_data(const float* diff, const float* filter, float* grad,
                            void* workspace, size_t IH, size_t IW, size_t IC,
                            size_t FH, size_t FW, size_t OH, size_t OW,
                            size_t OC, size_t PH, size_t PW, size_t SH,
                            size_t SW, bool xcorr) {
    std::memset(grad, 0, sizeof(float) * IC * OH * OW);
    for (size_t oc = 0; oc < OC; ++oc)
        for (size_t ic = 0; ic < IC; ++ic) {
            // ut for untransposed
            const float* fut = filter + oc * IC * FH * FW + ic * FH * FW;
            const float* f;
            if (!xcorr) {
                // need transpose
                f = (float*)workspace;
                for (size_t fh = 0; fh < FH; ++fh)
                    for (size_t fw = 0; fw < FW; ++fw) {
                        ((float*)f)[fh * FW + fw] =
                                fut[(FH - fh - 1) * FW + (FW - fw - 1)];
                    }
            } else {
                // do not need transpose
                f = fut;
            }
            conv_backdata_single_channel(diff + oc * IH * IW, f,
                                         grad + ic * OH * OW, IH, IW, FH, FW,
                                         OH, OW, PH, PW, SH, SW);
        }
}

} // namespace convolution
} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
