/**
 * \file dnn/src/common/local/local_def.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// simd_macro/*_helper.h should be included before including this file.
//
// The following functions would be defined in this file:
//
// void local_xcorr_MEGDNN_SIMD_NAME(const LocalKParam &kparam);
// void local_conv_MEGDNN_SIMD_NAME(const LocalKParam &kparam);
//

#include "src/common/local/local_decl.inl"

#include "src/common/utils.h"
#include "src/common/macro_helper.h"

namespace {

using namespace megdnn;

template <int N, int OC>
void local_xcorr_tpl(const LocalKParam &kparam) MEGDNN_SIMD_ATTRIBUTE_TARGET;
template <int N, int OC>
void local_xcorr_tpl(const LocalKParam &kparam)
{
    const float* src = static_cast<const float*>(kparam.src);
    const float* filter = static_cast<const float*>(kparam.filter);
    float* dst = static_cast<float*>(kparam.dst);
    float* workspace = static_cast<float*>(kparam.workspace);
    const int IC = kparam.ic, IH = kparam.ih, IW = kparam.iw, OH = kparam.oh,
              OW = kparam.ow, FH = kparam.fh, FW = kparam.fw;
    const uint32_t PH = kparam.ph, PW = kparam.pw, SH = kparam.sh,
                   SW = kparam.sw;
    const ptrdiff_t INP_BS = kparam.inp_bs, OUT_BS = kparam.out_bs;

    float *dst2 = workspace;
    const int width = MEGDNN_SIMD_WIDTH;
    // dst2 is (H, W, N, C)
    memset(dst2, 0, sizeof(float) * OH*OW*N*OC);
    float *dst2_hwnc = dst2;
    rep(oh, OH) rep(ow, OW) {
        const float *src_bak = src;
        rep(ic, IC) {
            rep(fh, FH) for (int fw = 0; fw < FW; ++fw, filter += OC) {
                int ih = -PH + oh*SH + fh;
                int iw = -PW + ow*SW + fw;
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
                float *dst2_bak = dst2;
                rep(n, N) {
                    float s = src[n*INP_BS + ih*IW + iw];
                    const float *filter_bak = filter;
                    MEGDNN_SIMD_TYPE vs = MEGDNN_SIMD_SET1(s);
                    int oc = 0;
                    for (; oc+4*width <= OC; oc += 4*width, filter += 4*width) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vf2 = MEGDNN_SIMD_LOADU(filter + 2*width);
                        MEGDNN_SIMD_TYPE vf3 = MEGDNN_SIMD_LOADU(filter + 3*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        MEGDNN_SIMD_TYPE vd2 = MEGDNN_SIMD_LOADU(dst2 + oc + 2*width);
                        MEGDNN_SIMD_TYPE vd3 = MEGDNN_SIMD_LOADU(dst2 + oc + 3*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        vd2 = MEGDNN_SIMD_FMADD(vf2, vs, vd2);
                        vd3 = MEGDNN_SIMD_FMADD(vf3, vs, vd3);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 2*width, vd2);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 3*width, vd3);
                    }
                    if (oc+2*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        oc += 2*width;
                        filter += 2*width;
                    }
                    if (oc+1*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        oc += 1*width;
                        filter += 1*width;
                    }
                    for (; oc < OC; ++oc, ++filter) {
                        dst2[oc] += s * (*filter);
                    }
                    filter = filter_bak;
                    dst2 += OC;
                }
                dst2 = dst2_bak;
            }
            src += IH*IW;
        }
        src = src_bak;
        dst2 += N*OC;
    }
    transpose_knc2nsck(dst2_hwnc, dst, OH * OW, N, OC, OUT_BS);
}
void local_xcorr_generic(const LocalKParam &kparam) MEGDNN_SIMD_ATTRIBUTE_TARGET;
void local_xcorr_generic(const LocalKParam &kparam) {
    UNPACK_LOCAL_FLOAT_NONCONTIG_BATCH_KERN_PARAM(kparam, float);

    float *dst2 = workspace;
    const int width = MEGDNN_SIMD_WIDTH;
    // dst2 is (H, W, N, C)
    memset(dst2, 0, sizeof(float) * OH*OW*N*OC);
    float *dst2_hwnc = dst2;
    rep(oh, OH) rep(ow, OW) {
        const float *src_bak = src;
        rep(ic, IC) {
            rep(fh, FH) for (int fw = 0; fw < FW; ++fw, filter += OC) {
                int ih = -PH + oh*SH + fh;
                int iw = -PW + ow*SW + fw;
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
                float *dst2_bak = dst2;
                rep(n, N) {
                    float s = src[n*INP_BS + ih*IW + iw];
                    const float *filter_bak = filter;
                    MEGDNN_SIMD_TYPE vs = MEGDNN_SIMD_SET1(s);
                    int oc = 0;
                    for (; oc+4*width <= OC; oc += 4*width, filter += 4*width) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vf2 = MEGDNN_SIMD_LOADU(filter + 2*width);
                        MEGDNN_SIMD_TYPE vf3 = MEGDNN_SIMD_LOADU(filter + 3*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        MEGDNN_SIMD_TYPE vd2 = MEGDNN_SIMD_LOADU(dst2 + oc + 2*width);
                        MEGDNN_SIMD_TYPE vd3 = MEGDNN_SIMD_LOADU(dst2 + oc + 3*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        vd2 = MEGDNN_SIMD_FMADD(vf2, vs, vd2);
                        vd3 = MEGDNN_SIMD_FMADD(vf3, vs, vd3);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 2*width, vd2);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 3*width, vd3);
                    }
                    if (oc+2*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        oc += 2*width;
                        filter += 2*width;
                    }
                    if (oc+1*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        oc += 1*width;
                        filter += 1*width;
                    }
                    for (; oc < OC; ++oc, ++filter) {
                        dst2[oc] += s * (*filter);
                    }
                    filter = filter_bak;
                    dst2 += OC;
                }
                dst2 = dst2_bak;
            }
            src += IH*IW;
        }
        src = src_bak;
        dst2 += N*OC;
    }
    transpose_knc2nsck(dst2_hwnc, dst, OH * OW, N, OC, OUT_BS);
}

template <int N, int OC>
void local_conv_tpl(const LocalKParam &kparam) MEGDNN_SIMD_ATTRIBUTE_TARGET;
template <int N, int OC>
void local_conv_tpl(const LocalKParam &kparam)
{
    const float* src = static_cast<const float*>(kparam.src);
    const float* filter = static_cast<const float*>(kparam.filter);
    float* dst = static_cast<float*>(kparam.dst);
    float* workspace = static_cast<float*>(kparam.workspace);
    const int IC = kparam.ic, IH = kparam.ih, IW = kparam.iw, OH = kparam.oh,
              OW = kparam.ow, FH = kparam.fh, FW = kparam.fw;
    const uint32_t PH = kparam.ph, PW = kparam.pw, SH = kparam.sh,
                   SW = kparam.sw;
    const ptrdiff_t INP_BS = kparam.inp_bs, OUT_BS = kparam.out_bs;

    float *dst2 = workspace;
    const int width = MEGDNN_SIMD_WIDTH;
    // dst2 is (H, W, N, C)
    memset(dst2, 0, sizeof(float) * OH*OW*N*OC);
    float *dst2_hwnc = dst2;
    rep(oh, OH) rep(ow, OW) {
        const float *src_bak = src;
        rep(ic, IC) {
            rep(fh, FH) for (int fw = 0; fw < FW; ++fw, filter += OC) {
                int ih = -PH + oh*SH + (FH-fh-1);
                int iw = -PW + ow*SW + (FW-fw-1);
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
                float *dst2_bak = dst2;
                rep(n, N) {
                    float s = src[n*INP_BS + ih*IW + iw];
                    const float *filter_bak = filter;
                    MEGDNN_SIMD_TYPE vs = MEGDNN_SIMD_SET1(s);
                    int oc = 0;
                    for (; oc+4*width <= OC; oc += 4*width, filter += 4*width) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vf2 = MEGDNN_SIMD_LOADU(filter + 2*width);
                        MEGDNN_SIMD_TYPE vf3 = MEGDNN_SIMD_LOADU(filter + 3*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        MEGDNN_SIMD_TYPE vd2 = MEGDNN_SIMD_LOADU(dst2 + oc + 2*width);
                        MEGDNN_SIMD_TYPE vd3 = MEGDNN_SIMD_LOADU(dst2 + oc + 3*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        vd2 = MEGDNN_SIMD_FMADD(vf2, vs, vd2);
                        vd3 = MEGDNN_SIMD_FMADD(vf3, vs, vd3);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 2*width, vd2);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 3*width, vd3);
                    }
                    if (oc+2*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        oc += 2*width;
                        filter += 2*width;
                    }
                    if (oc+1*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        oc += 1*width;
                        filter += 1*width;
                    }
                    for (; oc < OC; ++oc, ++filter) {
                        dst2[oc] += s * (*filter);
                    }
                    filter = filter_bak;
                    dst2 += OC;
                }
                dst2 = dst2_bak;
            }
            src += IH*IW;
        }
        src = src_bak;
        dst2 += N*OC;
    }
    transpose_knc2nsck(dst2_hwnc, dst, OH * OW, N, OC, OUT_BS);
}

void local_conv_generic(const LocalKParam &kparam) MEGDNN_SIMD_ATTRIBUTE_TARGET;
void local_conv_generic(const LocalKParam &kparam) {
    UNPACK_LOCAL_FLOAT_NONCONTIG_BATCH_KERN_PARAM(kparam, float);

    float *dst2 = workspace;
    const int width = MEGDNN_SIMD_WIDTH;
    // dst2 is (H, W, N, C)
    memset(dst2, 0, sizeof(float) * OH*OW*N*OC);
    float *dst2_hwnc = dst2;
    rep(oh, OH) rep(ow, OW) {
        const float *src_bak = src;
        rep(ic, IC) {
            rep(fh, FH) for (int fw = 0; fw < FW; ++fw, filter += OC) {
                int ih = -PH + oh*SH + (FH-fh-1);
                int iw = -PW + ow*SW + (FW-fw-1);
                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
                float *dst2_bak = dst2;
                rep(n, N) {
                    float s = src[n*INP_BS + ih*IW + iw];
                    const float *filter_bak = filter;
                    MEGDNN_SIMD_TYPE vs = MEGDNN_SIMD_SET1(s);
                    int oc = 0;
                    for (; oc+4*width <= OC; oc += 4*width, filter += 4*width) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vf2 = MEGDNN_SIMD_LOADU(filter + 2*width);
                        MEGDNN_SIMD_TYPE vf3 = MEGDNN_SIMD_LOADU(filter + 3*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        MEGDNN_SIMD_TYPE vd2 = MEGDNN_SIMD_LOADU(dst2 + oc + 2*width);
                        MEGDNN_SIMD_TYPE vd3 = MEGDNN_SIMD_LOADU(dst2 + oc + 3*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        vd2 = MEGDNN_SIMD_FMADD(vf2, vs, vd2);
                        vd3 = MEGDNN_SIMD_FMADD(vf3, vs, vd3);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 2*width, vd2);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 3*width, vd3);
                    }
                    if (oc+2*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vf1 = MEGDNN_SIMD_LOADU(filter + 1*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        MEGDNN_SIMD_TYPE vd1 = MEGDNN_SIMD_LOADU(dst2 + oc + 1*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        vd1 = MEGDNN_SIMD_FMADD(vf1, vs, vd1);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 1*width, vd1);
                        oc += 2*width;
                        filter += 2*width;
                    }
                    if (oc+1*width <= OC) {
                        MEGDNN_SIMD_TYPE vf0 = MEGDNN_SIMD_LOADU(filter + 0*width);
                        MEGDNN_SIMD_TYPE vd0 = MEGDNN_SIMD_LOADU(dst2 + oc + 0*width);
                        vd0 = MEGDNN_SIMD_FMADD(vf0, vs, vd0);
                        MEGDNN_SIMD_STOREU(dst2 + oc + 0*width, vd0);
                        oc += 1*width;
                        filter += 1*width;
                    }
                    for (; oc < OC; ++oc, ++filter) {
                        dst2[oc] += s * (*filter);
                    }
                    filter = filter_bak;
                    dst2 += OC;
                }
                dst2 = dst2_bak;
            }
            src += IH*IW;
        }
        src = src_bak;
        dst2 += N*OC;
    }
    transpose_knc2nsck(dst2_hwnc, dst, OH * OW, N, OC, OUT_BS);
}

} // anonymous namespace

namespace megdnn {

#define FUNC_NAME CONCAT_STR(local_xcorr_, MEGDNN_SIMD_NAME)

void FUNC_NAME(const LocalKParam &kparam) {
    auto N = kparam.n, OC = kparam.oc;
#define DISPATCH_WITH_N_OC(N, OC) do { \
    local_xcorr_tpl<N, OC>(kparam); \
    return; \
} while (0)

#define DISPATCH_WITH_N(N) \
    switch (OC) { \
        case 16: DISPATCH_WITH_N_OC(N, 16); break; \
        case 32: DISPATCH_WITH_N_OC(N, 32); break; \
        case 48: DISPATCH_WITH_N_OC(N, 48); break; \
        case 64: DISPATCH_WITH_N_OC(N, 64); break; \
    }
#define DISPATCH() \
    switch (N) { \
        case 1: DISPATCH_WITH_N(1); break; \
        case 2: DISPATCH_WITH_N(2); break; \
    }

    DISPATCH();

#undef DISPATCH
#undef DISPATCH_WITH_N
#undef DISPATCH_WITH_N_OC
    local_xcorr_generic(kparam);
}

#undef FUNC_NAME



#define FUNC_NAME CONCAT_STR(local_conv_, MEGDNN_SIMD_NAME)

void FUNC_NAME(const LocalKParam &kparam) {
    auto N = kparam.n, OC = kparam.oc;
#define DISPATCH_WITH_N_OC(N, OC) do { \
    local_conv_tpl<N, OC>(kparam); \
    return; \
} while (0)

#define DISPATCH_WITH_N(N) \
    switch (OC) { \
        case 16: DISPATCH_WITH_N_OC(N, 16); break; \
        case 32: DISPATCH_WITH_N_OC(N, 32); break; \
        case 48: DISPATCH_WITH_N_OC(N, 48); break; \
        case 64: DISPATCH_WITH_N_OC(N, 64); break; \
    }
#define DISPATCH() \
    switch (N) { \
        case 1: DISPATCH_WITH_N(1); break; \
        case 2: DISPATCH_WITH_N(2); break; \
    }

    DISPATCH();

#undef DISPATCH
#undef DISPATCH_WITH_N
#undef DISPATCH_WITH_N_OC
    local_conv_generic(kparam);
}

#undef FUNC_NAME

} // namespace megdnn

#include "src/common/macro_helper_epilogue.h"
