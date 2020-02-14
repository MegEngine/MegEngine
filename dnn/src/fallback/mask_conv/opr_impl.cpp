/**
 * \file dnn/src/fallback/mask_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/opr_delegate.h"
#include "src/fallback/mask_conv/opr_impl.h"

namespace {

using namespace megdnn;
using namespace fallback;

template <bool is_xcorr>
void img2col_mask(const float* src, float* dst, const size_t OC,
                  const size_t OH, const size_t OW, const size_t IC,
                  const size_t IH, const size_t IW, const size_t FH,
                  const size_t FW, const size_t SH, const size_t SW,
                  const size_t DH, const size_t DW, const unsigned int* maskInd,
                  const size_t maskN) {
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(OW);
    size_t i = 0;
    rep(ic, IC) {
        rep(fh, FH) {
            rep(fw, FW) {
                rep(ind, maskN) {
                    size_t oh = maskInd[ind] >> 16;
                    size_t ow = maskInd[ind] & 0xFFFF;
                    size_t fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }
                    dst[i++] = src[ic * IH * IW + (oh * SH + fh2 * DH) * IW +
                                   (ow * SW + fw2 * DW)];
                }
            }
        }
    }
}

template <typename ctype>
void get_mask_index(const ctype* mask, const size_t OH, const size_t OW,
                    unsigned int* maskInd, size_t* maskN) {
    size_t length = 0;
    rep(oh, OH) rep(ow, OW) {
        size_t idx = oh * OW + ow;
        if (mask[idx]) {
            maskInd[length++] = oh << 16 | ow;
        }
    }
    *maskN = length;
}

void index_to_dst(const float* result, float* dst, const size_t OC,
                  const size_t OH, const size_t OW, unsigned int* maskInd,
                  const size_t maskN) {
    const float* addr = result;
    rep(oc, OC) rep(ind, maskN) {
        size_t oh = maskInd[ind] >> 16;
        size_t ow = maskInd[ind] & 0xFFFF;
        dst[oc * OH * OW + oh * OW + ow] = *addr++;
    }
}

template <typename ctype>
void exec_internel(const float* src, const float* filter, const ctype* mask,
                   float* dst, WorkspaceBundle wbundle, MatrixMul* opr,
                   size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                   size_t OH, size_t OW, size_t PH, size_t PW, size_t SH,
                   size_t SW, size_t FH, size_t FW, size_t DH, size_t DW,
                   bool is_xcorr) {
    memset(dst, 0, sizeof(float) * N * OC * OH * OW);
    unsigned int* maskInd = static_cast<unsigned int*>(wbundle.get(0));
    size_t maskN;
    get_mask_index(mask, OH, OW, maskInd, &maskN);

    void* matmul_workspace_ptr = wbundle.get(3);
    size_t matmul_wsize = wbundle.get_size(3);

    size_t IH2 = IH + 2 * PH;
    size_t IW2 = IW + 2 * PW;

    rep(n, N) {
        const float* src_t = src + n * IC * IW * IH;
        if (PH > 0 || PW > 0) {
            float* src_pad = static_cast<float*>(wbundle.get(1));
            src_t = src_pad;
            rep(ic, IC) {
                if (PH) {
                    memset(src_pad, 0, IW2 * PH * sizeof(float));
                    src_pad += IW2 * PH;
                }
                rep(ih, IH) {
                    rep(i, PW) { *src_pad++ = 0; }
                    memcpy(src_pad, src + (n * IC + ic) * IH * IW + ih * IW,
                           IW * sizeof(float));
                    src_pad += IW;
                    rep(i, PW) { *src_pad++ = 0; }
                }
                if (PH) {
                    memset(src_pad, 0, IW2 * PH * sizeof(float));
                    src_pad += IW2 * PH;
                }
            }
        }
        float* B_mat = static_cast<float*>(wbundle.get(2));
        if (is_xcorr) {
            img2col_mask<true>(src_t, B_mat, OC, OH, OW, IC, IH2, IW2, FH, FW,
                               SH, SW, DH, DW, maskInd, maskN);
        } else {
            img2col_mask<false>(src_t, B_mat, OC, OH, OW, IC, IH2, IW2, FH, FW,
                                SH, SW, DH, DW, maskInd, maskN);
        }
        float* result = static_cast<float*>(wbundle.get(1));
        TensorND A((float*)filter,
                   TensorLayout({OC, IC * FH * FW}, dtype::Float32())),
                B((float*)B_mat,
                  TensorLayout({IC * FH * FW, maskN}, dtype::Float32())),
                C((float*)result, TensorLayout({OC, maskN}, dtype::Float32()));

        Workspace workspace(static_cast<megdnn::dt_byte*>(matmul_workspace_ptr),
                            matmul_wsize);

        opr->exec(A, B, C, workspace);

        index_to_dst(result, dst + n * OC * OH * OW, OC, OH, OW, maskInd,
                     maskN);
    }
}

}  // namespace

namespace megdnn {
namespace fallback {

MaskConvForwardImpl::MaskConvForwardImpl(Handle* handle)
        : MaskConvForward(handle) {
    m_matmul_opr = inplace_cpu_handle()->create_operator<MatrixMul>();
}

WorkspaceBundle MaskConvForwardImpl::get_wbundle(
        const size_t OC, const size_t OH, const size_t OW, const size_t IC,
        const size_t IH, const size_t IW, const size_t FH, const size_t FW,
        const size_t PH, const size_t PW) {
    size_t maskInd = OH * OW * sizeof(int);
    size_t src_pad = IC * (IH + PH * 2) * (IW + PW * 2) * sizeof(float);
    size_t matmul_dst = OC * OH * OW * sizeof(float);
    size_t tmp = std::max<size_t>(src_pad, matmul_dst);
    size_t img2col = IC * FH * FW * OH * OW * sizeof(float);
    size_t matmul_cal;
    {
        TensorLayout A({OC, IC * FH * FW}, dtype::Float32());
        TensorLayout B({IC * FH * FW, OH * OW}, dtype::Float32());
        TensorLayout C({OC, OH * OW}, dtype::Float32());
        matmul_cal = m_matmul_opr->get_workspace_in_bytes(A, B, C);
    }
    return WorkspaceBundle{nullptr, {maskInd, tmp, img2col, matmul_cal}};
}

void MaskConvForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                               _megdnn_tensor_in mask, _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, mask.layout, dst.layout,
               workspace.size);
    size_t N = src.layout[0], OC = filter.layout[0], IC = filter.layout[1],
           IH = src.layout[2], IW = src.layout[3], OH = dst.layout[2],
           OW = dst.layout[3], PH = param().pad_h, PW = param().pad_w,
           SH = param().stride_h, SW = param().stride_w, FH = filter.layout[2],
           FW = filter.layout[3], DH = param().dilate_h, DW = param().dilate_w;
    bool is_xcorr = param().mode != Mode::CONVOLUTION;
    auto wbundle = get_wbundle(OC, OH, OW, IC, IH, IW, FH, FW, PH, PW);
    wbundle.set(workspace.ptr<void>());
    if (filter.layout.dtype == dtype::Float32()) {
#define cb(DType)                                                            \
    if (mask.layout.dtype == DType()) {                                      \
        using ctype = typename DTypeTrait<DType>::ctype;                     \
        MEGDNN_DISPATCH_CPU_KERN(                                            \
                static_cast<HandleImpl*>(handle()),                          \
                exec_internel<ctype>(src.ptr<float>(), filter.ptr<float>(),  \
                                     mask.ptr<ctype>(), dst.ptr<float>(),    \
                                     wbundle, m_matmul_opr.get(), N, IC, OC, \
                                     IH, IW, OH, OW, PH, PW, SH, SW, FH, FW, \
                                     DH, DW, is_xcorr););                    \
        return;                                                              \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
    }
    megdnn_assert(0);
}

size_t MaskConvForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                   const TensorLayout& filter,
                                                   const TensorLayout& mask,
                                                   const TensorLayout& dst) {
    megdnn_ignore(mask);
    size_t OC = filter[0], IC = filter[1], IH = src[2], IW = src[3],
           OH = dst[2], OW = dst[3], FH = filter[2], FW = filter[3],
           PH = param().pad_h, PW = param().pad_w;
    auto wbundle = get_wbundle(OC, OH, OW, IC, IH, IW, FH, FW, PH, PW);
    return wbundle.total_size_in_bytes();
}

}  // namespace fallback
}  // namespace megdnn
