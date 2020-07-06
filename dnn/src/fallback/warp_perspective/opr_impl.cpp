/**
 * \file dnn/src/fallback/warp_perspective/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/warp_perspective/opr_impl.h"

#include "src/naive/warp_perspective/warp_perspective_cv.h"

#include "src/common/utils.h"
#include "src/common/cv/helper.h"
#include "src/naive/handle.h"
#include "src/common/warp_common.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_warpperspective)

namespace {

using namespace megdnn;

WorkspaceBundle get_bundle(size_t OH, size_t OW)
{
    WorkspaceBundle bundle(nullptr, {
            // tabsh0
            sizeof(int) * OH,
            // tabsh1
            sizeof(int) * OH,
            // tabsw0
            sizeof(int) * OW,
            // tabsw1
            sizeof(int) * OW,
            // tabrh
            sizeof(float) * OH,
            // tabrw
            sizeof(float) * OW,
            // cache0
            sizeof(float) * OW,
            // cache1
            sizeof(float) * OW
            });
    return bundle;
}

}  // anonymous namespace

namespace megdnn {
namespace fallback {

size_t WarpPerspectiveImpl::get_workspace_in_bytes(const TensorLayout &,
        const TensorLayout &,
        const TensorLayout &,
        const TensorLayout &dst)
{
    if (param().format == param::WarpPerspective::Format::NCHW) {
        size_t OH = dst.shape[2], OW = dst.shape[3];
        return get_bundle(OH, OW).total_size_in_bytes();
    } else {
        return 0;
    }
}

void WarpPerspectiveImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
                               _megdnn_tensor_in mat_idx, _megdnn_tensor_in dst,
                               _megdnn_workspace workspace) {
    check_exec_allow_nhwc_mat_idx(src.layout, mat.layout, mat_idx.layout,
                                  dst.layout, workspace.size);
    size_t nr_threads = static_cast<naive::HandleImpl*>(handle())
                                ->megcore_dispatcher()
                                ->nr_threads();
    //! When single thread, it will optimize when resize is usable
    //! When multi threads, it can't use the resize optimizaion, because
    //! not all N can use resize optimizaion, so it can't use the same
    //! logic in parallel in all N, so it will go to naive
    if (param().format == Format::NCHW && nr_threads == 1_z) {
#define cb(dt, ct, mct)                                                      \
    case DTypeTrait<dt>::enumv: {                                            \
        auto kparam = KernParam<ct, mct>::from_tensors(                      \
                param().format, param().bmode, param().border_val, src, mat, \
                mat_idx, dst, workspace);                                    \
        MIDOUT_BEGIN(megdnn_fallback_warpperspective, midout_iv(0), dt, ct,  \
                     mct) {                                                  \
            MEGDNN_DISPATCH_CPU_KERN_OPR(kern_fallback(kparam));             \
            return;                                                          \
        }                                                                    \
        MIDOUT_END();                                                        \
    }

        switch (src.layout.dtype.enumv()) {
            cb(dtype::Float32, float, float);
            MEGDNN_INC_FLOAT16(cb(dtype::Float16, dt_float16, float));
            cb(dtype::Int8, int8_t, float);
            cb(dtype::QuantizedS8, int8_t, float);
            cb(dtype::Uint8, uint8_t, float);
            cb(dtype::Quantized8Asymm, uint8_t, float);
            default:
                megdnn_throw(ssprintf("Unsupported input DType in "
                                      "WarpPerspective: %s",
                                      src.layout.dtype.name())
                                     .c_str());
        }
#undef cb
    }
    naive::WarpPerspectiveForwardImpl::exec(
            src, mat, mat_idx, dst, workspace);
}

template <typename ctype, typename mtype>
void WarpPerspectiveImpl::kern_fallback(
        const KernParam<ctype, mtype>& kern_param) {
    UNPACK_WARP_PERSPECTIVE_FWD_KERN_PARAM(kern_param);

    // cause error if accidentally used
    sptr = nullptr;
    mptr = nullptr;
    dptr = nullptr;
    MEGDNN_MARK_USED_VAR(sptr);
    MEGDNN_MARK_USED_VAR(mptr);
    MEGDNN_MARK_USED_VAR(dptr);
    MEGDNN_MARK_USED_VAR(border_val);

    KernParam<ctype, mtype> sub_param = kern_param;
    sub_param.n_src = 1;
    sub_param.n_mat = 1;
    sub_param.midx_ptr = nullptr;
    rep(n, N_MAT) {
        if (midx_ptr) {
            size_t idx = midx_ptr[n];
            megdnn_assert(idx < N_SRC,
                    "mat_idx out of bound: mat_idx[%zu]=%zu src_batch=%zu",
                    n, idx, N_SRC);
            sub_param.sptr = kern_param.sptr + idx * (C * IH * IW);
        } else if (n) {
            sub_param.sptr += C * IH * IW;
        }
        if (is_resize_optimizable(sub_param.mptr)) {
            if (bmode == BorderMode::CONSTANT) {
                MIDOUT_BEGIN(megdnn_fallback_warpperspective, midout_iv(1),
                             midout_iv(true), ctype, mtype) {
                    kern_resize<true, ctype, mtype>(sub_param);
                }
                MIDOUT_END();
            } else {
                MIDOUT_BEGIN(megdnn_fallback_warpperspective, midout_iv(1),
                             midout_iv(false), ctype, mtype) {
                    kern_resize<false, ctype, mtype>(sub_param);
                }
                MIDOUT_END();
            }
        } else {
            MIDOUT_BEGIN(megdnn_fallback_warpperspective, midout_iv(2), ctype,
                         mtype) {
                rep(oh, OH) kern_naive<ctype, mtype>(sub_param, oh);
            }
            MIDOUT_END();
        }
        sub_param.mptr += 3 * 3;
        sub_param.dptr += C*OH*OW;
    }

}

template <typename mtype>
bool WarpPerspectiveImpl::is_resize_optimizable(mtype *mat) {
    if (mat[1] != 0) return false;
    if (mat[3] != 0) return false;
    if (mat[6] != 0) return false;
    if (mat[7] != 0) return false;
    return true;
}

template <bool is_border_constant, typename ctype, typename mtype>
void WarpPerspectiveImpl::kern_resize(const KernParam<ctype, mtype> &kern_param) {
    UNPACK_WARP_PERSPECTIVE_FWD_KERN_PARAM(kern_param);
    MEGDNN_MARK_USED_VAR(N_SRC);
    MEGDNN_MARK_USED_VAR(N_MAT);
    MEGDNN_MARK_USED_VAR(midx_ptr);
    MEGDNN_MARK_USED_VAR(bmode);

    rounding::RoundingConverter<ctype> output_converter;
    auto bundle = get_bundle(OH, OW);
    bundle.set(kern_param.workspace.raw_ptr);
    int *tabsh0 = static_cast<int *>(bundle.get(0));
    int *tabsh1 = static_cast<int *>(bundle.get(1));
    int *tabsw0 = static_cast<int *>(bundle.get(2));
    int *tabsw1 = static_cast<int *>(bundle.get(3));
    float *tabrh = static_cast<float *>(bundle.get(4));
    float *tabrw = static_cast<float *>(bundle.get(5));
    float *cache0 = static_cast<float *>(bundle.get(6));
    float *cache1 = static_cast<float *>(bundle.get(7));

    float bval = border_val; // filled in UNPACK_WARP_PERSPECTIVE_FWD_KERN_PARAM

    auto src = sptr;
    auto mat = mptr;
    auto dst = dptr;

    //       | k_x  0  c_1 |
    // mat = |  0  k_y c_2 |
    //       |  0   0  c_3 |
    float kh = static_cast<float>(mat[4]) / mat[8]; // k_y / c_3
    float bh = static_cast<float>(mat[5]) / mat[8]; // c_2 / c_3
    float kw = static_cast<float>(mat[0]) / mat[8]; // k_x / c_3
    float bw = static_cast<float>(mat[2]) / mat[8]; // c_1 / c_3
    // build tab
    for (size_t h = 0; h < OH; ++h) {
        float f = static_cast<float>(h)*kh + bh;
        tabsh0[h] = get_real_coord(std::floor(f)+0, IH);
        tabsh1[h] = get_real_coord(std::floor(f)+1, IH);
        tabrh[h] = f - std::floor(f);
    }
    for (size_t w = 0; w < OW; ++w) {
        float f = static_cast<float>(w)*kw + bw;
        tabsw0[w] = get_real_coord(std::floor(f)+0, IW);
        tabsw1[w] = get_real_coord(std::floor(f)+1, IW);
        tabrw[w] = f - std::floor(f);
    }
    // (1, 2) -> (0, 1)
    auto calc_cache_backward = [&](size_t oh) {
        std::swap(cache0, cache1);
        // rebuild cache0
        size_t ih0 = tabsh0[oh];
        const ctype *psrc0 = src + ih0*IW;
        if (is_border_constant && ih0 >= IH) {
            for (size_t ow = 0; ow < OW; ++ow) cache0[ow] = bval;
        } else {
            for (size_t ow = 0; ow < OW; ++ow) {
                size_t iw0 = tabsw0[ow], iw1 = tabsw1[ow];
                float v0 = (is_border_constant && iw0 >= IW) ? bval : psrc0[iw0];
                float v1 = (is_border_constant && iw1 >= IW) ? bval : psrc0[iw1];
                cache0[ow] = v0*(1.0f - tabrw[ow]) + v1*tabrw[ow];
            }
        }
    };
    // (0, 1) -> (1, 2)
    auto calc_cache_forward = [&](size_t oh) {
        std::swap(cache0, cache1);
        // rebuild cache1
        size_t ih1 = tabsh1[oh];
        const ctype *psrc1 = src + ih1*IW;
        if (is_border_constant && ih1 >= IH) {
            for (size_t ow = 0; ow < OW; ++ow) cache1[ow] = bval;
        } else {
            for (size_t ow = 0; ow < OW; ++ow) {
                size_t iw0 = tabsw0[ow], iw1 = tabsw1[ow];
                float v0 = (is_border_constant && iw0 >= IW) ? bval : psrc1[iw0];
                float v1 = (is_border_constant && iw1 >= IW) ? bval : psrc1[iw1];
                cache1[ow] = v0*(1.0f - tabrw[ow]) + v1*tabrw[ow];
            }
        }
    };
    auto calc_cache_all = [&](size_t oh) {
        // rebuild cache0
        size_t ih0 = tabsh0[oh];
        size_t ih1 = tabsh1[oh];
        const ctype *psrc0 = src + ih0*IW;
        if (is_border_constant && ih0 >= IH) {
            for (size_t ow = 0; ow < OW; ++ow) cache0[ow] = bval;
        } else {
            for (size_t ow = 0; ow < OW; ++ow) {
                size_t iw0 = tabsw0[ow], iw1 = tabsw1[ow];
                float v0 = (is_border_constant && iw0 >= IW) ? bval : psrc0[iw0];
                float v1 = (is_border_constant && iw1 >= IW) ? bval : psrc0[iw1];
                cache0[ow] = v0*(1.0f - tabrw[ow]) + v1*tabrw[ow];
            }
        }
        // rebuild cache1
        const ctype *psrc1 = src + ih1*IW;
        if (is_border_constant && ih1 >= IH) {
            for (size_t ow = 0; ow < OW; ++ow) cache1[ow] = bval;
        } else {
            for (size_t ow = 0; ow < OW; ++ow) {
                size_t iw0 = tabsw0[ow], iw1 = tabsw1[ow];
                float v0 = (is_border_constant && iw0 >= IW) ? bval : psrc1[iw0];
                float v1 = (is_border_constant && iw1 >= IW) ? bval : psrc1[iw1];
                cache1[ow] = v0*(1.0f - tabrw[ow]) + v1*tabrw[ow];
            }
        }
    };
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < OH; ++h) {
            enum class CacheType {
                NONE,
                FORWARD,
                BACKWARD,
                ALL
            } cache_type;
            if (h == 0) {
                cache_type = CacheType::ALL;
            } else if (tabsh0[h] != -1 &&
                    tabsh0[h] == tabsh0[h-1] &&
                    tabsh1[h] != -1 &&
                    tabsh1[h] == tabsh1[h-1]) {
                cache_type = CacheType::NONE;
            } else if (tabsh0[h] != -1 && tabsh0[h] == tabsh1[h-1]) {
                cache_type = CacheType::FORWARD;
            } else if (tabsh1[h] != -1 && tabsh1[h] == tabsh0[h-1]) {
                cache_type = CacheType::BACKWARD;
            } else {
                cache_type = CacheType::ALL;
            }
            if (cache_type == CacheType::ALL) {
                calc_cache_all(h);
            } else if (cache_type == CacheType::FORWARD) {
                calc_cache_forward(h);
            } else if (cache_type == CacheType::BACKWARD) {
                calc_cache_backward(h);
            }
            ctype *pdst = dst + h*OW;
            for (size_t w = 0; w < OW; ++w) {
                float result = cache0[w]*(1.0f - tabrh[h]) +
                        cache1[w]*tabrh[h];
                if (is_border_constant) {
                    // nan check
                    result = std::isfinite(result) ? result : bval;
                }
                pdst[w] = output_converter(result);
            }
        }
        src += IH*IW;
        dst += OH*OW;
    }
}

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen
