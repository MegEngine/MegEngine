#include "src/naive/images2neibs/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

namespace megdnn {
namespace naive {

template <typename T>
void Images2NeibsForwardImpl::exec_internal(
        _megdnn_tensor_in src, _megdnn_tensor_out dst) {
    megdnn_assert(src.layout.ndim == 5 || src.layout.ndim == 4);
    if (src.layout.ndim == 5) {
        int N = src.layout.shape[0], C = src.layout.shape[2], IH = src.layout.shape[1],
            IW = src.layout.shape[3];
        auto sptr = src.ptr<T>();
        auto dptr = dst.ptr<T>();
        size_t idx = 0;
        int window_h = static_cast<int>(param().window_h);
        int window_w = static_cast<int>(param().window_w);
        int pad_h = static_cast<int>(param().pad_h);
        int pad_w = static_cast<int>(param().pad_w);
        int stride_h = static_cast<int>(param().stride_h);
        int stride_w = static_cast<int>(param().stride_w);
        int dilate_h = static_cast<int>(param().dilate_h);
        int dilate_w = static_cast<int>(param().dilate_w);
        int equ_window_h = dilate_h * (window_h - 1) + 1;
        int equ_window_w = dilate_w * (window_w - 1) + 1;

        auto src_stride = src.layout.stride;
        auto dst_stride = dst.layout.stride;

        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c) {
                int ih = -pad_h;
                int hc = 0;
                for (; ih <= IH + pad_h - equ_window_h; ih += stride_h, hc++) {
                    int iw = -pad_w;
                    int wc = 0;
                    for (; iw <= IW + pad_w - equ_window_w; iw += stride_w, wc++) {
                        for (int kh = 0; kh < window_h; ++kh)
                            for (int kw = 0; kw < window_w; ++kw) {
                                for (int cn = 0; cn < 4; cn++) {
                                    int ih2 = ih + dilate_h * kh,
                                        iw2 = iw + dilate_w * kw;
                                    int dst_pos =
                                            n * dst_stride[0] + hc * dst_stride[1] +
                                            c * dst_stride[2] + wc * dst_stride[3] +
                                            kh * dst_stride[4] + kw * dst_stride[5] +
                                            cn * dst_stride[6];
                                    int src_pos =
                                            n * src_stride[0] + ih2 * src_stride[1] +
                                            c * src_stride[2] + iw2 * src_stride[3] +
                                            cn * src_stride[4];
                                    if (ih2 >= 0 && ih2 < IH && iw2 >= 0 && iw2 < IW) {
                                        dptr[dst_pos] = sptr[src_pos];
                                    } else {
                                        dptr[dst_pos] = 0.0f;
                                    }
                                }
                            }
                        ++idx;
                    }
                }
            }
    } else {
        int N = src.layout.shape[0], C = src.layout.shape[1], IH = src.layout.shape[2],
            IW = src.layout.shape[3];
        auto sptr = src.ptr<T>();
        auto dptr = dst.ptr<T>();
        size_t idx = 0;
        int window_h = static_cast<int>(param().window_h);
        int window_w = static_cast<int>(param().window_w);
        int pad_h = static_cast<int>(param().pad_h);
        int pad_w = static_cast<int>(param().pad_w);
        int stride_h = static_cast<int>(param().stride_h);
        int stride_w = static_cast<int>(param().stride_w);
        int dilate_h = static_cast<int>(param().dilate_h);
        int dilate_w = static_cast<int>(param().dilate_w);
        int equ_window_h = dilate_h * (window_h - 1) + 1;
        int equ_window_w = dilate_w * (window_w - 1) + 1;
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c) {
                int ih = -pad_h;
                for (; ih + equ_window_h <= IH + pad_h; ih += stride_h) {
                    int iw = -pad_w;
                    for (; iw + equ_window_w <= IW + pad_w; iw += stride_w) {
                        for (int kh = 0; kh < window_h; ++kh)
                            for (int kw = 0; kw < window_w; ++kw) {
                                int ih2 = ih + dilate_h * kh, iw2 = iw + dilate_w * kw;
                                int src_pos =
                                        n * C * IH * IW + c * IH * IW + ih2 * IW + iw2;
                                int dst_pos =
                                        idx * window_h * window_w + kh * window_w + kw;
                                if (ih2 >= 0 && ih2 < IH && iw2 >= 0 && iw2 < IW) {
                                    dptr[dst_pos] = sptr[src_pos];
                                } else {
                                    dptr[dst_pos] = 0.0f;
                                }
                            }
                        ++idx;
                    }
                }
            }
    }
}

void Images2NeibsForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType)                                                             \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {               \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                         \
                exec_internal<typename DTypeTrait<DType>::ctype>(src, dst);); \
        return;                                                               \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
#else
    __builtin_trap();
#endif
}

template <typename T>
void Images2NeibsBackwardImpl::exec_internal(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad) {
    int N = grad.layout.shape[0], C = grad.layout.shape[1], IH = grad.layout.shape[2],
        IW = grad.layout.shape[3];
    auto sptr = grad.ptr<T>();
    auto dptr = diff.ptr<T>();
    size_t idx = 0;
    int window_h = static_cast<int>(param().window_h);
    int window_w = static_cast<int>(param().window_w);
    int pad_h = static_cast<int>(param().pad_h);
    int pad_w = static_cast<int>(param().pad_w);
    int stride_h = static_cast<int>(param().stride_h);
    int stride_w = static_cast<int>(param().stride_w);
    int dilate_h = static_cast<int>(param().dilate_h);
    int dilate_w = static_cast<int>(param().dilate_w);
    int equ_window_h = dilate_h * (window_h - 1) + 1;
    int equ_window_w = dilate_w * (window_w - 1) + 1;
    memset(sptr, 0, sizeof(T) * N * C * IH * IW);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            int ih = -pad_h;
            for (; ih + equ_window_h <= IH + pad_h; ih += stride_h) {
                int iw = -pad_w;
                for (; iw + equ_window_w <= IW + pad_w; iw += stride_w) {
                    for (int kh = 0; kh < window_h; ++kh)
                        for (int kw = 0; kw < window_w; ++kw) {
                            int ih2 = ih + dilate_h * kh, iw2 = iw + dilate_w * kw;
                            if (ih2 >= 0 && ih2 < IH && iw2 >= 0 && iw2 < IW) {
                                sptr[n * C * IH * IW + c * IH * IW + ih2 * IW + iw2] +=
                                        dptr[idx * window_h * window_w + kh * window_w +
                                             kw];
                            }
                        }
                    ++idx;
                }
            }
        }
}

void Images2NeibsBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(diff.layout, grad.layout, workspace.size);
#define cb(DType)                                                               \
    if (diff.layout.dtype == DType()) {                                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                           \
                exec_internal<typename DTypeTrait<DType>::ctype>(diff, grad);); \
        return;                                                                 \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb);
#undef cb
    megdnn_assert_internal(0);
#else
    __builtin_trap();
#endif
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
