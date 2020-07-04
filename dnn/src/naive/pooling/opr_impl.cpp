/**
 * \file dnn/src/naive/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/pooling/opr_impl.h"

#include <cstring>
#include "megdnn/dtype.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_pooling)

namespace {

using namespace megdnn;

template <typename ctype_>
struct MaxPooler {
    using ctype = ctype_;
    ctype answer;
    bool fed;
    MaxPooler(size_t, DType) : answer(DTypeTrait<ctype>::min()) {}
    void init() {
        answer = DTypeTrait<ctype>::min();
        fed = false;
    }
    void feed(ctype x) {
        answer = answer > x ? answer : x;
        fed = true;
    }
    ctype get_ans() {
        if (!fed) {
            megdnn_throw("The pooling window lies outside completely");
        }
        return answer;
    }
};

template <typename stype_, typename ctype_>
struct MeanIncludePoolerBase {
    using stype = stype_;
    using ctype = ctype_;
    ctype sum;
    const ctype count;
    MeanIncludePoolerBase(size_t count, DType) : count(ctype(count)) {}
    void init() { sum = ctype(0); }
    void feed(stype x) { sum += x; }
};

template <typename T>
struct MeanIncludePooler : public MeanIncludePoolerBase<T, T> {
    using MeanIncludePoolerBase<T, T>::MeanIncludePoolerBase;
    using ctype = typename MeanIncludePoolerBase<T, T>::ctype;
    ctype get_ans() { return this->sum / this->count; }
};

template <>
struct MeanIncludePooler<int8_t>
        : public MeanIncludePoolerBase<int8_t, int32_t> {
    using MeanIncludePoolerBase::MeanIncludePoolerBase;
    ctype get_ans() {
        return std::min<int32_t>(
                std::max<int32_t>(std::numeric_limits<int8_t>::min(),
                                  sum / count),
                std::numeric_limits<int8_t>::max());
    }
};

template <>
struct MeanIncludePooler<dt_quint8> {
    int32_t sum;
    size_t feed_count;
    const int32_t count;
    const int32_t zero_point;

    MeanIncludePooler(size_t count, DType dtype)
            : count(int32_t(count)),
              zero_point(dtype.param<dtype::Quantized8Asymm>().zero_point) {}

    void init() {
        sum = 0;
        feed_count = 0;
    }

    void feed(dt_quint8 x) {
        sum += x.as_uint8();
        ++feed_count;
    }

    dt_quint8 get_ans() {
        int32_t summie = sum + (count - feed_count) * zero_point;
        int32_t rounded = std::round(static_cast<float>(summie) / count);
        return dt_quint8(std::min<int32_t>(
                std::max<int32_t>(rounded, std::numeric_limits<uint8_t>::min()),
                std::numeric_limits<uint8_t>::max()));
    }
};

/*!
 * \brief Average pooling operation within a single window.
 *        Works on integers. Rounds toward +INF.
 * \tparam T input data type
 * \tparam U convert input data type to U before accumulating
 * \tparam ICType data type for intermediate result
 */
template <typename T, typename U = T, typename ICType = int32_t>
struct MeanIncludeRoundedPooler {
    ICType sum;
    const int32_t count;

    MeanIncludeRoundedPooler(size_t count, DType) : count(ICType(count)) {}
    void init() { sum = 0; }
    void feed(T x) { sum += static_cast<ICType>(static_cast<U>(x)); }
    T get_ans() { return T(std::round(static_cast<float>(sum) / count)); }
};

template <>
struct MeanIncludePooler<dt_qint32>
        : MeanIncludeRoundedPooler<dt_qint32, int32_t> {
    using MeanIncludeRoundedPooler::MeanIncludeRoundedPooler;
};
template <>
struct MeanIncludePooler<dt_qint8>
        : MeanIncludeRoundedPooler<dt_qint8, int8_t> {
    using MeanIncludeRoundedPooler::MeanIncludeRoundedPooler;
};

struct NCHWIdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w,
                          size_t /* N */, size_t C, size_t H, size_t W) {
        return ((n * C + c) * H + h) * W + w;
    }
};

struct NHWCIdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w,
                          size_t /* N */, size_t C, size_t H, size_t W) {
        return ((n * H + h) * W + w) * C + c;
    }
};

struct NHWCD4IdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w,
                          size_t /* N */, size_t C, size_t H, size_t W) {
        return (((n * H + h) * (C >> 2) + (c >> 2)) * W + w) * 4 + (c & 0x3);
    }
};

struct NCHW4IdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w, size_t,
                          size_t C, size_t H, size_t W) {
        return (((n * (C >> 2) + (c >> 2)) * H + h) * W + w) * 4 + (c & 0b11);
    }
};
struct NCHW88IdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w, size_t,
                          size_t C, size_t H, size_t W) {
        size_t id =
                (((n * (C >> 3) + (c >> 3)) * H + h) * W + w) * 8 + (c & 0b111);
        return id;
    }
};
struct NCHW44IdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w, size_t,
                          size_t C, size_t H, size_t W) {
        size_t id = (((n * (C >> 2) + (c >> 2)) * H + h) * W + w) * 4 + (c % 4);
        return id;
    }
};

struct CHWN4IdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w, size_t N,
                          size_t, size_t H, size_t W) {
        return ((((c >> 2) * H + h) * W + w) * N + n) * 4 + (c & 0b11);
    }
};

struct NCHW32IdxGetter {
    static size_t get_idx(size_t n, size_t c, size_t h, size_t w, size_t,
                          size_t C, size_t H, size_t W) {
        return (((n * (C >> 5) + (c >> 5)) * H + h) * W + w) * 32 + (c & 0x1f);
    }
};
/*!
 * Pooler for AVERAGE_COUNT_EXCLUDE_PADDING mode
 */
template <typename ctype>
struct MeanExcludePooler {
    ctype sum;
    size_t count;
    MeanExcludePooler(size_t, DType) {}
    void init() {
        sum = 0.0f;
        count = 0u;
    }
    void feed(ctype x) {
        sum += x;
        ++count;
    }
    ctype get_ans() {
        if (count == 0u) {
            megdnn_throw("The pooling window lies outside completely");
        }
        return sum / static_cast<ctype>(count);
    }
};

/*!
 * \brief Average pooling operation within a single window.
 *        Works on integers. Rounds toward +INF.
 * \tparam T input data type
 * \tparam U convert input data type to U before accumulating
 * \tparam ICType data type for intermediate result
 */
template <typename T, typename U, typename ICType = U>
struct MeanExcludeRoundedPooler {
    ICType sum;
    size_t count;

    MeanExcludeRoundedPooler(size_t, DType) {}

    void init() {
        sum = 0;
        count = 0;
    }
    void feed(T x) {
        sum += U(x);
        ++count;
    }
    T get_ans() {
        if (count == 0u) {
            megdnn_throw("The pooling window lies outside completely");
        }
        return T(std::round(static_cast<float>(sum) / count));
    }
};

template <>
struct MeanExcludePooler<dt_quint8>
        : MeanExcludeRoundedPooler<dt_quint8, uint8_t, uint32_t> {
    using MeanExcludeRoundedPooler::MeanExcludeRoundedPooler;
};

template <>
struct MeanExcludePooler<dt_qint32>
        : MeanExcludeRoundedPooler<dt_qint32, int32_t> {
    using MeanExcludeRoundedPooler::MeanExcludeRoundedPooler;
};

template <>
struct MeanExcludePooler<dt_qint8>
        : MeanExcludeRoundedPooler<dt_qint8, int8_t, int32_t> {
    using MeanExcludeRoundedPooler::MeanExcludeRoundedPooler;
};

template <typename Pooler, typename IdxGetter,
          typename ctype = typename Pooler::ctype>
void pooling_forward_impl(const ctype* __restrict src, ctype* __restrict dst,
                          DType src_dtype, size_t N, size_t C, size_t IH,
                          size_t IW, size_t OH, size_t OW, size_t PH, size_t PW,
                          size_t SH, size_t SW, size_t FH, size_t FW) {
    rep(n, N) rep(c, C) rep(oh, OH) rep(ow, OW) {
        Pooler pooler(FH * FW, src_dtype);
        pooler.init();
        rep(fh, FH) rep(fw, FW) {
            size_t ih = -PH + oh * SH + fh;
            size_t iw = -PW + ow * SW + fw;
            if (ih < IH && iw < IW) {
                size_t idx = IdxGetter::get_idx(n, c, ih, iw, N, C, IH, IW);
                pooler.feed(src[idx]);
            }
        }
        size_t idx = IdxGetter::get_idx(n, c, oh, ow, N, C, OH, OW);
        dst[idx] = pooler.get_ans();
    }
}

template <typename ctype, typename IdxGetter>
void pooling_backward_avg_impl(const ctype* __restrict /* src */,
                               const ctype* __restrict /* dst */,
                               const ctype* __restrict diff,
                               ctype* __restrict grad, size_t N, size_t C,
                               size_t IH, size_t IW, size_t OH, size_t OW,
                               size_t PH, size_t PW, size_t SH, size_t SW,
                               size_t FH, size_t FW, bool is_include = true) {
    std::memset(grad, 0, sizeof(ctype) * (N * C * IH * IW));
    rep(n, N) rep(c, C) rep(oh, OH) rep(ow, OW) {
        size_t count = 0u;
        rep(fh, FH) rep(fw, FW) {
            size_t ih = -PH + oh * SH + fh;
            size_t iw = -PW + ow * SW + fw;
            if (ih < IH && iw < IW)
                ++count;
        }
        if (is_include)
            count = FH * FW;
        if (count == 0u) {
            megdnn_throw("The pooling window lies outside completely");
        }
        rep(fh, FH) rep(fw, FW) {
            size_t ih = -PH + oh * SH + fh;
            size_t iw = -PW + ow * SW + fw;
            if (ih < IH && iw < IW) {
                size_t gi = IdxGetter::get_idx(n, c, ih, iw, N, C, IH, IW);
                size_t di = IdxGetter::get_idx(n, c, oh, ow, N, C, OH, OW);
                auto& gval = grad[gi];
                auto dval = diff[di];
                gval += dval / ctype(count);
            }
        }
    }
}

template <typename ctype, typename IdxGetter>
void pooling_backward_avg_expd_impl(const ctype* __restrict src,
                                    const ctype* __restrict dst,
                                    const ctype* __restrict diff,
                                    ctype* __restrict grad, size_t N, size_t C,
                                    size_t IH, size_t IW, size_t OH, size_t OW,
                                    size_t PH, size_t PW, size_t SH, size_t SW,
                                    size_t FH, size_t FW) {
    pooling_backward_avg_impl<ctype, IdxGetter>(src, dst, diff, grad, N, C, IH,
                                                IW, OH, OW, PH, PW, SH, SW, FH,
                                                FW, false);
}

template <typename ctype, typename IdxGetter>
void pooling_backward_max_impl(const ctype* __restrict src,
                               const ctype* __restrict dst,
                               const ctype* __restrict diff,
                               ctype* __restrict grad, size_t N, size_t C,
                               size_t IH, size_t IW, size_t OH, size_t OW,
                               size_t PH, size_t PW, size_t SH, size_t SW,
                               size_t FH, size_t FW) {
    std::memset(grad, 0, sizeof(ctype) * (N * C * IH * IW));
    rep(n, N) rep(c, C) rep(oh, OH) rep(ow, OW) {
        size_t count = 0u;
        rep(fh, FH) rep(fw, FW) {
            size_t ih = -PH + oh * SH + fh;
            size_t iw = -PW + ow * SW + fw;
            if (ih < IH && iw < IW)
                ++count;
        }
        if (count == 0u) {
            megdnn_throw("The pooling window lies outside completely");
        }
        rep(fh, FH) rep(fw, FW) {
            size_t ih = -PH + oh * SH + fh;
            size_t iw = -PW + ow * SW + fw;
            if (ih < IH && iw < IW) {
                size_t si = IdxGetter::get_idx(n, c, ih, iw, N, C, IH, IW);
                size_t di = IdxGetter::get_idx(n, c, oh, ow, N, C, OH, OW);
                auto sval = src[si];
                auto& gval = grad[si];
                auto dst_val = dst[di];
                auto diff_val = diff[di];
                if (sval == dst_val)
                    gval += diff_val;
            }
        }
    }
}

}  // namespace

namespace megdnn {
namespace naive {

void PoolingForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                              _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t c_pos, spatial_pos, batch_pos = 0;
    if (param().format == Param::Format::NCHW ||
        param().format == Param::Format::NCHW4 ||
        param().format == Param::Format::NCHW88 ||
        param().format == Param::Format::NCHW44 ||
        param().format == Param::Format::NCHW32) {
        c_pos = 1;
        spatial_pos = 2;
    } else if (param().format == Param::Format::NHWC) {
        c_pos = 3;
        spatial_pos = 1;
    } else if (param().format == Param::Format::CHWN4) {
        c_pos = 0;
        spatial_pos = 1;
        batch_pos = 3;
    } else {
        megdnn_assert(param().format == Param::Format::NHWCD4);
        c_pos = 2;
        spatial_pos = 1;
    }
    size_t N = src.layout.shape[batch_pos], C = src.layout.shape[c_pos],
           IH = src.layout.shape[spatial_pos + 0],
           IW = src.layout.shape[spatial_pos + 1];
    size_t OH = dst.layout.shape[spatial_pos + 0],
           OW = dst.layout.shape[spatial_pos + 1];
    if (param().format == Param::Format::NHWCD4) {
        C *= 4;
        IW = src.layout.shape[spatial_pos + 2];
        OW = dst.layout.shape[spatial_pos + 2];
    }
    if (param().format == Param::Format::NCHW4 ||
        param().format == Param::Format::NCHW44 ||
        param().format == Param::Format::CHWN4) {
        C *= 4;
    }
    if (param().format == Param::Format::NCHW88) {
        C *= 8;
    }
    if (param().format == Param::Format::NCHW32) {
        C *= 32;
    }
    size_t PH = param().pad_h, PW = param().pad_w;
    size_t FH = param().window_h, FW = param().window_w;
    size_t SH = param().stride_h, SW = param().stride_w;
#define DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, IdxGetter)                 \
    MIDOUT_BEGIN(megdnn_naive_pooling, midout_iv(#Pooler #IdxGetter##_hash)) { \
        MEGDNN_DISPATCH_CPU_KERN(                                              \
                static_cast<naive::HandleImpl*>(handle()),                     \
                pooling_forward_impl<Pooler MEGDNN_COMMA IdxGetter>(           \
                        sptr, dptr, src.layout.dtype, N, C, IH, IW, OH, OW,    \
                        PH, PW, SH, SW, FH, FW));                              \
    }                                                                          \
    MIDOUT_END();

#define DISPATCH_WITH_POOLER(Pooler)                                      \
    switch (param().format) {                                             \
        case Param::Format::NCHW:                                         \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NCHWIdxGetter);   \
            break;                                                        \
        case Param::Format::NHWC:                                         \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NHWCIdxGetter);   \
            break;                                                        \
        case Param::Format::NHWCD4:                                       \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NHWCD4IdxGetter); \
            break;                                                        \
        case Param::Format::NCHW4:                                        \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NCHW4IdxGetter);  \
            break;                                                        \
        case Param::Format::NCHW88:                                       \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NCHW88IdxGetter); \
            break;                                                        \
        case Param::Format::NCHW44:                                       \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NCHW44IdxGetter); \
            break;                                                        \
        case Param::Format::NCHW32:                                       \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, NCHW32IdxGetter); \
            break;                                                        \
        case Param::Format::CHWN4:                                        \
            DISPATCH_WITH_POOLER_AND_IDX_GETTER(Pooler, CHWN4IdxGetter);  \
            break;                                                        \
        default:                                                          \
            megdnn_throw("invalid pooling format");                       \
    }

#define cb(DType)                                               \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype;        \
        switch (param().mode) {                                 \
            case Mode::MAX: {                                   \
                auto sptr = src.ptr<ctype>();                   \
                auto dptr = dst.ptr<ctype>();                   \
                DISPATCH_WITH_POOLER(MaxPooler<ctype>);         \
                return;                                         \
            }                                                   \
            case Mode::AVERAGE: {                               \
                auto sptr = src.ptr<ctype>();                   \
                auto dptr = dst.ptr<ctype>();                   \
                DISPATCH_WITH_POOLER(MeanIncludePooler<ctype>); \
                return;                                         \
            }                                                   \
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING: {         \
                auto sptr = src.ptr<ctype>();                   \
                auto dptr = dst.ptr<ctype>();                   \
                DISPATCH_WITH_POOLER(MeanExcludePooler<ctype>); \
                return;                                         \
            }                                                   \
        }                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
#undef DISPATCH_WITH_POOLER_AND_IDX_GETTER
#undef DISPATCH_WITH_POOLER
    megdnn_assert_internal(0);
}

WorkspaceBundle PoolingBackwardImpl::get_workspace_bundle(
        void* ptr, const TensorLayout& src, const TensorLayout& dst,
        const TensorLayout& diff, const TensorLayout& grad) const {
    SmallVector<size_t> sizes;
    TensorLayout fsrc = src;
    TensorLayout fdst = dst;
    TensorLayout fdiff = diff;
    TensorLayout fgrad = grad;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (MEGDNN_FLOAT16_SELECT(layout.dtype == dtype::BFloat16(), false)) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fsrc);
    get_workspace(fdst);
    get_workspace(fdiff);
    get_workspace(fgrad);
    return {ptr, std::move(sizes)};
}

size_t PoolingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst,
        const TensorLayout& diff, const TensorLayout& grad) {
    return get_workspace_bundle(nullptr, src, dst, diff, grad)
            .total_size_in_bytes();
}

void PoolingBackwardImpl::exec(_megdnn_tensor_in ssrc, _megdnn_tensor_in sdst,
                               _megdnn_tensor_in sdiff,
                               _megdnn_tensor_out sgrad,
                               _megdnn_workspace workspace) {
    check_exec(ssrc.layout, sdst.layout, sdiff.layout, sgrad.layout,
               workspace.size);
    TensorND src = ssrc;
    TensorND dst = sdst;
    TensorND diff = sdiff;
    TensorND grad = sgrad;
#if !MEGDNN_DISABLE_FLOAT16
    auto wsb = get_workspace_bundle(workspace.raw_ptr, ssrc.layout, sdst.layout,
                                    sdiff.layout, sgrad.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            static_cast<HandleImpl*>(handle()), &wsb);
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(ssrc, src)
                .src_to_comp_type(sdst, dst)
                .src_to_comp_type(sdiff, diff)
                .src_to_comp_type(sgrad, grad);
    }
#endif
    size_t c_pos, spatial_pos;
    if (param().format == Param::Format::NCHW) {
        c_pos = 1;
        spatial_pos = 2;
    } else {
        megdnn_assert(param().format == Param::Format::NHWC);
        c_pos = 3;
        spatial_pos = 1;
    }
    size_t N = src.layout.shape[0], C = src.layout.shape[c_pos],
           IH = src.layout.shape[spatial_pos + 0],
           IW = src.layout.shape[spatial_pos + 1];
    size_t OH = dst.layout.shape[spatial_pos + 0],
           OW = dst.layout.shape[spatial_pos + 1];
    size_t PH = param().pad_h, PW = param().pad_w;
    size_t FH = param().window_h, FW = param().window_w;
    size_t SH = param().stride_h, SW = param().stride_w;
#define DISPATCH_WITH_FUNC_AND_IDX_GETTER(Func, ctype, IdxGetter)            \
    MEGDNN_DISPATCH_CPU_KERN(static_cast<naive::HandleImpl*>(handle()),      \
                             Func<ctype MEGDNN_COMMA IdxGetter>(             \
                                     sptr, dptr, diffptr, gradptr, N, C, IH, \
                                     IW, OH, OW, PH, PW, SH, SW, FH, FW));   \

#define DISPATCH_WITH_FUNC(Func, ctype)                                    \
    switch (param().format) {                                              \
        case Param::Format::NCHW:                                          \
            DISPATCH_WITH_FUNC_AND_IDX_GETTER(Func, ctype, NCHWIdxGetter); \
            break;                                                         \
        case Param::Format::NHWC:                                          \
            DISPATCH_WITH_FUNC_AND_IDX_GETTER(Func, ctype, NHWCIdxGetter); \
            break;                                                         \
        default:                                                           \
            megdnn_throw("invalid pooling format");                        \
    }

#define cb(DType)                                                              \
    if (src.layout.dtype == DType()) {                                         \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        switch (param().mode) {                                                \
            case Mode::AVERAGE: {                                              \
                auto sptr = src.ptr<ctype>(), dptr = dst.ptr<ctype>(),         \
                     diffptr = diff.ptr<ctype>(), gradptr = grad.ptr<ctype>(); \
                DISPATCH_WITH_FUNC(pooling_backward_avg_impl, ctype);          \
                break;                                                         \
            }                                                                  \
            case Mode::AVERAGE_COUNT_EXCLUDE_PADDING: {                        \
                auto sptr = src.ptr<ctype>(), dptr = dst.ptr<ctype>(),         \
                     diffptr = diff.ptr<ctype>(), gradptr = grad.ptr<ctype>(); \
                DISPATCH_WITH_FUNC(pooling_backward_avg_expd_impl, ctype);     \
                break;                                                         \
            }                                                                  \
            case Mode::MAX: {                                                  \
                auto sptr = src.ptr<ctype>(), dptr = dst.ptr<ctype>(),         \
                     diffptr = diff.ptr<ctype>(), gradptr = grad.ptr<ctype>(); \
                DISPATCH_WITH_FUNC(pooling_backward_max_impl, ctype);          \
                break;                                                         \
            }                                                                  \
            default:                                                           \
                megdnn_assert_internal(0);                                     \
        }                                                                      \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
#undef DISPATCH_WITH_FUNC_AND_IDX_GETTER
#undef DISPATCH_WITH_FUNC
#if !MEGDNN_DISABLE_FLOAT16
    if (sgrad.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(grad, sgrad);
    }
#endif
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
