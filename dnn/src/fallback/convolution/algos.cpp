/**
 * \file dnn/src/fallback/convolution/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/convolution/algos.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/convolution/col2img_helper.h"
#include "src/fallback/convolution/run_conv.h"

#include "midout.h"

using namespace megdnn;
using namespace fallback;

MIDOUT_DECL(megdnn_fallback_conv)

namespace {

template <typename T>
void incr_ptr(T*& dst, ptrdiff_t delta) {
    dst = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(dst) + delta);
}

using NCBKernSizeParam = ConvolutionBackwardDataImpl::NCBKernSizeParam;
using NCBKernParam = ConvolutionBackwardDataImpl::NCBKernParam;

Relayout* get_relayout_opr() {
    static CpuOprDelegationStorage<> storage;
    return storage.get<Relayout>();
}

MatrixMul* get_matmul_opr(const NCBKernSizeParam& param) {
    using ConvCM = param::Convolution::ComputeMode;
    using MmCM = param::MatrixMul::ComputeMode;
    static CpuOprDelegationStorage<2> storage;
    switch (param.compute_mode) {
        default:
            return storage.get<MatrixMul, 0>({});
        case ConvCM::FLOAT32: {
            MatrixMul::Param p;
            p.compute_mode = MmCM::FLOAT32;
            return storage.get<MatrixMul, 1>(p);
        }
    }
}

WorkspaceBundle get_bundle(const NCBKernSizeParam& param) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(OW);
    bool can_matrix_mul_direct =
            (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0);
    // temp space to store unrolled matrix
    // workspace for matrix mul opr
    // workspace for relayout opr
    size_t part0, part1, part2;
    if (can_matrix_mul_direct) {
        part0 = 0;
    } else {
        part0 = (IC * FH * FW * IH * IW) * param.grad_type.size();
    }
    part2 = (OC * IC * FH * FW) * param.filter_type.size();
    {
        TensorLayout A_, B_, C_;
        A_ = TensorLayout({IC * FH * FW, OC}, param.filter_type);
        B_ = TensorLayout({OC, IH * IW}, param.diff_type);
        C_ = TensorLayout({IC * FH * FW, IH * IW}, param.grad_type);
        part1 = get_matmul_opr(param)->get_workspace_in_bytes(A_, B_, C_);
    }
    return {nullptr, {part0, part1, part2}};
}

template <typename ftype, typename dtype, typename gtype>
void kern_matmul(const NCBKernParam& param) {
    bool is_xcorr = !param.filter_meta.should_flip;
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    auto bundle = get_bundle(param);
    bundle.set(param.workspace_ptr);
    bool is1X1 =
            (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0);

    typedef void (*Func1)(const gtype*, gtype*, int, int, int, int, int, int,
                          int);
    typedef void (*Func2)(const gtype*, gtype*, int, int, int, int, int, int,
                          int, int, int, int, int);
    Func1 f1 = nullptr;
    Func2 f2 = nullptr;
    if (is_xcorr) {
        f1 = col2img<true>;
        f2 = col2img_stride_padding<true>;
    } else {
        f1 = col2img<false>;
        f2 = col2img_stride_padding<false>;
    }
    ftype* filter = const_cast<ftype*>(param.filter<ftype>());
    TensorND A_src, A_dst;
    {
        A_src.layout = TensorLayout({IC * FH * FW, OC},
                                    {static_cast<std::ptrdiff_t>(1),
                                     static_cast<std::ptrdiff_t>(IC * FH * FW)},
                                    param.filter_type);
        A_src.raw_ptr = static_cast<void*>(filter);
        A_dst.layout = TensorLayout({IC * FH * FW, OC}, param.filter_type);
        A_dst.raw_ptr = static_cast<void*>(bundle.get(2));
        // TODO Should be removed once armv8 convolution support transpose.
        get_relayout_opr()->exec(A_src, A_dst, inplace_cpu_handle().get());
    }
    for (size_t n = 0; n < N; ++n) {
        gtype *C_src, *C_dst;
        dtype* diff =
                const_cast<dtype*>(param.diff<dtype>() + n * param.inp_bs);
        gtype* grad = param.grad<gtype>() + n * param.out_bs;
        if (is1X1) {
            C_src = grad;
        } else {
            C_src = static_cast<gtype*>(bundle.get(0));
        }
        {
            TensorND B_, C_;
            B_.layout = TensorLayout({OC, IH * IW}, param.diff_type);
            B_.raw_ptr = static_cast<void*>(diff);
            C_.layout = TensorLayout({IC * FH * FW, IH * IW}, param.grad_type);
            C_.raw_ptr = C_src;
            Workspace workspace(static_cast<dt_byte*>(bundle.get(1)),
                                bundle.get_size(1));
            get_matmul_opr(param)->exec(A_dst, B_, C_, workspace);
        }

        if (!is1X1) {
            C_dst = grad;
            std::memset(C_dst, 0, param.grad_type.size() * IC * OH * OW);
            if (PH == 0 && PW == 0 && SH == 1 && SW == 1) {
                f1(C_src, C_dst, OH, OW, IC, IH, IW, FH, FW);
            } else {
                f2(C_src, C_dst, OH, OW, IC, IH, IW, FH, FW, SH, SW, PH, PW);
            }
        }
    }
}

void kern_direct(const NCBKernParam& param) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    auto diff = param.diff<float>(), filter = param.filter<float>();
    auto grad = param.grad<float>();
    for (size_t n = 0; n < N; ++n) {
        convolution::run_conv_backward_data(
                diff + n * param.inp_bs, filter, grad + n * param.out_bs,
                param.workspace_ptr, IH, IW, IC, FH, FW, OH, OW, OC, PH, PW, SH,
                SW, !param.filter_meta.should_flip);
    }
}

}  // namespace

/* ===================== fallback algo ===================== */

bool ConvolutionImpl::AlgoFallback::usable(
        ConvolutionImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    auto&& fm = param.filter_meta;
    return fm.format == param::Convolution::Format::NCHW &&
           param.src_type.enumv() == DTypeEnum::Float32 &&
           param.filter_type.enumv() == DTypeEnum::Float32 &&
           param.dst_type.enumv() == DTypeEnum::Float32 &&
           fm.spatial_ndim == 2 && fm.dilation[0] == 1 && fm.dilation[1] == 1;
}

size_t ConvolutionImpl::AlgoFallback::get_workspace(
        ConvolutionImpl*, const NCBKernSizeParam& param) const {
    auto FH = param.filter_meta.spatial[0], FW = param.filter_meta.spatial[1];
    size_t nr_threads = param.nr_threads;
    if (param.filter_meta.should_flip) {
        // need transpose filter
        return WorkspaceBundle{nullptr, {FH * FW * sizeof(float)}}
                       .total_size_in_bytes() *
               nr_threads;
    } else {
        return 0;
    }
}

SmallVector<ConvolutionImpl::NCBKern>
ConvolutionImpl::AlgoFallback::dispatch_kern(
        ConvolutionImpl* opr, const NCBKernSizeParam& param) const {
    size_t group = param.filter_meta.group;
    size_t N = param.n;
    size_t nr_threads = param.nr_threads;
    size_t workspace_per_thread = get_workspace(opr, param) / nr_threads;
    auto kern_fallback = [workspace_per_thread](const NCBKernParam& p,
                                                const NCBKernIndex& ncb_index) {
        UNPACK_CONV_F32_NCB_KERN_SIZES(p);
        MEGDNN_MARK_USED_VAR(N);
        auto src = p.src<float>(), filter = p.filter<float>();
        auto dst = p.dst<float>();
        size_t thread_id = ncb_index.thread_id;
        void* workspace_ptr = reinterpret_cast<void*>(
                reinterpret_cast<ptrdiff_t>(p.workspace_ptr) +
                workspace_per_thread * thread_id);
        convolution::run_conv(src, filter, dst, workspace_ptr, IH, IW, IC, FH,
                              FW, OH, OW, OC, PH, PW, SH, SW,
                              !p.filter_meta.should_flip);
    };
    return {{kern_fallback, {group, N, 1_z}}};
}

/* ===================== naive algo ===================== */

bool ConvolutionImpl::AlgoNaive::usable(
        ConvolutionImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    bool ret = false;

#define cb(dt) ret |= (param.src_type.enumv() == DTypeTrait<dt>::enumv);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
#define cb(dt_src, dt_dst)                                            \
    ret |= (param.src_type.enumv() == DTypeTrait<dt_src>::enumv &&    \
            param.filter_type.enumv() == DTypeTrait<dt_src>::enumv && \
            param.dst_type.enumv() == DTypeTrait<dt_dst>::enumv)
    cb(dtype::Int8, dtype::Int16);
    cb(dtype::Int8, dtype::Int32);
    cb(dtype::Quantized8Asymm, dtype::QuantizedS32);
    cb(dtype::QuantizedS8, dtype::QuantizedS32);
#undef cb
    ret = ret &&
          (param.filter_meta.format == param::Convolution::Format::NCHW ||
           param.filter_meta.format == param::Convolution::Format::NHWC);
    return ret;
}

SmallVector<ConvolutionImpl::NCBKern> ConvolutionImpl::AlgoNaive::dispatch_kern(
        ConvolutionImpl*, const NCBKernSizeParam& param) const {
    size_t N = param.n;
    size_t group = param.filter_meta.group;
#define cb(dt, cmode, compute_type)                                      \
    do {                                                                 \
        if (param.src_type.enumv() == DTypeTrait<dt>::enumv &&           \
            param.compute_mode == param::ConvBias::ComputeMode::cmode) { \
            using ctype = DTypeTrait<dt>::ctype;                         \
            using comp_type = DTypeTrait<compute_type>::ctype;           \
            return {{kern_naive_forward<ctype, ctype, comp_type>,        \
                     {group, N, 1_z}}};                                  \
        }                                                                \
    } while (0)

    cb(dtype::Float32, DEFAULT, dtype::Float32);
#if !MEGDNN_DISABLE_FLOAT16
    cb(dtype::Float16, DEFAULT, dtype::Float16);
    cb(dtype::Float16, FLOAT32, dtype::Float32);
#endif
#undef cb

#define cb(dt_src, dt_dst)                                            \
    do {                                                              \
        if (param.src_type.enumv() == DTypeTrait<dt_src>::enumv &&    \
            param.filter_type.enumv() == DTypeTrait<dt_src>::enumv && \
            param.dst_type.enumv() == DTypeTrait<dt_dst>::enumv) {    \
            return {{kern_naive_forward<DTypeTrait<dt_src>::ctype,    \
                                        DTypeTrait<dt_dst>::ctype,    \
                                        DTypeTrait<dt_dst>::ctype>,   \
                     {group, N, 1_z}}};                               \
        }                                                             \
    } while (0)
    cb(dtype::Int8, dtype::Int16);
    cb(dtype::Int8, dtype::Int32);
    cb(dtype::Quantized8Asymm, dtype::QuantizedS32);
    cb(dtype::QuantizedS8, dtype::QuantizedS32);
    megdnn_throw(megdnn_mangle("unknown convolution data type"));
#undef cb
}

/* ===================== default algo ===================== */

ConvolutionImpl::AlgoDefault::AlgoDefault(fallback::ConvBiasImpl* conv_bias_opr,
                                          ConvBiasImpl::AlgoBase* algorithm)
        : m_conv_bias_opr(conv_bias_opr), m_algorithm(algorithm) {
    megdnn_assert_internal(algorithm);
    m_name = ssprintf("CONVOLUTION_DEFAULT_%s", m_algorithm->name());
}

ConvBiasImpl::NCBKernSizeParam
ConvolutionImpl::AlgoDefault::AlgoDefault::init_convbias_opr_and_param(
        ConvBiasImpl* conv_bias_opr, const NCBKernSizeParam& param) {
    DType bias_type = param.dst_type;
    if (bias_type.category() == DTypeCategory::QUANTIZED) {
        bias_type = dtype::QuantizedS32(
                mul_scale(param.src_type, param.filter_type));
    }

    ::ConvBiasImpl::NCBKernSizeParam conv_bias_size_param(
            param, 0, param::MatrixMul::Format::DEFAULT, bias_type, 0,
            BiasMode::NO_BIAS, param::ConvBias::NonlineMode::IDENTITY);
    // nonline mode
    conv_bias_opr->param().nonlineMode = conv_bias_size_param.nonlineMode;
    // convolution mode
    if (conv_bias_size_param.filter_meta.should_flip) {
        conv_bias_opr->param().mode = param::ConvolutionV0::Mode::CONVOLUTION;
    } else {
        conv_bias_opr->param().mode =
                param::ConvolutionV0::Mode::CROSS_CORRELATION;
    }
    // sparse
    if (conv_bias_size_param.filter_meta.group > 1) {
        conv_bias_opr->param().sparse = param::ConvolutionV0::Sparse::GROUP;
    } else {
        conv_bias_opr->param().sparse = param::ConvolutionV0::Sparse::DENSE;
    }
    // format
    conv_bias_opr->param().format = conv_bias_size_param.filter_meta.format;
    // pad stride dilate
    conv_bias_opr->param().pad_h = conv_bias_size_param.filter_meta.padding[0];
    conv_bias_opr->param().pad_w = conv_bias_size_param.filter_meta.padding[1];
    conv_bias_opr->param().stride_h =
            conv_bias_size_param.filter_meta.stride[0];
    conv_bias_opr->param().stride_w =
            conv_bias_size_param.filter_meta.stride[1];
    conv_bias_opr->param().dilate_h =
            conv_bias_size_param.filter_meta.dilation[0];
    conv_bias_opr->param().dilate_w =
            conv_bias_size_param.filter_meta.dilation[1];
    // output_block_size
    conv_bias_opr->param().output_block_size =
            conv_bias_size_param.output_block_size;
    // compute_mode
    conv_bias_opr->param().compute_mode = conv_bias_size_param.compute_mode;

    return conv_bias_size_param;
}

bool ConvolutionImpl::AlgoDefault::is_preferred(
        ConvolutionImpl*, const NCBKernSizeParam& param) const {
    ::ConvBiasImpl::NCBKernSizeParam conv_bias_param =
            init_convbias_opr_and_param(m_conv_bias_opr, param);
    return m_algorithm->is_preferred(m_conv_bias_opr, conv_bias_param);
}

bool ConvolutionImpl::AlgoDefault::usable(
        ConvolutionImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    ::ConvBiasImpl::NCBKernSizeParam conv_bias_param =
            init_convbias_opr_and_param(m_conv_bias_opr, param);
    return m_algorithm->usable(m_conv_bias_opr, conv_bias_param,
                               static_cast<ConvBiasImpl::AlgoSelectionStrategy>(
                                       algo_selection_strategy));
}

WorkspaceBundle ConvolutionImpl::AlgoDefault::get_bundle(
        const NCBKernSizeParam& param) const {
    ::ConvBiasImpl::NCBKernSizeParam conv_bias_param =
            init_convbias_opr_and_param(m_conv_bias_opr, param);
    m_conv_bias_opr->execution_policy() = {m_algorithm};
    return WorkspaceBundle(nullptr, {m_algorithm->get_workspace(
                                            m_conv_bias_opr, conv_bias_param)});
}

size_t ConvolutionImpl::AlgoDefault::get_workspace(
        ConvolutionImpl*, const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}

//! Return the implment kernel
SmallVector<ConvolutionImpl::NCBKern> ConvolutionImpl::AlgoDefault::get_kimpl(
        ::ConvBiasImpl* conv_bias_opr, ConvBiasImpl::AlgoBase* algo,
        const NCBKernSizeParam& param) {
    MIDOUT_BEGIN(megdnn_fallback_conv, midout_iv(0)) {
        // construct the conv_bias kern param
        ::ConvBiasImpl::NCBKernParam conv_bias_param;
        ::ConvBiasImpl::NCBKernSizeParam conv_bias_size_param =
                init_convbias_opr_and_param(conv_bias_opr, param);
        static_cast<::ConvBiasImpl::NCBKernSizeParam&>(conv_bias_param) =
                conv_bias_size_param;
        auto conv_bias_kerns =
                algo->dispatch_kerns(conv_bias_opr, conv_bias_param);
        SmallVector<ConvolutionImpl::NCBKern> convolution_kerns;

        //! Set the conv_bias param using convolution param
        auto set_copy_param_run_time_address =
                [](const NCBKernParam& conv_param,
                   ::ConvBiasImpl::NCBKernParam& copied_param) {
                    copied_param.src_ptr = conv_param.src_ptr;
                    copied_param.filter_ptr = conv_param.filter_ptr;
                    copied_param.dst_ptr = conv_param.dst_ptr;
                    copied_param.workspace_ptr = conv_param.workspace_ptr;
                    copied_param.workspace_size = conv_param.workspace_size;
                };
        for (size_t i = 0; i < conv_bias_kerns.size(); i++) {
            auto kernel = conv_bias_kerns[i];
            //! If the kerenl batch parallel
            auto run = [=](const NCBKernParam& p,
                           const NCBKernIndex& ncb_index) {
                auto copy_param = conv_bias_param;
                set_copy_param_run_time_address(p, copy_param);
                kernel.kern(copy_param,
                            {ncb_index.thread_id, ncb_index.ndrange_id});
            };
            convolution_kerns.push_back({run, kernel.global_size});
        }
        return convolution_kerns;
    }
    MIDOUT_END();
}

/* ===================== direct algo ===================== */

bool ConvolutionBackwardDataImpl::AlgoDirect::usable(
        ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    return fm.format == param::Convolution::Format::NCHW &&
           param.diff_type.enumv() == DTypeEnum::Float32 &&
           param.filter_type.enumv() == DTypeEnum::Float32 &&
           param.grad_type.enumv() == DTypeEnum::Float32 &&
           fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1;
}

size_t ConvolutionBackwardDataImpl::AlgoDirect::get_workspace(
        ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param) const {
    auto FH = param.filter_meta.spatial[0], FW = param.filter_meta.spatial[1];
    if (param.filter_meta.should_flip) {
        // need transpose filter
        return FH * FW * sizeof(float);
    } else {
        return 0;
    }
}

ConvolutionBackwardDataImpl::ncb_kern_t
ConvolutionBackwardDataImpl::AlgoDirect::dispatch_kern(
        ConvolutionBackwardDataImpl*, const NCBKernSizeParam&) const {
    return kern_direct;
}

/* ===================== Matrix mul algo ===================== */

bool ConvolutionBackwardDataImpl::AlgoMatrixMul::usable(
        ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    return fm.format == param::Convolution::Format::NCHW &&
           fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1;
}

size_t ConvolutionBackwardDataImpl::AlgoMatrixMul::get_workspace(
        ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}

ConvolutionBackwardDataImpl::ncb_kern_t
ConvolutionBackwardDataImpl::AlgoMatrixMul::dispatch_kern(
        ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param) const {
#define cb(dt)                                                    \
    do {                                                          \
        if (param.filter_type.enumv() == DTypeTrait<dt>::enumv) { \
            using ctype = DTypeTrait<dt>::ctype;                  \
            return kern_matmul<ctype, ctype, ctype>;              \
        }                                                         \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb

#define cb(dt_src, dt_dst)                                            \
    do {                                                              \
        if (param.diff_type.enumv() == DTypeTrait<dt_src>::enumv &&   \
            param.filter_type.enumv() == DTypeTrait<dt_src>::enumv && \
            param.grad_type.enumv() == DTypeTrait<dt_dst>::enumv) {   \
            return kern_matmul<DTypeTrait<dt_src>::ctype,             \
                               DTypeTrait<dt_src>::ctype,             \
                               DTypeTrait<dt_dst>::ctype>;            \
        }                                                             \
    } while (0)
    cb(dtype::Int8, dtype::Int32);
    cb(dtype::QuantizedS8, dtype::QuantizedS32);
    cb(dtype::Quantized8Asymm, dtype::QuantizedS32);
    megdnn_throw("unsupported data type on matrix mul");
#undef cb
}

// vim: syntax=cpp.doxygen
