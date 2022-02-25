#include "src/x86/pooling/algo.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/pooling/opr_impl.h"
#include "src/naive/handle.h"
#include "src/x86/handle.h"
#include "src/x86/pooling/do_max_pooling_3x3_s2x2_float_sse.h"
#include "src/x86/pooling/pooling_special_cases.h"
#include "src/x86/utils.h"

#include "src/x86/avx_helper.h"

using namespace megdnn;
using namespace x86;

namespace {

#if MEGDNN_X86_WITH_MKL_DNN
template <dnnl::memory::format_tag format_tag, bool use_mkl_mem>
dnnl::memory tensor_to_mkl_memory(
        _megdnn_tensor_in src, const dnnl::engine& mkldnn_eng,
        dnnl::memory::data_type mkldnn_datatype) {
    megdnn_assert(
            format_tag == dnnl::memory::format_tag::nChw8c ||
                    format_tag == dnnl::memory::format_tag::nchw ||
                    format_tag == dnnl::memory::format_tag::nhwc,
            "not support format");

    dnnl::memory::dims src_shape = {
            static_cast<long>(src.layout[0]), static_cast<long>(src.layout[1]),
            static_cast<long>(src.layout[2]), static_cast<long>(src.layout[3])};
    if (format_tag == dnnl::memory::format_tag::nChw8c) {
        src_shape = {
                static_cast<long>(src.layout[0]), static_cast<long>(src.layout[1] * 8),
                static_cast<long>(src.layout[2]), static_cast<long>(src.layout[3])};
    }
    auto megdnn_src_md = dnnl::memory::desc({src_shape}, mkldnn_datatype, format_tag);
    if (use_mkl_mem) {
        auto megdnn_src_memory = dnnl::memory(megdnn_src_md, mkldnn_eng);
        return megdnn_src_memory;
    } else {
        auto megdnn_src_memory = dnnl::memory(megdnn_src_md, mkldnn_eng, src.raw_ptr());
        return megdnn_src_memory;
    }
}

#endif

}  // namespace

PoolingImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_mean_w2s2_avx);
    all_algos.push_back(&algo_mean_w2s2_sse3);
    all_algos.push_back(&algo_max_w2s2_sse);
    all_algos.push_back(&algo_max_w3s3_sse);
    all_algos.push_back(&algo_max_w13s1_nchw88_avx);
#if MEGDNN_X86_WITH_MKL_DNN
    all_algos.push_back(&algo_mkldnn_nchw);
    all_algos.push_back(&algo_mkldnn_nchw88);
#endif
    all_algos.push_back(&algo_fallback);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

PoolingImpl::AlgoPack PoolingImpl::sm_algo_pack;
MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingImpl)

PoolingImpl::AlgoBase::SizeArgs::SizeArgs(
        PoolingImpl* o, const TensorLayout& src, const TensorLayout& dst)
        : handle{static_cast<x86::HandleImpl*>(o->handle())},
          opr{o},
          layout_src{src},
          layout_dst{dst} {}

PoolingImpl::AlgoBase::ExecArgs::ExecArgs(
        PoolingImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, dst.layout),
          src_tensor{src},
          dst_tensor{dst},
          workspace{workspace} {}

std::string PoolingImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf(
            "src=%s, dst=%s", layout_src.to_string().c_str(),
            layout_dst.to_string().c_str());
}

bool PoolingImpl::AlgoMeanW2S2AVX::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::AVX) && args.opr->param().mode == Mode::AVERAGE &&
            args.opr->param().format == Param::Format::NCHW &&
            args.layout_src.dtype == dtype::Float32() && FH == 2 && FW == 2 &&
            SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMeanW2S2AVX::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;

    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor.raw_ptr());
            auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor.raw_ptr());
            rep(n, N) rep(c, C) {
                mean_pooling_w2x2_s2x2_avx(
                        sptr + n * C * IH * IW + c * IH * IW, IH, IW,
                        dptr + n * C * OH * OW + c * OH * OW, OH, OW, PH, PW, true);
            });
}

bool PoolingImpl::AlgoMeanW2S2SSE3::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::SSE3) && args.opr->param().mode == Mode::AVERAGE &&
            args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().format == Param::Format::NCHW && FH == 2 && FW == 2 &&
            SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMeanW2S2SSE3::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;

    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor.raw_ptr());
            auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor.raw_ptr());
            rep(n, N) rep(c, C) {
                mean_pooling_w2x2_s2x2_sse3(
                        sptr + n * C * IH * IW + c * IH * IW, IH, IW,
                        dptr + n * C * OH * OW + c * OH * OW, OH, OW, PH, PW, true);
            });
}

bool PoolingImpl::AlgoMaxW2S2SSE::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::SSE) && args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW && FH == 2 && FW == 2 &&
            SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMaxW2S2SSE::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;

    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor.raw_ptr());
            auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor.raw_ptr());
            rep(n, N) rep(c, C) {
                max_pooling_w2x2_s2x2_sse(
                        sptr + n * C * IH * IW + c * IH * IW, IH, IW,
                        dptr + n * C * OH * OW + c * OH * OW, OH, OW, PH, PW);
            });
}

bool PoolingImpl::AlgoMaxW3S3SSE::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::SSE) && args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW && FH == 3 && FW == 3 &&
            SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMaxW3S3SSE::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto handle = [=]() { return args.handle; };
    WorkspaceBundle ws =
            get_bundle(args.layout_src, args.layout_dst, args.opr->param());
    ws.set(args.workspace.raw_ptr);
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor.raw_ptr());
            auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor.raw_ptr());
            rep(n, N) rep(c, C) {
                do_max_pooling_3x3_s2x2_float_SSE(
                        sptr + n * C * IH * IW + c * IH * IW,
                        dptr + n * C * OH * OW + c * OH * OW, IH, IW, OH, OW, PH, PW,
                        ws);
            });
}

#if MEGDNN_X86_WITH_MKL_DNN
bool PoolingImpl::AlgoMKLDNNNCHW::is_available(const SizeArgs& args) const {
    return ((args.layout_src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
             args.layout_src.dtype.enumv() == DTypeEnum::Int8) &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW);
}

void PoolingImpl::AlgoMKLDNNNCHW::exec(const ExecArgs& args) const {
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto handle = [=]() { return args.handle; };

    auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
    auto mkldnn_eng = x86_handle->mkldnn_engine();
    auto mkldnn_stream = x86_handle->mkldnn_stream();
    auto mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
    dnnl::memory::dims pool_strides = {SH, SW};
    dnnl::memory::dims pool_padding = {PH, PW};
    dnnl::memory::dims pool_kernel = {FH, FW};

    auto run = [args, pool_strides, pool_padding, pool_kernel, mkldnn_eng,
                mkldnn_stream, mkldnn_pooling_mode](void) {
        dnnl::memory&& megdnn_src_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nchw, false>(
                        args.src_tensor, mkldnn_eng, dnnl::memory::data_type::s8);
        dnnl::memory&& megdnn_dst_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nchw, false>(
                        args.dst_tensor, mkldnn_eng, dnnl::memory::data_type::s8);

        dnnl::memory&& megdnn_src_memory =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nhwc, true>(
                        args.src_tensor, mkldnn_eng, dnnl::memory::data_type::s8);
        dnnl::memory&& megdnn_dst_memory =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nhwc, true>(
                        args.dst_tensor, mkldnn_eng, dnnl::memory::data_type::s8);

        auto reorder_src = dnnl::reorder(megdnn_src_memory_ori, megdnn_src_memory);
        auto reorder_dst = dnnl::reorder(megdnn_dst_memory, megdnn_dst_memory_ori);
        auto pool1_desc = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_inference, mkldnn_pooling_mode,
                megdnn_src_memory.get_desc(), megdnn_dst_memory.get_desc(),
                pool_strides, pool_kernel, pool_padding, pool_padding);
        auto pool_pd = dnnl::pooling_forward::primitive_desc(pool1_desc, mkldnn_eng);
        auto pool = dnnl::pooling_forward(pool_pd);
        MEGDNN_MARK_USED_VAR(mkldnn_eng);
        auto mkl_stream = mkldnn_stream;
        reorder_src.execute(
                mkl_stream, {{DNNL_ARG_FROM, megdnn_src_memory_ori},
                             {DNNL_ARG_TO, megdnn_src_memory}});
        pool.execute(
                mkl_stream,
                {{DNNL_ARG_SRC, megdnn_src_memory}, {DNNL_ARG_DST, megdnn_dst_memory}});
        reorder_dst.execute(
                mkl_stream, {{DNNL_ARG_FROM, megdnn_dst_memory},
                             {DNNL_ARG_TO, megdnn_dst_memory_ori}});
        mkl_stream.wait();
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(run());
}

#endif

#if MEGDNN_X86_WITH_MKL_DNN
bool PoolingImpl::AlgoMKLDNNNCHW88::is_available(const SizeArgs& args) const {
    return (args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW88);
}

void PoolingImpl::AlgoMKLDNNNCHW88::exec(const ExecArgs& args) const {
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto handle = [=]() { return args.handle; };

    auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
    auto mkldnn_eng = x86_handle->mkldnn_engine();
    auto mkldnn_stream = x86_handle->mkldnn_stream();
    auto mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
    switch (args.opr->param().mode) {
        case Mode::MAX:
            mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
            break;
        case Mode::AVERAGE:
            mkldnn_pooling_mode = dnnl::algorithm::pooling_avg_include_padding;
            break;
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mkldnn_pooling_mode = dnnl::algorithm::pooling_avg_exclude_padding;
            break;
        default:
            megdnn_throw("not supported pooling mode\n");
    };

    dnnl::memory::dims pool_strides = {SH, SW};
    dnnl::memory::dims pool_padding = {PH, PW};
    dnnl::memory::dims pool_kernel = {FH, FW};
    auto run = [args, pool_strides, pool_padding, pool_kernel, mkldnn_eng,
                mkldnn_stream, mkldnn_pooling_mode](void) {
        dnnl::memory&& megdnn_src_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nChw8c, false>(
                        args.src_tensor, mkldnn_eng, dnnl::memory::data_type::f32);
        dnnl::memory&& megdnn_dst_memory_ori =
                tensor_to_mkl_memory<dnnl::memory::format_tag::nChw8c, false>(
                        args.dst_tensor, mkldnn_eng, dnnl::memory::data_type::f32);
        auto pool_desc = dnnl::pooling_forward::desc(
                dnnl::prop_kind::forward_inference, mkldnn_pooling_mode,
                megdnn_src_memory_ori.get_desc(), megdnn_dst_memory_ori.get_desc(),
                pool_strides, pool_kernel, pool_padding, pool_padding);
        auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, mkldnn_eng);
        auto pool = dnnl::pooling_forward(pool_pd);
        MEGDNN_MARK_USED_VAR(mkldnn_eng);
        auto mkl_stream = mkldnn_stream;

        pool.execute(
                mkl_stream, {{DNNL_ARG_SRC, megdnn_src_memory_ori},
                             {DNNL_ARG_DST, megdnn_dst_memory_ori}});
        mkl_stream.wait();
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(run());
}

#endif

namespace {
MEGDNN_ATTRIBUTE_TARGET("avx")
void max_pooling_s1_nchw88_avx_kern(
        const float* src, float* dst, int IH, int IW, int OH, int OW, int PH, int PW,
        int WH, int WW) {
    static float min_float = -std::numeric_limits<float>::max();
    static int VECSIZE = 8;

    __m256 ymm[16];
    const float* psrc = src;
    float* pdst = dst;

    //! deal all rows
    for (int row = 0; row < IH; ++row) {
        for (int j = 0; j < PW; ++j) {
            ymm[j] = _mm256_set1_ps(min_float);
        }
        int col_end = WW - PW < IW ? WW - PW : IW;
        for (int j = 0; j < col_end; ++j) {
            ymm[j + PW] = _mm256_loadu_ps(psrc + j * VECSIZE);
        }
        for (int j = col_end + PW; j < WW; ++j) {
            ymm[j] = _mm256_set1_ps(min_float);
        }

        int col_next = WW - PW;
        for (int j = 0; j < OW; ++j) {
            for (int i = WW - 2; i >= 0; --i) {
                ymm[i] = _mm256_max_ps(ymm[i], ymm[i + 1]);
            }
            _mm256_storeu_ps(pdst, ymm[0]);
            pdst += VECSIZE;
            for (int i = 0; i < WW - 1; ++i) {
                ymm[i] = ymm[i + 1];
            }
            if (col_next < IW) {
                ymm[WW - 1] = _mm256_loadu_ps(psrc + col_next * VECSIZE);
                col_next++;
            } else {
                ymm[WW - 1] = _mm256_set1_ps(min_float);
            }
        }
        psrc += IW * VECSIZE;
    }

    //! deal all cols
    float* src1 = dst;
    for (int col = 0; col < OW; ++col) {
        for (int j = 0; j < PH; ++j) {
            ymm[j] = _mm256_set1_ps(min_float);
        }
        int row_end = WH - PH < IH ? WH - PH : IH;
        for (int j = 0; j < row_end; ++j) {
            ymm[j + PH] = _mm256_loadu_ps(src1 + j * OW * VECSIZE);
        }
        for (int j = row_end + PH; j < WH; ++j) {
            ymm[j] = _mm256_set1_ps(min_float);
        }

        int row_next = WH - PH;
        pdst = src1;
        for (int j = 0; j < OH; ++j) {
            for (int i = WH - 2; i >= 0; --i) {
                ymm[i] = _mm256_max_ps(ymm[i], ymm[i + 1]);
            }
            _mm256_storeu_ps(pdst, ymm[0]);
            pdst += OW * VECSIZE;
            for (int i = 0; i < WH - 1; ++i) {
                ymm[i] = ymm[i + 1];
            }
            if (row_next < IH) {
                ymm[WH - 1] = _mm256_loadu_ps(src1 + row_next * OW * VECSIZE);
                row_next++;
            } else {
                ymm[WH - 1] = _mm256_set1_ps(min_float);
            }
        }
        src1 += VECSIZE;
    }
}
}  // namespace

bool PoolingImpl::AlgoMaxS1NCHW88AVX::is_available(const SizeArgs& args) const {
    bool is_dtype_ok = args.layout_src.dtype == dtype::Float32();
    bool is_mode_ok = args.opr->param().mode == Mode::MAX;
    bool is_format_ok = args.opr->param().format == Param::Format::NCHW88;
    bool is_shape_ok =
            args.opr->param().window_h >= 10 && args.opr->param().window_h <= 15 &&
            args.opr->param().window_w >= 10 && args.opr->param().window_w <= 15;
    bool is_stride_ok =
            args.opr->param().stride_h == 1 && args.opr->param().stride_w == 1;
    //! this condition guarantee size of dst's memory is bigger enough because
    //! dst's memory will be used as workspace to store intermediate result.
    bool is_pad_ok = args.opr->param().pad_h >= args.opr->param().window_h / 2 &&
                     args.opr->param().pad_w >= args.opr->param().window_w / 2;
    bool is_ins_ok = is_supported(SIMDType::AVX);
    return is_dtype_ok && is_mode_ok && is_format_ok && is_shape_ok && is_pad_ok &&
           is_stride_ok && is_ins_ok;
}

void PoolingImpl::AlgoMaxS1NCHW88AVX::exec(const ExecArgs& args) const {
    auto handle = args.handle;
    size_t N = args.layout_src.shape[0];
    static size_t VECSIZE = 8;
    size_t PH = args.opr->param().pad_h;
    size_t PW = args.opr->param().pad_w;
    size_t WH = args.opr->param().window_h;
    size_t WW = args.opr->param().window_w;
    size_t IC = args.layout_src.shape[1];
    size_t IH = args.layout_src.shape[2];
    size_t IW = args.layout_src.shape[3];
    size_t OH = args.layout_dst.shape[2];
    size_t OW = args.layout_dst.shape[3];

    auto run = [args, IC, IH, IW, OH, OW, PH, PW, WH, WW](size_t index, size_t) {
        float* src_ptr = reinterpret_cast<float*>(args.src_tensor.raw_ptr());
        float* dst_ptr = reinterpret_cast<float*>(args.dst_tensor.raw_ptr());
        size_t n = index / IC;
        size_t c = index % IC;
        float* src = src_ptr + n * IH * IW * IC * VECSIZE + IH * IW * c * VECSIZE;
        float* dst = dst_ptr + n * OH * OW * IC * VECSIZE + OH * OW * c * VECSIZE;
        max_pooling_s1_nchw88_avx_kern(src, dst, IH, IW, OH, OW, PH, PW, WH, WW);
    };
    MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(handle, N * IC, run);
}
