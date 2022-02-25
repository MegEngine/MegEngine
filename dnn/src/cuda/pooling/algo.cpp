#include "./algo.h"
#include "./pooling2d_qint.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

namespace {
#define V1(v) #v
#define V(v)  V1(v)
#define DEF_NAME(NAME) \
#NAME "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)
}  // namespace

PoolingForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_chwn4);
    all_algos.push_back(&algo_nchw4);
    all_algos.push_back(&algo_nchw32);
    all_algos.push_back(&algo_nhwc);
    all_algos.push_back(&algo_nchw64);
    all_algos.push_back(&algo_cudnn);
#if CUDNN_VERSION >= 6000
    all_algos.push_back(&algo_cudnn_max_deterministic);
#endif

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

PoolingForwardImpl::AlgoPack PoolingForwardImpl::sm_algo_pack;
MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingForwardImpl)

PoolingForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        PoolingForwardImpl* o, const TensorLayout& src, const TensorLayout& dst)
        : handle{concrete_handle(o->handle())},
          opr{o},
          layout_src{&src},
          layout_dst{&dst} {}

PoolingForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        PoolingForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, dst.layout),
          src_tensor{&src},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string PoolingForwardImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf(
            "src=%s, dst=%s", layout_src->to_string().c_str(),
            layout_dst->to_string().c_str());
}

WorkspaceBundle PoolingForwardImpl::AlgoBase::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    SmallVector<size_t> sizes;
    TensorLayout fsrc = *args.layout_src;
    TensorLayout fdst = *args.layout_dst;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fsrc);
    get_workspace(fdst);
    return {ptr, std::move(sizes)};
}

size_t PoolingForwardImpl::AlgoBase::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

bool PoolingForwardImpl::AlgoCUDNN::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return (((args.opr->param().format == Format::NCHW ||
              args.opr->param().format == Format::NHWC) &&
             (args.layout_src->dtype.enumv() == DTypeEnum::Float16 ||
              args.layout_src->dtype.enumv() == DTypeEnum::BFloat16 ||
              args.layout_src->dtype.enumv() == DTypeEnum::Float32 ||
              args.layout_src->dtype.enumv() == DTypeEnum::Int8 ||
              args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS32 ||
              args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8 ||
              args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm)) ||
            ((args.opr->param().format == Format::NCHW4 ||
              args.opr->param().format == Format::NCHW32) &&
             (args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8 ||
              args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm)));
}

void PoolingForwardImpl::AlgoCUDNN::init_mode(
        const ExecArgs& args, cudnnPoolingMode_t& mode) const {
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = CUDNN_POOLING_MAX;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }
}

void PoolingForwardImpl::AlgoCUDNN::exec(const ExecArgs& args) const {
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;
    auto wsb = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(args.handle), &wsb);
    if (args.layout_src->dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(*args.src_tensor, src)
                .src_to_comp_type(*args.dst_tensor, dst);
    }
    {
        dt_float32 alpha = 1.0f, beta = 0.0f;
        TensorDesc src_desc, dst_desc;
        src_desc.set(src.layout, args.opr->param().format);
        dst_desc.set(dst.layout, args.opr->param().format);

        cudnnPoolingMode_t mode;
        init_mode(args, mode);

        cudnnPoolingDescriptor_t cudnn_desc;
        cudnn_check(cudnnCreatePoolingDescriptor(&cudnn_desc));
        cudnn_check(cudnnSetPooling2dDescriptor(
                cudnn_desc, mode, CUDNN_NOT_PROPAGATE_NAN, args.opr->param().window_h,
                args.opr->param().window_w, args.opr->param().pad_h,
                args.opr->param().pad_w, args.opr->param().stride_h,
                args.opr->param().stride_w));
        cudnn_check(cudnnPoolingForward(
                args.handle->cudnn_handle(), cudnn_desc, &alpha, src_desc.desc,
                src.raw_ptr(), &beta, dst_desc.desc, dst.raw_ptr()));
        cudnn_check(cudnnDestroyPoolingDescriptor(cudnn_desc));
    }
    if (args.layout_src->dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(dst, *args.dst_tensor);
    }
}

#if CUDNN_VERSION >= 6000
bool PoolingForwardImpl::AlgoCUDNNMAXDETERMINISTIC::is_available(
        const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return (args.opr->param().mode == param::Pooling::Mode::MAX &&
            (((args.opr->param().format == Format::NCHW ||
               args.opr->param().format == Format::NHWC) &&
              (args.layout_src->dtype.enumv() == DTypeEnum::Float16 ||
               args.layout_src->dtype.enumv() == DTypeEnum::BFloat16 ||
               args.layout_src->dtype.enumv() == DTypeEnum::Float32 ||
               args.layout_src->dtype.enumv() == DTypeEnum::Int8 ||
               args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS32 ||
               args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8 ||
               args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm)) ||
             ((args.opr->param().format == Format::NCHW4 ||
               args.opr->param().format == Format::NCHW32) &&
              (args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8 ||
               args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm))));
}

void PoolingForwardImpl::AlgoCUDNNMAXDETERMINISTIC::init_mode(
        const ExecArgs& args, cudnnPoolingMode_t& mode) const {
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = CUDNN_POOLING_MAX_DETERMINISTIC;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }
}

void PoolingForwardImpl::AlgoCUDNNMAXDETERMINISTIC::exec(const ExecArgs& args) const {
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;
    auto wsb = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(args.handle), &wsb);
    if (args.layout_src->dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(*args.src_tensor, src)
                .src_to_comp_type(*args.dst_tensor, dst);
    }
    {
        dt_float32 alpha = 1.0f, beta = 0.0f;
        TensorDesc src_desc, dst_desc;
        src_desc.set(src.layout, args.opr->param().format);
        dst_desc.set(dst.layout, args.opr->param().format);

        cudnnPoolingMode_t mode;
        init_mode(args, mode);

        cudnnPoolingDescriptor_t cudnn_desc;
        cudnn_check(cudnnCreatePoolingDescriptor(&cudnn_desc));
        cudnn_check(cudnnSetPooling2dDescriptor(
                cudnn_desc, mode, CUDNN_NOT_PROPAGATE_NAN, args.opr->param().window_h,
                args.opr->param().window_w, args.opr->param().pad_h,
                args.opr->param().pad_w, args.opr->param().stride_h,
                args.opr->param().stride_w));
        cudnn_check(cudnnPoolingForward(
                args.handle->cudnn_handle(), cudnn_desc, &alpha, src_desc.desc,
                src.raw_ptr(), &beta, dst_desc.desc, dst.raw_ptr()));
        cudnn_check(cudnnDestroyPoolingDescriptor(cudnn_desc));
    }
    if (args.layout_src->dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(dst, *args.dst_tensor);
    }
}
#endif

bool PoolingForwardImpl::AlgoCHWN4::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return (args.opr->param().format == Format::CHWN4 &&
            (args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm ||
             args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8));
}

void PoolingForwardImpl::AlgoCHWN4::exec(const ExecArgs& args) const {
    pooling2d::Param kern_param;
    size_t c = (*args.layout_src)[0], hi = (*args.layout_src)[1],
           wi = (*args.layout_src)[2], n = (*args.layout_src)[3],
           ho = (*args.layout_dst)[1], wo = (*args.layout_dst)[2];
    c = c * 4;
    size_t ph = args.opr->param().pad_h, pw = args.opr->param().pad_w;
    size_t window_h = args.opr->param().window_h, window_w = args.opr->param().window_w;
    size_t sh = args.opr->param().stride_h, sw = args.opr->param().stride_w;
    kern_param.n = n, kern_param.c = c, kern_param.hi = hi, kern_param.wi = wi,
    kern_param.ho = ho, kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.window_h = window_h, kern_param.window_w = window_w, kern_param.sh = sh,
    kern_param.sw = sw;
    auto&& stream = cuda_stream(args.handle);
    pooling2d::do_pooling2d_int8_cdiv4hwn4(
            args.src_tensor->compatible_ptr<int8_t>(),
            args.dst_tensor->compatible_ptr<int8_t>(), kern_param, stream,
            static_cast<uint32_t>(args.opr->param().mode));
}

bool PoolingForwardImpl::AlgoNCHW4::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return args.opr->param().format == Format::NCHW4 &&
           (args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm ||
            args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8);
}

void PoolingForwardImpl::AlgoNCHW4::exec(const ExecArgs& args) const {
    pooling2d::Param kern_param;
    size_t n = (*args.layout_src)[0], hi = (*args.layout_src)[2],
           wi = (*args.layout_src)[3], c = (*args.layout_src)[1],
           ho = (*args.layout_dst)[2], wo = (*args.layout_dst)[3];
    c = c * 4;
    size_t ph = args.opr->param().pad_h, pw = args.opr->param().pad_w;
    size_t window_h = args.opr->param().window_h, window_w = args.opr->param().window_w;
    size_t sh = args.opr->param().stride_h, sw = args.opr->param().stride_w;
    kern_param.n = n, kern_param.c = c, kern_param.hi = hi, kern_param.wi = wi,
    kern_param.ho = ho, kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.window_h = window_h, kern_param.window_w = window_w, kern_param.sh = sh,
    kern_param.sw = sw;
    auto&& stream = cuda_stream(args.handle);
    pooling2d::do_pooling2d_int8_ncdiv4hw4(
            args.src_tensor->compatible_ptr<int8_t>(),
            args.dst_tensor->compatible_ptr<int8_t>(), kern_param, stream,
            static_cast<uint32_t>(args.opr->param().mode));
}

bool PoolingForwardImpl::AlgoNCHW32::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return (args.opr->param().format == Format::NCHW32 &&
            (args.layout_src->dtype.enumv() == DTypeEnum::Quantized8Asymm ||
             args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS8));
}

void PoolingForwardImpl::AlgoNCHW32::exec(const ExecArgs& args) const {
    pooling2d::Param kern_param;
    size_t n = (*args.layout_src)[0], hi = (*args.layout_src)[2],
           wi = (*args.layout_src)[3], c = (*args.layout_src)[1],
           ho = (*args.layout_dst)[2], wo = (*args.layout_dst)[3];
    c = c * 32;
    size_t ph = args.opr->param().pad_h, pw = args.opr->param().pad_w;
    size_t window_h = args.opr->param().window_h, window_w = args.opr->param().window_w;
    size_t sh = args.opr->param().stride_h, sw = args.opr->param().stride_w;
    kern_param.n = n, kern_param.c = c, kern_param.hi = hi, kern_param.wi = wi,
    kern_param.ho = ho, kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.window_h = window_h, kern_param.window_w = window_w, kern_param.sh = sh,
    kern_param.sw = sw;
    auto&& stream = cuda_stream(args.handle);
    pooling2d::do_pooling2d_int8_ncdiv32hw32(
            args.src_tensor->compatible_ptr<int8_t>(),
            args.dst_tensor->compatible_ptr<int8_t>(), kern_param, stream,
            static_cast<uint32_t>(args.opr->param().mode));
}

bool PoolingForwardImpl::AlgoNHWC::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return (args.opr->param().format == Format::NHWC &&
            (args.layout_src->dtype.enumv() == DTypeEnum::Quantized4Asymm ||
             args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS4));
}

void PoolingForwardImpl::AlgoNHWC::exec(const ExecArgs& args) const {
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;
    {
        megdnn_assert(
                src.layout.dtype.enumv() == dst.layout.dtype.enumv(),
                "src and dst dtype must equal");
        pooling2d::Param kern_param;
        size_t n = src.layout[0], hi = src.layout[1], wi = src.layout[2],
               c = src.layout[3], ho = dst.layout[1], wo = dst.layout[2];
        size_t ph = args.opr->param().pad_h, pw = args.opr->param().pad_w;
        size_t window_h = args.opr->param().window_h,
               window_w = args.opr->param().window_w;
        size_t sh = args.opr->param().stride_h, sw = args.opr->param().stride_w;
        kern_param.n = n, kern_param.c = c, kern_param.hi = hi, kern_param.wi = wi,
        kern_param.ho = ho, kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
        kern_param.window_h = window_h, kern_param.window_w = window_w,
        kern_param.sh = sh, kern_param.sw = sw;
        bool uint_case = false;
        int zero_point = 0;
        if (src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
            uint_case = true;
            zero_point = src.layout.dtype.param<dtype::Quantized4Asymm>().zero_point;
        }
        auto&& stream = cuda_stream(args.handle);
        pooling2d::do_pooling2d_int4_nhwc(
                (int8_t*)src.raw_ptr(), (int8_t*)dst.raw_ptr(), kern_param, stream,
                static_cast<uint32_t>(args.opr->param().mode), uint_case, zero_point);
    }
}

inline void PoolingForwardImpl::AlgoNCHW64::deduce_reformat_layout(
        std::unique_ptr<RelayoutFormat>& relayout, const TensorLayout& src_layout,
        TensorLayout& dst_layout, RelayoutFormat::Param::Mode mode, const int oc = 0,
        const int group = 1) const {
    if (src_layout.ndim > 0) {
        RelayoutFormat::Param trans_param;
        trans_param.mode = mode;
        trans_param.oc = oc;
        trans_param.group = group;
        relayout->param() = trans_param;
        relayout->deduce_layout(src_layout, dst_layout);
    } else {
        dst_layout = src_layout;
    }
}

void PoolingForwardImpl::AlgoNCHW64::get_inner_layout(
        const TensorLayout& src, const TensorLayout& dst, TensorLayout& inner_src,
        TensorLayout& inner_dst, Handle* handle,
        PoolingForwardImpl::Param::Format format) const {
    auto relayout_opr = handle->create_operator<RelayoutFormat>();
    deduce_reformat_layout(
            relayout_opr, src, inner_src, RelayoutFormat::Param::Mode::NCHW_NCHW64, 0,
            1);
    deduce_reformat_layout(
            relayout_opr, dst, inner_dst, RelayoutFormat::Param::Mode::NCHW_NCHW64, 0,
            1);
}

WorkspaceBundle PoolingForwardImpl::AlgoNCHW64::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    SmallVector<size_t> sizes;
    TensorLayout fsrc = *args.layout_src;
    TensorLayout fdst = *args.layout_dst;
    if (args.opr->param().format == Format::NCHW) {
        get_inner_layout(
                *args.layout_src, *args.layout_dst, fsrc, fdst, args.handle,
                args.opr->param().format);
        sizes.push_back(fsrc.span().dist_byte());
        sizes.push_back(fdst.span().dist_byte());
    }
    return {ptr, std::move(sizes)};
}

bool PoolingForwardImpl::AlgoNCHW64::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
    return ((args.opr->param().format == Format::NCHW ||
             args.opr->param().format == Format::NCHW64) &&
            (args.layout_src->dtype.enumv() == DTypeEnum::QuantizedS4 ||
             args.layout_src->dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
            (args.layout_dst->dtype.enumv() == DTypeEnum::QuantizedS4 ||
             args.layout_dst->dtype.enumv() == DTypeEnum::Quantized4Asymm));
}

void PoolingForwardImpl::AlgoNCHW64::exec(const ExecArgs& args) const {
    using Format = param::Pooling::Format;
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;
    if (args.opr->param().format == Format::NCHW) {
        auto wsb = get_workspace_bundle(args.workspace.raw_ptr, args);
        auto handle_ptr = args.handle;
        get_inner_layout(
                *args.layout_src, *args.layout_dst, src.layout, dst.layout, handle_ptr,
                args.opr->param().format);
        src = TensorND{wsb.get(0), src.layout};
        dst = TensorND{wsb.get(1), dst.layout};
        auto relayout_opr = handle_ptr->create_operator<RelayoutFormat>();
        RelayoutFormat::Param trans_param;
        trans_param.mode = RelayoutFormat::Param::Mode::NCHW_NCHW64;
        relayout_opr->param() = trans_param;
        relayout_opr->exec(*args.src_tensor, src, {});
    }

    {
        pooling2d::Param kern_param;
        size_t n = src.layout[0], hi = src.layout[2], wi = src.layout[3],
               c = src.layout[1], ho = dst.layout[2], wo = dst.layout[3];
        c = c * 64;
        size_t ph = args.opr->param().pad_h, pw = args.opr->param().pad_w;
        size_t window_h = args.opr->param().window_h,
               window_w = args.opr->param().window_w;
        size_t sh = args.opr->param().stride_h, sw = args.opr->param().stride_w;
        kern_param.n = n, kern_param.c = c, kern_param.hi = hi, kern_param.wi = wi,
        kern_param.ho = ho, kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
        kern_param.window_h = window_h, kern_param.window_w = window_w,
        kern_param.sh = sh, kern_param.sw = sw;
        bool uint_case = false;
        int zero_point = 0;
        if (src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
            uint_case = true;
            zero_point = src.layout.dtype.param<dtype::Quantized4Asymm>().zero_point;
        }
        auto&& stream = cuda_stream(args.handle);
        pooling2d::do_pooling2d_int4_ncdiv64hw64(
                (int8_t*)src.raw_ptr(), (int8_t*)dst.raw_ptr(), kern_param, stream,
                static_cast<uint32_t>(args.opr->param().mode), uint_case, zero_point);
    }
    if (args.layout_dst->ndim == 4) {
        auto relayout_opr = args.handle->create_operator<RelayoutFormat>();
        RelayoutFormat::Param trans_param;
        trans_param.mode = RelayoutFormat::Param::Mode::NCHW64_NCHW;
        relayout_opr->param() = trans_param;
        relayout_opr->exec(dst, *args.dst_tensor, {});
    }
}

PoolingBackwardImpl::AlgoPack::AlgoPack() {
    algo_cudnn.push_back({DEF_NAME(cudnnUnreproducible), false});
    algo_cudnn.push_back({DEF_NAME(cudnnReproducible), true});

    for (auto&& i : algo_cudnn) {
        all_algos.push_back(&i);
    }

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

PoolingBackwardImpl::AlgoPack PoolingBackwardImpl::sm_algo_pack;
MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingBackwardImpl)

PoolingBackwardImpl::AlgoBase::SizeArgs::SizeArgs(
        PoolingBackwardImpl* o, const TensorLayout& src, const TensorLayout& dst,
        const TensorLayout& diff, const TensorLayout& grad)
        : handle{concrete_handle(o->handle())},
          opr{o},
          layout_src{&src},
          layout_dst{&dst},
          layout_diff{&diff},
          layout_grad{&grad} {}

PoolingBackwardImpl::AlgoBase::ExecArgs::ExecArgs(
        PoolingBackwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in dst,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, dst.layout, diff.layout, grad.layout),
          src_tensor{&src},
          dst_tensor{&dst},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string PoolingBackwardImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf(
            "src=%s, dst=%s, diff=%s, grad=%s", layout_src->to_string().c_str(),
            layout_dst->to_string().c_str(), layout_diff->to_string().c_str(),
            layout_grad->to_string().c_str());
}

bool PoolingBackwardImpl::AlgoCUDNN::is_available(const SizeArgs& args) const {
    using Format = param::Pooling::Format;
#if CUDNN_VERSION < 6000
    return ((args.opr->param().format == Format::NCHW ||
             args.opr->param().format == Format::NHWC ||
             args.opr->param().format == Format::NCHW4 ||
             args.opr->param().format == Format::NCHW32) &&
            (m_is_reproducible ^
             (args.opr->param().mode == param::Pooling::Mode::MAX)));
#else
    return ((args.opr->param().format == Format::NCHW ||
             args.opr->param().format == Format::NHWC ||
             args.opr->param().format == Format::NCHW4 ||
             args.opr->param().format == Format::NCHW32) &&
            (m_is_reproducible || args.opr->param().mode == param::Pooling::Mode::MAX));
#endif
}

WorkspaceBundle PoolingBackwardImpl::AlgoBase::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    SmallVector<size_t> sizes;
    TensorLayout fsrc = *args.layout_src;
    TensorLayout fdst = *args.layout_dst;
    TensorLayout fdiff = *args.layout_diff;
    TensorLayout fgrad = *args.layout_grad;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
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

size_t PoolingBackwardImpl::AlgoBase::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void PoolingBackwardImpl::AlgoCUDNN::init_mode(
        const ExecArgs& args, cudnnPoolingMode_t& mode) const {
    if (m_is_reproducible) {
        switch (args.opr->param().mode) {
#if CUDNN_VERSION >= 6000
            case param::Pooling::Mode::MAX:
                mode = CUDNN_POOLING_MAX_DETERMINISTIC;
                break;
#endif
            case param::Pooling::Mode::AVERAGE:
                mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                break;
            case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
                mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                break;
            default:
                megdnn_throw(ssprintf(
                        "Unspport pooling mode : {%d}",
                        static_cast<int>(args.opr->param().mode)));
        }
    } else if (args.opr->param().mode == param::Pooling::Mode::MAX) {
        mode = CUDNN_POOLING_MAX;
    } else {
        megdnn_throw("init_mode failed\n");
    }
}

void PoolingBackwardImpl::AlgoCUDNN::exec(const ExecArgs& args) const {
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;
    TensorND diff = *args.diff_tensor;
    TensorND grad = *args.grad_tensor;
    auto wsb = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(args.handle), &wsb);
    if (args.layout_src->dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(*args.src_tensor, src)
                .src_to_comp_type(*args.dst_tensor, dst)
                .src_to_comp_type(*args.diff_tensor, diff)
                .src_to_comp_type(*args.grad_tensor, grad);
    }
    {
        dt_float32 alpha = 1.0f, beta = 0.0f;
        TensorDesc src_desc, dst_desc, diff_desc, grad_desc;
        src_desc.set(src.layout, args.opr->param().format);
        dst_desc.set(dst.layout, args.opr->param().format);
        diff_desc.set(diff.layout, args.opr->param().format);
        grad_desc.set(grad.layout, args.opr->param().format);

        cudnnPoolingMode_t mode;
        init_mode(args, mode);

        cudnnPoolingDescriptor_t cudnn_desc;
        cudnn_check(cudnnCreatePoolingDescriptor(&cudnn_desc));
        cudnn_check(cudnnSetPooling2dDescriptor(
                cudnn_desc, mode, CUDNN_NOT_PROPAGATE_NAN, args.opr->param().window_h,
                args.opr->param().window_w, args.opr->param().pad_h,
                args.opr->param().pad_w, args.opr->param().stride_h,
                args.opr->param().stride_w));
        cudnn_check(cudnnPoolingBackward(
                args.handle->cudnn_handle(), cudnn_desc, &alpha, dst_desc.desc,
                dst.raw_ptr(), diff_desc.desc, diff.raw_ptr(), src_desc.desc,
                src.raw_ptr(), &beta, grad_desc.desc, grad.raw_ptr()));
        cudnn_check(cudnnDestroyPoolingDescriptor(cudnn_desc));
    }
    if (args.layout_src->dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(grad, *args.grad_tensor);
    }
}

// vim: syntax=cpp.doxygen
