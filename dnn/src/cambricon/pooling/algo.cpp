#include "./algo.h"
#include <string.h>
#include <vector>
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_types.h"
#include "src/cambricon/utils.h"

using namespace megdnn;
using namespace cambricon;

PoolingForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_cnnl);

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

size_t PoolingForwardImpl::AlgoCNNL::get_workspace_in_bytes(
        const SizeArgs& args) const {
    size_t workspace_size;
    cnnlHandle_t handle = args.handle->cnnl_handle();
    cnnlPoolingMode_t mode;
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = CNNL_POOLING_MAX;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }

    int out_h, out_w;
    using Format = param::Pooling::Format;
    if (args.opr->param().format == Format::NHWC) {
        out_h = int(args.layout_dst->shape[1]);
        out_w = int(args.layout_dst->shape[2]);
    } else if (args.opr->param().format == Format::NCHW) {
        out_h = int(args.layout_dst->shape[2]);
        out_w = int(args.layout_dst->shape[3]);
    } else {
        megdnn_throw(ssprintf(
                "Unspport pooling format : {%d}",
                static_cast<int>(args.opr->param().format)));
    }

    cnnl_check(
            cnnlGetPoolingWorkspaceSize(handle, mode, out_w, out_h, &workspace_size));
    return workspace_size;
}

bool PoolingForwardImpl::AlgoCNNL::is_available(const SizeArgs& args) const {
    auto src_dtype = args.layout_src->dtype.enumv();
    auto dst_dtype = args.layout_dst->dtype.enumv();
    auto src_layout = args.layout_src;
    auto dst_layout = args.layout_dst;
    size_t pad_h = args.opr->param().pad_h;
    size_t pad_w = args.opr->param().pad_w;
    size_t stride_h = args.opr->param().stride_h;
    size_t stride_w = args.opr->param().stride_w;

    // TODO: add support for other type.
    return src_layout->is_contiguous() && dst_layout->is_contiguous() &&
           (args.opr->param().format == Param::Format::NCHW ||
            args.opr->param().format == Param::Format::NHWC) &&
           (src_layout->shape[0] > 0 && src_layout->shape[1] > 0 &&
            src_layout->shape[2] > 0 && src_layout->shape[3] > 0) &&
           (dst_layout->shape[0] > 0 && dst_layout->shape[1] > 0 &&
            dst_layout->shape[2] > 0 && dst_layout->shape[3] > 0) &&
           (pad_h >= 0 && pad_w >= 0 && stride_h >= 1 && stride_w >= 1) &&
           ((src_dtype == DTypeEnum::Float16 && dst_dtype == DTypeEnum::Float16) ||
            (src_dtype == DTypeEnum::Float32 && dst_dtype == DTypeEnum::Float32));
}

void PoolingForwardImpl::AlgoCNNL::init_mode(
        const ExecArgs& args, cnnlPoolingMode_t& mode) const {
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = CNNL_POOLING_MAX;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }
}

void PoolingForwardImpl::AlgoCNNL::exec(const ExecArgs& args) const {
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;

    cnnlPoolingMode_t mode;
    init_mode(args, mode);

    CnnlPoolingDescriptor cnnl_pooling_desc;
    cnnl_pooling_desc.set2D(
            mode, CNNL_NOT_PROPAGATE_NAN, args.opr->param().window_h,
            args.opr->param().window_w, args.opr->param().pad_h,
            args.opr->param().pad_h, args.opr->param().pad_w, args.opr->param().pad_w,
            args.opr->param().stride_h, args.opr->param().stride_w);

    CnnlTensorDescriptor cnnl_src_desc, cnnl_dst_desc;
    cnnlTensorLayout_t cnnl_layout;
    if (args.opr->param().format == param::Pooling::Format::NHWC) {
        cnnl_layout = CNNL_LAYOUT_NHWC;
    } else if (args.opr->param().format == param::Pooling::Format::NCHW) {
        cnnl_layout = CNNL_LAYOUT_NCHW;
    } else {
        megdnn_throw(ssprintf(
                "Unspport pooling format : {%d}",
                static_cast<int>(args.opr->param().format)));
    }

    cnnl_src_desc.set(&src, cnnl_layout);
    cnnl_dst_desc.set(&dst, cnnl_layout);

    cnnl_check(cnnlPoolingForward(
            args.handle->cnnl_handle(), cnnl_pooling_desc.desc(), nullptr,
            cnnl_src_desc.desc(), src.raw_ptr(), nullptr, cnnl_dst_desc.desc(),
            dst.raw_ptr(), args.workspace.ptr<void>(), args.workspace.size));
}

PoolingBackwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_cnnl);

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

size_t PoolingBackwardImpl::AlgoCNNL::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return 0;
}

bool PoolingBackwardImpl::AlgoCNNL::is_available(const SizeArgs& args) const {
    auto src_dtype = args.layout_src->dtype.enumv();
    auto dst_dtype = args.layout_dst->dtype.enumv();
    auto src_layout = args.layout_src;
    auto dst_layout = args.layout_dst;
    size_t pad_h = args.opr->param().pad_h;
    size_t pad_w = args.opr->param().pad_w;
    size_t stride_h = args.opr->param().stride_h;
    size_t stride_w = args.opr->param().stride_w;
    size_t window_h = args.opr->param().window_h;
    size_t window_w = args.opr->param().window_w;

    auto device_name = concrete_handle(args.handle)->device_info().name;
    if (0 == strcmp(device_name, "MLU590")) {
        if ((src_dtype == DTypeEnum::Float32 && window_h * window_w > 49150) ||
            (src_dtype == DTypeEnum::Float16 && window_h * window_w > 65536)) {
            return false;
        }
    } else {
        if (window_h * window_w > 65536) {
            return false;
        }
    }

    return src_layout->is_contiguous() && dst_layout->is_contiguous() &&
           args.opr->param().format == Param::Format::NHWC &&
           (src_layout->shape[0] > 0 && src_layout->shape[1] > 0 &&
            src_layout->shape[2] > 0 && src_layout->shape[3] > 0) &&
           (dst_layout->shape[0] > 0 && dst_layout->shape[1] > 0 &&
            dst_layout->shape[2] > 0 && dst_layout->shape[3] > 0) &&
           (pad_h >= 0 && pad_w >= 0 && window_h >= 1 && window_w >= 1 &&
            stride_h >= 1 && stride_w >= 1) &&
           ((src_dtype == DTypeEnum::Float16 && dst_dtype == DTypeEnum::Float16) ||
            (src_dtype == DTypeEnum::Float32 && dst_dtype == DTypeEnum::Float32));
}

void PoolingBackwardImpl::AlgoCNNL::init_mode(
        const ExecArgs& args, cnnlPoolingMode_t& mode) const {
    switch (args.opr->param().mode) {
        case param::Pooling::Mode::MAX:
            mode = CNNL_POOLING_MAX;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            break;
        default:
            megdnn_throw(ssprintf(
                    "Unspport pooling mode : {%d}",
                    static_cast<int>(args.opr->param().mode)));
    }
}

void PoolingBackwardImpl::AlgoCNNL::exec(const ExecArgs& args) const {
    TensorND src = *args.src_tensor;
    TensorND dst = *args.dst_tensor;
    TensorND diff = *args.diff_tensor;
    TensorND grad = *args.grad_tensor;

    cnnlPoolingMode_t mode;
    init_mode(args, mode);

    CnnlPoolingDescriptor cnnl_pooling_desc;
    cnnl_pooling_desc.set2D(
            mode, CNNL_NOT_PROPAGATE_NAN, args.opr->param().window_h,
            args.opr->param().window_w, args.opr->param().pad_h,
            args.opr->param().pad_h, args.opr->param().pad_w, args.opr->param().pad_w,
            args.opr->param().stride_h, args.opr->param().stride_w);

    CnnlTensorDescriptor cnnl_src_desc, cnnl_dy_desc, cnnl_dx_desc;
    cnnlTensorLayout_t cnnl_layout;
    if (args.opr->param().format == param::Pooling::Format::NHWC) {
        cnnl_layout = CNNL_LAYOUT_NHWC;
    } else {
        megdnn_throw(ssprintf(
                "Unspport pooling format : {%d}",
                static_cast<int>(args.opr->param().format)));
    }
    cnnlDataType_t cnnl_src_dtype =
            convert_to_cnnl_datatype(args.layout_src->dtype.enumv());
    cnnlDataType_t cnnl_dy_dtype =
            convert_to_cnnl_datatype(args.layout_diff->dtype.enumv());
    cnnlDataType_t cnnl_dx_dtype =
            convert_to_cnnl_datatype(args.layout_grad->dtype.enumv());

    int src_dimNb = int(args.src_tensor->layout.ndim),
        dy_dimNb = int(args.diff_tensor->layout.ndim),
        dx_dimNb = int(args.grad_tensor->layout.ndim);
    std::vector<size_t> src_dimSize(
            args.src_tensor->layout.shape, args.src_tensor->layout.shape + src_dimNb);
    std::vector<size_t> dy_dimSize(
            args.diff_tensor->layout.shape, args.diff_tensor->layout.shape + dy_dimNb);
    std::vector<size_t> dx_dimSize(
            args.grad_tensor->layout.shape, args.grad_tensor->layout.shape + dx_dimNb);
    cnnl_src_desc.set(src_dimNb, src_dimSize, cnnl_src_dtype, cnnl_layout);
    cnnl_dy_desc.set(dy_dimNb, dy_dimSize, cnnl_dy_dtype, cnnl_layout);
    cnnl_dx_desc.set(dx_dimNb, dx_dimSize, cnnl_dx_dtype, cnnl_layout);

    cnnl_check(cnnlPoolingBackward(
            args.handle->cnnl_handle(), cnnl_pooling_desc.desc(), nullptr, nullptr,
            nullptr, cnnl_dy_desc.desc(), diff.raw_ptr(), cnnl_src_desc.desc(),
            src.raw_ptr(), nullptr, cnnl_dx_desc.desc(), grad.raw_ptr()));
}
