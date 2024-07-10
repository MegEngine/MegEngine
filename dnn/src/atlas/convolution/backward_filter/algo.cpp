#include "./algo.h"
#include "aclnnop/aclnn_convolution_backward.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

ConvolutionBackwardFilterImpl::AlgoPack ConvolutionBackwardFilterImpl::sm_algo_pack;

ConvolutionBackwardFilterImpl::AlgoPack::AlgoPack() {
    all_algos.emplace_back(&default_impl);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardFilterImpl)

ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvolutionBackwardFilterImpl* o, const TensorLayout& src,
        const TensorLayout& diff, const TensorLayout& grad)
        : SizeArgs(o, src, diff, grad, o->make_canonized_filter_meta(src.ndim, grad)) {}

ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvolutionBackwardFilterImpl* o, const TensorLayout& src,
        const TensorLayout& diff, const TensorLayout& grad,
        const CanonizedFilterMeta& grad_meta)
        : src_layout{&src},
          diff_layout{&diff},
          grad_layout{&grad},
          grad_filter_meta{grad_meta},
          opr{o} {}

ConvolutionBackwardFilterImpl::AlgoBase::ExecArgs::ExecArgs(
        const ConvolutionBackwardFilterImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, diff.layout, grad.layout),
          src_tensor{&src},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = grad_filter_meta;
    return ssprintf(
            "src=%s diff=%s grad_filter=%u{%u,%u,%u,%u}, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            src_layout->to_string().c_str(), diff_layout->to_string().c_str(), fm.group,
            fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1], fm.padding[0],
            fm.padding[1], fm.stride[0], fm.stride[1], fm.dilation[0], fm.dilation[1],
            !fm.should_flip, src_layout->dtype.name(), diff_layout->dtype.name());
}

bool ConvolutionBackwardFilterImpl::AlgoDefault::is_available(
        const SizeArgs& args) const {
    if (args.grad_layout->dtype.enumv() != args.src_layout->dtype.enumv() ||
        args.grad_layout->dtype.enumv() != args.diff_layout->dtype.enumv()) {
        return false;
    }
    if (args.opr->param().format != param::Convolution::Format::NCHW) {
        return false;
    }
    if (args.opr->param().compute_mode != param::Convolution::ComputeMode::DEFAULT) {
        return false;
    }
    return true;
}

size_t ConvolutionBackwardFilterImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs&) const {
    return 0;
}

void ConvolutionBackwardFilterImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());

    TensorLayout layout = args.grad_tensor->layout;
    if (layout.ndim == 5) {
        size_t* src_shape = layout.shape;
        SmallVector<size_t> target_shape;
        target_shape.push_back(src_shape[0] * src_shape[1]);
        target_shape.push_back(src_shape[2]);
        target_shape.push_back(src_shape[3]);
        target_shape.push_back(src_shape[4]);
        TensorShape target_tensor_shape(target_shape);
        layout.try_reshape(layout, target_tensor_shape);
    }
    TensorND grad_nd(args.grad_tensor->raw_ptr(), layout);

    TensorND fake_filter_tensor(grad_nd);
    AclTensor acl_diff(*(args.diff_tensor), aclFormat::ACL_FORMAT_NCHW),
            acl_src(*(args.src_tensor), aclFormat::ACL_FORMAT_NCHW),
            acl_fake_filter(fake_filter_tensor, aclFormat::ACL_FORMAT_NCHW),
            acl_grad(grad_nd, aclFormat::ACL_FORMAT_NCHW);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    AclIntArray acl_bias_shape({static_cast<int64_t>(args.grad_layout->shape[0])}),
            acl_strides({args.opr->param().stride_h, args.opr->param().stride_w}),
            acl_paddings({args.opr->param().pad_h, args.opr->param().pad_w}),
            acl_dilations({args.opr->param().dilate_h, args.opr->param().dilate_w}),
            acl_dst_paddings({0});
    int64_t group = 1;
    if (args.opr->param().sparse == param::ConvBias::Sparse::GROUP) {
        group = args.grad_filter_meta.group;
    }
    AclBoolArray acl_output_mask({false, true, false});

    //! null filter will introduce error.
    aclnn_check(aclnnConvolutionBackwardGetWorkspaceSize(
            acl_diff.get(), acl_src.get(), acl_fake_filter.get(), acl_bias_shape.get(),
            acl_strides.get(), acl_paddings.get(), acl_dilations.get(), false,
            acl_dst_paddings.get(), group, acl_output_mask.get(), 0, nullptr,
            acl_grad.get(), nullptr, &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(
            aclnnConvolutionBackward(ws.ptr(), ws_size, executor, handle->stream()));
}

// vim: syntax=cpp.doxygen
