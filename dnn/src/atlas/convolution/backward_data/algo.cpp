#include "./algo.h"
#include "aclnnop/aclnn_convolution_backward.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;
ConvolutionBackwardDataImpl::AlgoPack::AlgoPack() {
    all_algos.emplace_back(&default_impl);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl)

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const TensorLayout& diff, const TensorLayout& grad)
        : SizeArgs(
                  o, filter, o->make_canonized_filter_meta(grad.ndim, filter), diff,
                  grad) {}

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const CanonizedFilterMeta& filter_meta, const TensorLayout& diff,
        const TensorLayout& grad)
        : filter_meta{filter_meta},
          diff_layout{&diff},
          grad_layout{&grad},
          filter_layout{&filter},
          opr{o} {}

ConvolutionBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(
        const ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace)
        : SizeArgs(opr, filter.layout, diff.layout, grad.layout),
          filter_tensor{&filter},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    return ssprintf(
            "filter=%u{%u,%u,%u,%u}, diff=%s, grad=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            fm.group, fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1],
            diff_layout->to_string().c_str(), grad_layout->to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1], fm.dilation[0],
            fm.dilation[1], !fm.should_flip, diff_layout->dtype.name(),
            grad_layout->dtype.name());
}

bool ConvolutionBackwardDataImpl::AlgoDefault::is_available(
        const SizeArgs& args) const {
    if (args.grad_layout->dtype.enumv() != args.filter_layout->dtype.enumv() ||
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

size_t ConvolutionBackwardDataImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs&) const {
    return 0;
}

void ConvolutionBackwardDataImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());
    TensorND fake_input_tensor(*(args.grad_tensor));

    TensorLayout layout = args.filter_tensor->layout;
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
    TensorND filter_nd(args.filter_tensor->raw_ptr(), layout);

    AclTensor acl_diff((*args.diff_tensor), aclFormat::ACL_FORMAT_NCHW),
            acl_filter(filter_nd, aclFormat::ACL_FORMAT_NCHW),
            acl_fake_input(fake_input_tensor, aclFormat::ACL_FORMAT_NCHW),
            acl_grad(*(args.grad_tensor), aclFormat::ACL_FORMAT_NCHW);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    AclIntArray acl_bias_shape({static_cast<int64_t>(args.filter_layout->shape[0])}),
            acl_strides({args.opr->param().stride_h, args.opr->param().stride_w}),
            acl_paddings({args.opr->param().pad_h, args.opr->param().pad_w}),
            acl_dilations({args.opr->param().dilate_h, args.opr->param().dilate_w}),
            acl_dst_paddings({0});
    int64_t group = 1;
    if (args.opr->param().sparse == param::ConvBias::Sparse::GROUP) {
        group = args.filter_meta.group;
    }
    AclBoolArray acl_output_mask({true, false, false});

    //! null input will introduce error.
    aclnn_check(aclnnConvolutionBackwardGetWorkspaceSize(
            acl_diff.get(), acl_fake_input.get(), acl_filter.get(),
            acl_bias_shape.get(), acl_strides.get(), acl_paddings.get(),
            acl_dilations.get(), false, acl_dst_paddings.get(), group,
            acl_output_mask.get(), 0, acl_grad.get(), nullptr, nullptr, &ws_size,
            &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(
            aclnnConvolutionBackward(ws.ptr(), ws_size, executor, handle->stream()));
}
// vim: syntax=cpp.doxygen
