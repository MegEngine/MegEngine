#include "src/cambricon/convolution/backward_data/algo.h"
#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"

#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
namespace megdnn {
namespace cambricon {

ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;

ConvolutionBackwardDataImpl::AlgoPack::AlgoPack() {
    all_algos.emplace_back(&default_impl);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

struct ConvolutionBackwardDataCnnlDescs {
    CnnlTensorDescriptor filter_desc, diff_desc, grad_desc;
    CnnlConvolutionDescriptor conv_desc;
    ConvolutionBackwardDataCnnlDescs(
            const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args) {
        cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
        cnnlDataType_t compute_dtype =
                convert_to_cnnl_datatype(args.diff_layout->dtype.enumv());
        // todo: check shape
        filter_desc.set(*args.filter_layout, layout);
        diff_desc.set(*args.diff_layout, layout);
        grad_desc.set(*args.grad_layout, layout);
        conv_desc.set(
                args.filter_meta.stride[0], args.filter_meta.stride[1],
                args.filter_meta.padding[0], args.filter_meta.padding[1],
                args.filter_meta.dilation[0], args.filter_meta.dilation[1],
                args.filter_meta.group, compute_dtype);
    }
};

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const TensorLayout& diff, const TensorLayout& grad)
        : SizeArgs(
                  o, filter, diff, grad,
                  o->make_canonized_filter_meta(grad.ndim, filter)) {}

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const TensorLayout& diff, const TensorLayout& grad,
        const CanonizedFilterMeta& filter_meta)
        : opr{o},
          filter_layout(&filter),
          diff_layout(&diff),
          grad_layout(&grad),
          filter_meta(filter_meta) {}

ConvolutionBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace)
        : SizeArgs(opr, filter.layout, diff.layout, grad.layout),
          filter_tensor{&filter},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto PH = opr->param().pad_h, PW = opr->param().pad_w;
    auto SH = opr->param().stride_h, SW = opr->param().stride_w;
    auto DH = opr->param().dilate_h, DW = opr->param().dilate_w;
    return ssprintf(
            "src=%s, filter=%s, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            diff_layout->to_string().c_str(), filter_layout->to_string().c_str(),
            grad_layout->to_string().c_str(), PH, PW, SH, SW, DH, DW, 1,
            filter_layout->dtype.name(), grad_layout->dtype.name());
}

bool ConvolutionBackwardDataImpl::AlgoDefault::is_available(
        const SizeArgs& args) const {
    if (args.opr->param().format != param::Convolution::Format::NHWC) {
        return false;
    }
    return true;
}

size_t ConvolutionBackwardDataImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());

    ConvolutionBackwardDataCnnlDescs descs(args);
    cnnlConvolutionBwdDataAlgo_t algo;
    cnnl_check(cnnlGetConvolutionBackwardDataAlgorithm(
            handle->cnnl_handle(), descs.filter_desc.desc(), descs.diff_desc.desc(),
            descs.conv_desc.desc(), descs.grad_desc.desc(),
            CNNL_CONVOLUTION_BWD_DATA_FASTEST, &algo));

    size_t workspace_size = 0;
    cnnl_check(cnnlGetConvolutionBackwardDataWorkspaceSize(
            handle->cnnl_handle(), descs.filter_desc.desc(), descs.diff_desc.desc(),
            descs.conv_desc.desc(), descs.grad_desc.desc(), algo, &workspace_size));
    return workspace_size;
}

void ConvolutionBackwardDataImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());

    ConvolutionBackwardDataCnnlDescs descs(args);
    cnnlConvolutionBwdDataAlgo_t algo;
    cnnl_check(cnnlGetConvolutionBackwardDataAlgorithm(
            handle->cnnl_handle(), descs.filter_desc.desc(), descs.diff_desc.desc(),
            descs.conv_desc.desc(), descs.grad_desc.desc(),
            CNNL_CONVOLUTION_BWD_DATA_FASTEST, &algo));

    cnnl_check(cnnlConvolutionBackwardData(
            handle->cnnl_handle(), /*alpha=*/nullptr, descs.filter_desc.desc(),
            args.filter_tensor->raw_ptr(), descs.diff_desc.desc(),
            args.diff_tensor->raw_ptr(), descs.conv_desc.desc(), algo,
            args.workspace.raw_ptr, args.workspace.size, /*beta=*/nullptr,
            descs.grad_desc.desc(), args.grad_tensor->raw_ptr()));
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
