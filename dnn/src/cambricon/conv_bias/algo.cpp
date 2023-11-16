#include "src/cambricon/conv_bias/algo.h"
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

ConvBiasForwardImpl::AlgoPack ConvBiasForwardImpl::sm_algo_pack;

ConvBiasForwardImpl::AlgoPack::AlgoPack() {
    all_algos.emplace_back(&default_conv);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

struct ConvolutionCnnlDescs {
    CnnlTensorDescriptor input_desc, weight_desc, bias_desc, output_desc;
    CnnlConvolutionDescriptor conv_desc;
    ConvolutionCnnlDescs(const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
        cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
        // todo: set `compute_dtype` using compute_mode `param()->compute_mode`
        cnnlDataType_t compute_dtype =
                convert_to_cnnl_datatype(args.dst_layout->dtype.enumv());
        // todo: check shape
        input_desc.set(*args.src_layout, layout);
        weight_desc.set(*args.filter_layout, layout);
        if (args.bias_layout->ndim) {
            bias_desc.set(*args.bias_layout, layout);
        }
        output_desc.set(*args.dst_layout, layout);
        conv_desc.set(
                args.filter_meta.stride[0], args.filter_meta.stride[1],
                args.filter_meta.padding[0], args.filter_meta.padding[1],
                args.filter_meta.dilation[0], args.filter_meta.dilation[1],
                args.filter_meta.group, compute_dtype);
    }
};

using NonlineMode = param::ConvBias::NonlineMode;

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvBiasForwardImpl* opr, const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst)
        : SizeArgs(
                  opr, src, filter, opr->check_layout_fwd(src, filter, dst), bias, z,
                  dst) {}

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvBiasForwardImpl* o, const TensorLayout& src, const TensorLayout& filter,
        const CanonizedFilterMeta& filter_meta, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst)
        : opr{o},
          src_layout(&src),
          filter_layout(&filter),
          filter_meta(filter_meta),
          bias_layout(&bias),
          z_layout(&z),
          dst_layout(&dst) {}

ConvBiasForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvBiasForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in filter,
        _megdnn_tensor_in bias, _megdnn_tensor_in z, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, bias.layout, z.layout, dst.layout),
          src_tensor{&src},
          filter_tensor{&filter},
          bias_tensor{&bias},
          z_tensor{&z},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string ConvBiasForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf(
            "src=%s, filter=%u{%u,%u,%u,%u}, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            src_layout->to_string().c_str(), fm.group, fm.ocpg, fm.icpg, fm.spatial[0],
            fm.spatial[1], dst_layout->to_string().c_str(), fm.padding[0],
            fm.padding[1], fm.stride[0], fm.stride[1], fm.dilation[0], fm.dilation[1],
            !fm.should_flip, src_layout->dtype.name(), dst_layout->dtype.name());
}

bool ConvBiasForwardImpl::AlgoDefault::is_available(const SizeArgs& args) const {
    //! nonlinemode check
    auto nonlinemode = args.opr->param().nonlineMode;
    megdnn_assert(nonlinemode != NonlineMode::H_SWISH, "cambricon Unspports H_SWISH");

    // only support NHWC
    if (args.opr->param().format != param::Convolution::Format::NHWC) {
        return false;
    }
    // todo: check shape
    auto input_dtype = args.src_layout->dtype.enumv();
    auto filter_dtype = args.filter_layout->dtype.enumv();
    auto output_dtype = args.dst_layout->dtype.enumv();
    bool input_dtype_valid = check_dtype_float(input_dtype);
    bool filter_dtype_valid = check_dtype_float(filter_dtype);
    bool output_dtype_valid =
            input_dtype == filter_dtype && input_dtype == output_dtype;

    bool conv_param_valid =
            args.opr->param().mode == param::Convolution::Mode::CROSS_CORRELATION;

    return input_dtype_valid && conv_param_valid && output_dtype_valid &&
           filter_dtype_valid;
}

size_t ConvBiasForwardImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());

    ConvolutionCnnlDescs descs(args);
    cnnlConvolutionForwardAlgo_t algo;
    cnnl_check(cnnlGetConvolutionForwardAlgorithm(
            handle->cnnl_handle(), descs.conv_desc.desc(), descs.input_desc.desc(),
            descs.weight_desc.desc(), descs.output_desc.desc(),
            CNNL_CONVOLUTION_FWD_FASTEST, &algo));

    size_t workspace_size = 0;
    cnnl_check(cnnlGetConvolutionForwardWorkspaceSize(
            handle->cnnl_handle(), descs.input_desc.desc(), descs.weight_desc.desc(),
            descs.output_desc.desc(), descs.bias_desc.desc(), descs.conv_desc.desc(),
            algo, &workspace_size));
    return workspace_size;
}

void ConvBiasForwardImpl::AlgoDefault::exec(const ExecArgs& args) const {
    auto handle = concrete_handle(args.opr->handle());

    ConvolutionCnnlDescs descs(args);
    cnnlConvolutionForwardAlgo_t algo;
    cnnl_check(cnnlGetConvolutionForwardAlgorithm(
            handle->cnnl_handle(), descs.conv_desc.desc(), descs.input_desc.desc(),
            descs.weight_desc.desc(), descs.output_desc.desc(),
            CNNL_CONVOLUTION_FWD_FASTEST, &algo));

    cnnl_check(cnnlConvolutionForward(
            handle->cnnl_handle(), descs.conv_desc.desc(), algo, /*alpha=*/nullptr,
            descs.input_desc.desc(), args.src_tensor->raw_ptr(),
            descs.weight_desc.desc(), args.filter_tensor->raw_ptr(),
            descs.bias_desc.desc(), args.bias_tensor->raw_ptr(), args.workspace.raw_ptr,
            args.workspace.size, /*beta=*/nullptr, descs.output_desc.desc(),
            args.dst_tensor->raw_ptr()));

    //! create elemwise opr
    auto elemwise_opr = handle->create_operator<ElemwiseForward>();

    //! add z
    if (args.z_layout->ndim) {
        elemwise_opr->param().mode = param::Elemwise::Mode::ADD;
        elemwise_opr->exec({*args.dst_tensor, *args.z_tensor}, *args.dst_tensor);
    }

    auto nonlinemode = args.opr->param().nonlineMode;
    //! nonlinear
    if (nonlinemode == NonlineMode::RELU) {
        elemwise_opr->param().mode = param::Elemwise::Mode::RELU;
        elemwise_opr->exec({*args.dst_tensor}, *args.dst_tensor);
    } else if (nonlinemode == NonlineMode::SIGMOID) {
        elemwise_opr->param().mode = param::Elemwise::Mode::SIGMOID;
        elemwise_opr->exec({*args.dst_tensor}, *args.dst_tensor);
    }
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
